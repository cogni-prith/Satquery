"""GeoTIFF ingest: the only place a raster enters the system.

Reads a file with rasterio, maps its bands onto canonical names, infers its modality,
computes its GSD from the affine transform, and returns a fully populated `ImageRef`
alongside the pixel array.

Nothing here guesses. A missing CRS, an unreadable timestamp or an uninterpretable
transform each produce a warning on the `ImageRef` and a `None` field, never a
plausible default. Those warnings ride all the way through to `ToolResult.warnings`,
so a judge can see exactly what was and was not known about the input.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from satquery.io.modality import infer_modality, map_band_names
from satquery.preprocess.gsd import gsd_from_transform
from satquery.serve.contracts import ImageRef
from satquery.utils.logging import get_logger

__all__ = ["parse_timestamp", "read_image_ref", "read_raster", "write_raster"]

_LOG = get_logger(__name__)

# Tag keys that carry an acquisition time, most specific first.
_TIMESTAMP_TAGS: tuple[str, ...] = (
    "ACQUISITION_DATETIME",
    "ACQUISITION_DATE",
    "DATETIME",
    "TIFFTAG_DATETIME",
    "SENSING_TIME",
)

# Formats seen in the wild, in the order we try them.
_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y:%m:%d %H:%M:%S",  # the TIFF standard, colon separated
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%Y%m%d",
    "%Y%m%dT%H%M%S",
)


def parse_timestamp(raw: str | None) -> datetime | None:
    """Parse an acquisition timestamp from a raster tag, or return None.

    Returns None rather than raising: an unparsed timestamp downgrades a bi-temporal
    pair to a warning, which is recoverable, whereas an exception loses the raster.
    """
    if not raw:
        return None
    text = raw.strip()

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None

    if parsed is None:
        for fmt in _TIMESTAMP_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _timestamp_from_tags(tags: dict[str, Any]) -> datetime | None:
    """Find the first parseable acquisition time among a raster's tags."""
    for key in _TIMESTAMP_TAGS:
        parsed = parse_timestamp(tags.get(key))
        if parsed is not None:
            return parsed
    return None


def _build_ref(path: Path, dataset: Any) -> ImageRef:
    """Assemble an `ImageRef` from an open rasterio dataset."""
    warnings: list[str] = []

    band_names, name_warnings = map_band_names(list(dataset.descriptions))
    warnings.extend(name_warnings)

    modality, modality_warnings = infer_modality(band_names, dataset.count)
    warnings.extend(modality_warnings)

    transform: tuple[float, float, float, float, float, float] | None = None
    if dataset.transform is not None:
        transform = tuple(float(v) for v in tuple(dataset.transform)[:6])  # type: ignore[assignment]

    crs_string: str | None = None
    is_geographic = False
    if dataset.crs is None:
        warnings.append(
            "raster has no CRS; footprint comparison against a second image is not possible"
        )
    else:
        crs_string = dataset.crs.to_string()
        is_geographic = bool(dataset.crs.is_geographic)

    centre_latitude: float | None = None
    if is_geographic:
        try:
            bounds = dataset.bounds
            centre_latitude = float((bounds.bottom + bounds.top) / 2.0)
        except (AttributeError, ValueError):  # pragma: no cover - malformed dataset
            centre_latitude = None

    gsd_m, gsd_warnings = gsd_from_transform(
        transform, is_geographic=is_geographic, centre_latitude=centre_latitude
    )
    warnings.extend(gsd_warnings)

    timestamp = _timestamp_from_tags(dataset.tags())
    if timestamp is None:
        warnings.append(
            "no acquisition timestamp in the raster tags; bi-temporal ordering cannot be "
            f"verified from metadata (looked for {', '.join(_TIMESTAMP_TAGS)})"
        )

    return ImageRef(
        path=path,
        modality=modality,
        gsd_m=gsd_m,
        crs=crs_string,
        transform=transform,
        timestamp=timestamp,
        band_names=band_names,
        width=int(dataset.width),
        height=int(dataset.height),
        warnings=warnings,
    )


def read_image_ref(path: str | Path) -> ImageRef:
    """Read only the metadata of a raster. Cheap enough to call on every gate decision."""
    import rasterio

    resolved = Path(path).expanduser().resolve()
    with rasterio.open(resolved) as dataset:
        return _build_ref(resolved, dataset)


def read_raster(path: str | Path) -> tuple[np.ndarray, ImageRef]:
    """Read a GeoTIFF and return `(array, image_ref)`.

    Args:
        path: Path to any raster GDAL can open.

    Returns:
        `array` is `(bands, height, width)` in the file's own band order, matching
        `image_ref.band_names`. It is **not** reordered into the canonical band order;
        call `preprocess.optical.reorder_bands` for that, which needs the names and so
        cannot be wrong about which band is which.
    """
    import rasterio

    resolved = Path(path).expanduser().resolve()
    with rasterio.open(resolved) as dataset:
        array = dataset.read()
        ref = _build_ref(resolved, dataset)

    _LOG.debug(
        "read %s: %s bands %s, modality %s, gsd %s",
        resolved.name,
        dataset.count,
        ref.band_names,
        ref.modality.value,
        ref.gsd_token,
    )
    return array, ref


def write_raster(
    path: str | Path,
    array: np.ndarray,
    *,
    crs: str | None = None,
    transform: tuple[float, float, float, float, float, float] | None = None,
    nodata: float | None = None,
    band_names: list[str] | None = None,
) -> Path:
    """Write a `(bands, h, w)` or `(h, w)` array to a GeoTIFF, carrying georeferencing.

    Used for index maps, masks and overlays, so a judge can open the evidence in QGIS
    on top of the original scene.
    """
    import rasterio
    from affine import Affine

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(array)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise ValueError(f"expected a 2D or 3D array to write, got shape {data.shape}")

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": data.dtype.name,
        "compress": "deflate",
    }
    if crs is not None:
        profile["crs"] = crs
    if transform is not None:
        profile["transform"] = Affine(*transform)
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(resolved, "w", **profile) as dataset:
        dataset.write(data)
        if band_names is not None:
            for index, name in enumerate(band_names[: data.shape[0]], start=1):
                dataset.set_band_description(index, name)

    return resolved
