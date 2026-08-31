"""Synthetic in-memory rasters for the test suite.

No real satellite data is used in tests. Everything here is generated deterministically
from a fixed seed so a failure is always reproducible, and every scene has a known
ground truth (where the water is, where the vegetation is) so the assertions can check
values rather than just shapes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from satquery.io.raster import write_raster

__all__ = [
    "T1",
    "T2",
    "UTM_43N",
    "WGS84",
    "north_up_transform",
    "optical_stack",
    "sar_stack",
    "write_optical",
    "write_sar",
]

UTM_43N = "EPSG:32643"
WGS84 = "EPSG:4326"

T1 = datetime(2024, 1, 15, 5, 30, tzinfo=timezone.utc)
T2 = datetime(2024, 7, 20, 5, 30, tzinfo=timezone.utc)

OPTICAL_10_BANDS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12")


def north_up_transform(
    gsd_m: float, origin_x: float = 600000.0, origin_y: float = 2000000.0
) -> tuple[float, float, float, float, float, float]:
    """A standard north-up affine at the given ground sampling distance."""
    return (gsd_m, 0.0, origin_x, 0.0, -gsd_m, origin_y)


def optical_stack(
    size: int = 32, water_cols: int = 8, seed: int = 0
) -> tuple[np.ndarray, tuple[str, ...]]:
    """A 10-band multispectral scene with a known water strip and a vegetated remainder.

    Water occupies the leftmost `water_cols` columns: high green, near-zero NIR, which
    is exactly what NDWI keys on. The rest is vegetation: high NIR, low red.
    """
    rng = np.random.default_rng(seed)
    stack = rng.uniform(0.05, 0.35, size=(len(OPTICAL_10_BANDS), size, size)).astype("float32")
    index = OPTICAL_10_BANDS.index

    stack[index("B03"), :, :water_cols] = 0.22
    stack[index("B08"), :, :water_cols] = 0.02
    stack[index("B08"), :, water_cols:] = 0.45
    stack[index("B04"), :, water_cols:] = 0.06
    return stack, OPTICAL_10_BANDS


def sar_stack(
    size: int = 32, water_cols: int = 8, seed: int = 1, looks: float = 4.0
) -> tuple[np.ndarray, tuple[str, ...]]:
    """A dual-pol SAR scene in linear amplitude with gamma speckle and a dark water strip."""
    rng = np.random.default_rng(seed)
    speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=(size, size))
    vv = np.full((size, size), 0.30)
    vv[:, :water_cols] = 0.004
    vh = vv * 0.35
    return np.stack([vv * speckle, vh * speckle]).astype("float32"), ("VV", "VH")


def _stamp(path: Path, when: datetime | None) -> Path:
    """Write an acquisition timestamp into the raster tags, as a real product carries."""
    if when is None:
        return path
    import rasterio

    with rasterio.open(path, "r+") as dataset:
        dataset.update_tags(ACQUISITION_DATETIME=when.strftime("%Y-%m-%dT%H:%M:%S"))
    return path


def write_optical(
    path: Path,
    *,
    gsd_m: float = 10.0,
    size: int = 32,
    water_cols: int = 8,
    crs: str | None = UTM_43N,
    when: datetime | None = T1,
    seed: int = 0,
) -> Path:
    """Write a synthetic multispectral GeoTIFF and return its path."""
    stack, names = optical_stack(size=size, water_cols=water_cols, seed=seed)
    write_raster(path, stack, crs=crs, transform=north_up_transform(gsd_m), band_names=list(names))
    return _stamp(path, when)


def write_sar(
    path: Path,
    *,
    gsd_m: float = 10.0,
    size: int = 32,
    water_cols: int = 8,
    crs: str | None = UTM_43N,
    when: datetime | None = T1,
    single_pol: bool = False,
    seed: int = 1,
) -> Path:
    """Write a synthetic SAR GeoTIFF and return its path."""
    stack, names = sar_stack(size=size, water_cols=water_cols, seed=seed)
    if single_pol:
        stack, names = stack[:1], names[:1]
    write_raster(path, stack, crs=crs, transform=north_up_transform(gsd_m), band_names=list(names))
    return _stamp(path, when)
