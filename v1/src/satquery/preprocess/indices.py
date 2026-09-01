"""Closed-form spectral and backscatter indices. The project's safety net.

These functions have no weights, no training distribution and therefore nothing to
fall outside of. When the learned model is uncertain on unseen sensor data -- and a
20x resolution gap between 10 m Sentinel training tiles and sub-metre Cartosat-2S
evaluation imagery guarantees some uncertainty -- index agreement still produces a
defensible answer with a visible map.

Agreement between the learned output and the index output is the confidence signal
reported on `ToolResult.confidence`; see `agreement_score`.

Definitions, all with an eps-guarded denominator:

- NDWI = (Green - NIR) / (Green + NIR)      McFeeters 1996
- NDBI = (SWIR - NIR) / (SWIR + NIR)        Zha, Gao and Ni 2003
- NDVI = (NIR - Red) / (NIR + Red)          Rouse et al. 1974
"""

from __future__ import annotations

import numpy as np

from satquery.preprocess.constants import (
    BAND_ROLE_GREEN,
    BAND_ROLE_NIR,
    BAND_ROLE_RED,
    BAND_ROLE_SWIR,
    INDEX_EPS,
    NDBI_BUILTUP_THRESHOLD,
    NDVI_VEGETATION_THRESHOLD,
    NDWI_WATER_THRESHOLD,
    SAR_WATER_DB_THRESHOLD,
)
from satquery.preprocess.sar import linear_to_db

__all__ = [
    "INDEX_BAND_ROLES",
    "agreement_score",
    "builtup_mask",
    "compute_indices",
    "ndbi",
    "ndvi",
    "ndwi",
    "normalised_difference",
    "sar_water_mask",
    "vegetation_mask",
    "water_mask",
]

#: Which canonical bands each index needs. Used to decide what is computable from a
#: given stack without hardcoding band positions anywhere.
INDEX_BAND_ROLES: dict[str, tuple[str, ...]] = {
    "ndwi": (BAND_ROLE_GREEN, BAND_ROLE_NIR),
    "ndbi": (BAND_ROLE_SWIR, BAND_ROLE_NIR),
    "ndvi": (BAND_ROLE_NIR, BAND_ROLE_RED),
}


def normalised_difference(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    eps: float = INDEX_EPS,
) -> np.ndarray:
    """Return ``(positive - negative) / (positive + negative)``, eps-guarded and clipped.

    Where the denominator is smaller than `eps` in magnitude -- two bands that are both
    zero, typically a nodata pixel -- the result is 0.0 rather than an infinity. The
    output is clipped to ``[-1, 1]``, the mathematical range for non-negative
    reflectances, so a negative reflectance from an over-corrected atmosphere cannot
    produce an index of 40.
    """
    a = np.asarray(positive, dtype=np.float64)
    b = np.asarray(negative, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"band shapes must match, got {a.shape} and {b.shape}")

    denominator = a + b
    usable = np.abs(denominator) >= eps
    safe = np.where(usable, denominator, 1.0)
    result = np.where(usable, (a - b) / safe, 0.0)
    return np.clip(result, -1.0, 1.0)


def ndwi(green: np.ndarray, nir: np.ndarray, *, eps: float = INDEX_EPS) -> np.ndarray:
    """Normalised Difference Water Index. High over open water."""
    return normalised_difference(green, nir, eps=eps)


def ndbi(swir: np.ndarray, nir: np.ndarray, *, eps: float = INDEX_EPS) -> np.ndarray:
    """Normalised Difference Built-up Index. High over impervious surfaces."""
    return normalised_difference(swir, nir, eps=eps)


def ndvi(nir: np.ndarray, red: np.ndarray, *, eps: float = INDEX_EPS) -> np.ndarray:
    """Normalised Difference Vegetation Index. High over healthy vegetation."""
    return normalised_difference(nir, red, eps=eps)


def sar_water_mask(
    vv: np.ndarray,
    *,
    in_db: bool = False,
    threshold_db: float = SAR_WATER_DB_THRESHOLD,
) -> np.ndarray:
    """Threshold SAR co-polarised backscatter to find smooth open water.

    Calm water is a specular reflector: it bounces the radar pulse away from the
    sensor rather than back to it, so it appears very dark. This is the one water
    indicator that works through cloud and at night, which is why the optical and SAR
    answers are cross-checked rather than either being trusted alone.

    Args:
        vv: Co-polarised channel (VV, or HH on RISAT).
        in_db: True when `vv` is already in decibels. When False it is converted with
            the same frozen `linear_to_db` the rendering pipeline uses.
        threshold_db: Backscatter below this is classified as water.

    Returns:
        Boolean array, True where water is indicated.
    """
    values = np.asarray(vv, dtype=np.float64)
    decibels = values if in_db else linear_to_db(values)
    return decibels < threshold_db


def water_mask(ndwi_map: np.ndarray, *, threshold: float = NDWI_WATER_THRESHOLD) -> np.ndarray:
    """Boolean water mask from an NDWI map."""
    return np.asarray(ndwi_map) > threshold


def builtup_mask(ndbi_map: np.ndarray, *, threshold: float = NDBI_BUILTUP_THRESHOLD) -> np.ndarray:
    """Boolean built-up mask from an NDBI map."""
    return np.asarray(ndbi_map) > threshold


def vegetation_mask(
    ndvi_map: np.ndarray, *, threshold: float = NDVI_VEGETATION_THRESHOLD
) -> np.ndarray:
    """Boolean vegetation mask from an NDVI map."""
    return np.asarray(ndvi_map) > threshold


def compute_indices(
    stack: np.ndarray,
    band_names: list[str] | tuple[str, ...],
    which: list[str] | tuple[str, ...] = ("ndwi", "ndbi", "ndvi"),
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Compute the requested indices from a band stack, by band name.

    Bands are looked up by their canonical name, never by position, so a stack that
    was never reordered still produces correct indices.

    Args:
        stack: `(bands, height, width)` array.
        band_names: Canonical names in stored order, from `io/modality.py`.
        which: Index names to compute. Unknown names raise; known-but-uncomputable
            ones are skipped with a warning.

    Returns:
        `(maps, warnings)`. `maps` holds only the indices that could actually be
        computed. An index whose bands are absent is omitted, never faked with zeros.
    """
    array = np.asarray(stack, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D (bands, h, w) stack, got shape {array.shape}")
    if array.shape[0] != len(band_names):
        raise ValueError(
            f"band count mismatch: array has {array.shape[0]} bands but "
            f"{len(band_names)} names were given"
        )

    position = {name: index for index, name in enumerate(band_names)}
    maps: dict[str, np.ndarray] = {}
    warnings: list[str] = []

    for name in which:
        if name not in INDEX_BAND_ROLES:
            raise ValueError(f"unknown index {name!r}; expected one of {sorted(INDEX_BAND_ROLES)}")
        required = INDEX_BAND_ROLES[name]
        missing = [band for band in required if band not in position]
        if missing:
            warnings.append(
                f"{name.upper()} not computed: missing band(s) {missing}. "
                f"Available bands: {sorted(position)}"
            )
            continue
        first, second = (array[position[band]] for band in required)
        maps[name] = normalised_difference(first, second)

    return maps, warnings


def agreement_score(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over union of two boolean masks, used as the confidence signal.

    When the learned model and the deterministic index agree on where the water is,
    the answer is trustworthy. When they disagree, the reported confidence drops and
    both maps are returned so a human can adjudicate.

    Two empty masks agree perfectly and score 1.0: neither found anything, which is
    consistent, not uninformative.
    """
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes must match, got {a.shape} and {b.shape}")

    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)
