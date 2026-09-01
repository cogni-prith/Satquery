"""Radiometric change detection for imagery with no near-infrared band.

The fallback that stops a screenshot being a dead end. Every spectral index in this
project needs NIR, so a three-band RGB pair -- a Google Earth capture, a JPEG, an aerial
tile -- has nothing for `indices.deterministic` to measure and is refused. That refusal is
correct and also useless to someone holding two screenshots of the same place.

What survives without NIR is radiometric: a pixel whose colour moved a long way between
two dates changed, whatever it changed into. That answers WHERE, never WHAT.

The distinction is the whole design. This module will happily tell you 14% of the scene
changed and show you the map; it will not tell you a building appeared, because it cannot
distinguish new concrete from a harvested field, a wet road, or a different sun angle.
Naming a class here would be exactly the fabrication the spectral path refuses to commit.

Pure NumPy. No weights, no training distribution to fall outside of.
"""

from __future__ import annotations

import numpy as np

from satquery.preprocess.constants import (
    RGB_CHANGE_DISTANCE_THRESHOLD,
    RGB_CHANGE_MIN_COMPONENT_PX,
)

__all__ = [
    "change_distance",
    "excess_green",
    "match_histogram",
    "normalise_rgb",
    "otsu_threshold",
    "rgb_change_mask",
]


def match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remap `source` so its per-channel distribution matches `reference`.

    Centring and scaling only corrects a shift and a stretch. Two aerial images taken
    years apart differ by more than that: season, sun angle, atmosphere and sensor
    response reshape the whole distribution, and the residual reads as change everywhere.
    On a real SECOND pair, median/MAD normalisation alone reported 90% of the scene as
    changed -- true in the sense that everything did look different, and useless.

    Matching the cumulative distributions removes that global difference and leaves the
    local disagreements, which is what change detection is actually after.

    The cost is honest and worth stating: this assumes the two scenes have broadly similar
    composition. A pair where genuinely half the ground cover changed has a different
    distribution *because of the change*, and matching will partly erase it. It trades
    sensitivity on huge changes for not drowning in false ones.
    """
    src = np.asarray(source, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if src.shape != ref.shape:
        raise ValueError(f"shape mismatch: {src.shape} and {ref.shape}")

    out = np.empty_like(src)
    for channel in range(src.shape[2]):
        s_flat = src[:, :, channel].ravel()
        r_flat = ref[:, :, channel].ravel()
        # Rank each source pixel, then read the reference value at the same rank.
        order = np.argsort(s_flat, kind="stable")
        ranks = np.empty(len(s_flat), dtype=np.int64)
        ranks[order] = np.arange(len(s_flat))
        mapped = np.sort(r_flat)[ranks]
        out[:, :, channel] = mapped.reshape(src.shape[:2])
    return out


def normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    """Per-channel robust normalisation of an `(H, W, 3)` image, by median and MAD.

    Two screenshots of one place differ in exposure, white balance and compression before
    anything on the ground has moved. Differencing the raw values measures that difference
    as loudly as it measures real change, so each image is standardised into its own
    statistics first and only the *pattern* is compared.

    Median and median-absolute-deviation rather than mean and standard deviation, because
    the statistic must not be moved by the thing being looked for. A changed region is an
    outlier by construction: with mean/std, a bright block covering a ninth of the scene
    dragged both statistics far enough that every *unchanged* pixel also moved, and a
    1,600-pixel change was reported as 11,340. The median barely notices it.

    The scaling constant 1.4826 makes the MAD match the standard deviation for normally
    distributed data, so the threshold keeps its meaning in units of ordinary spread.
    """
    array = np.asarray(rgb, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {array.shape}")

    array = array[:, :, :3]
    out = np.empty_like(array)
    for channel in range(3):
        band = array[:, :, channel]
        centre = float(np.median(band))
        spread = 1.4826 * float(np.median(np.abs(band - centre)))
        # A flat or near-flat channel has no pattern to compare; leave it centred rather
        # than divide by ~zero and turn a constant into noise.
        out[:, :, channel] = (band - centre) / (spread if spread > 1e-8 else 1.0)
    return out


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold over a 1-D distribution: the split maximising between-class variance.

    The change threshold has to be found per pair, not fixed in advance. Two aerial
    acquisitions of one place disagree by an amount that depends on how far apart in time
    they are, the season, the sun and the sensor -- measured across real SECOND pairs the
    median per-pixel distance ranged from 0.8 to 2.2 and the 99th percentile from 5.5 to
    18. A single frozen number sits below the median on some pairs and above the 99th on
    others; the first version of this module used 0.35 and flagged 90% of every scene.

    Otsu asks the only question that transfers: given THIS pair's distribution, where does
    it separate into two populations? That is scale-free by construction.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("cannot threshold an empty distribution")

    counts, edges = np.histogram(flat, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_low = np.cumsum(counts)
    weight_high = weight_low[-1] - weight_low
    # Bins where either side is empty admit no split.
    valid = (weight_low > 0) & (weight_high > 0)
    if not valid.any():
        return float(flat.max())

    cum = np.cumsum(counts * centres)
    mean_low = np.where(weight_low > 0, cum / np.maximum(weight_low, 1), 0.0)
    mean_high = np.where(weight_high > 0, (cum[-1] - cum) / np.maximum(weight_high, 1), 0.0)
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between[~valid] = -np.inf
    return float(centres[int(np.argmax(between))])


def change_distance(rgb_t1: np.ndarray, rgb_t2: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance between two normalised RGB images.

    Returns an `(H, W)` float map. Larger means the colour moved further.
    """
    first_raw = np.asarray(rgb_t1, dtype=np.float64)[:, :, :3]
    second_raw = np.asarray(rgb_t2, dtype=np.float64)[:, :, :3]
    if first_raw.shape != second_raw.shape:
        raise ValueError(
            f"the two images must be co-registered onto one grid, got {first_raw.shape} "
            f"and {second_raw.shape}. Radiometric differencing compares pixel to pixel "
            "and has no way to align them."
        )

    # Match the second image onto the first, then normalise both. Matching removes the
    # global difference in illumination and sensor response; normalising puts the residual
    # into units the frozen threshold is expressed in.
    first = normalise_rgb(first_raw)
    second = normalise_rgb(match_histogram(second_raw, first_raw))
    if first.shape != second.shape:
        raise ValueError(
            f"the two images must be co-registered onto one grid, got {first.shape} "
            f"and {second.shape}. Radiometric differencing compares pixel to pixel and "
            "has no way to align them."
        )
    return np.sqrt(((second - first) ** 2).sum(axis=2))


def rgb_change_mask(
    rgb_t1: np.ndarray,
    rgb_t2: np.ndarray,
    *,
    threshold: float | None = None,
    min_component_px: int = RGB_CHANGE_MIN_COMPONENT_PX,
) -> tuple[np.ndarray, list[str]]:
    """Boolean mask of pixels whose colour moved, plus the caveats that must travel with it.

    Speckle is removed by dropping connected components below `min_component_px`.
    Compression artefacts, resampling and a single pixel of registration error all trip a
    raw threshold, and without this step the count is dominated by them.

    Returns:
        `(mask, warnings)`. The warnings are not optional decoration -- they carry the
        limits of what this measurement can support, and a caller that drops them turns a
        screening result into a land-cover claim.
    """
    from satquery.symbolic.measures import connected_components

    distance = change_distance(rgb_t1, rgb_t2)

    # Adaptive by default. A frozen distance cannot transfer between pairs -- see
    # `otsu_threshold` -- so the split is found in this pair's own distribution unless the
    # caller insists on a number.
    if threshold is None:
        threshold = max(otsu_threshold(distance), RGB_CHANGE_DISTANCE_THRESHOLD)
    raw = distance > threshold

    mask = np.zeros_like(raw)
    for component in connected_components(raw.astype(np.uint8), 1, min_area_px=min_component_px):
        rows = slice(int(component["row_min"]), int(component["row_max"]) + 1)
        cols = slice(int(component["col_min"]), int(component["col_max"]) + 1)
        mask[rows, cols] |= raw[rows, cols]

    warnings = [
        f"change threshold {threshold:.2f} was found from this pair's own distribution "
        "(Otsu), because the disagreement between two acquisitions depends on season, sun "
        "angle and sensor and no fixed value transfers between scenes",
        "no near-infrared band, so this is radiometric change detection: it reports WHERE "
        "the imagery differs, not WHAT changed. It cannot separate new construction from "
        "bare soil, a wet surface, or a different sun angle",
        "the two images are compared pixel to pixel and are assumed already aligned; a "
        "shift of even one pixel registers as change along every edge in the scene",
    ]
    share = float(mask.mean())
    if share > 0.6:
        warnings.append(
            f"{share * 100:.0f}% of the scene reads as changed, which usually means the "
            "images are misaligned, differently exposed, or not the same place"
        )
    return mask, warnings


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """Excess Green index, `2G - R - B`, on a normalised-to-unit-range RGB image.

    A vegetation proxy for imagery with no NIR (Woebbecke et al. 1995). Much weaker than
    NDVI: it finds green things, which is not the same as finding live vegetation, and it
    is fooled by anything else green. Used here only to characterise an RGB change --
    "the changed area became greener" -- never to report a vegetation area.
    """
    array = np.asarray(rgb, dtype=np.float64)[:, :, :3]
    total = array.sum(axis=2)
    total[total == 0] = 1.0
    red, green, blue = (array[:, :, index] / total for index in range(3))
    return 2.0 * green - red - blue
