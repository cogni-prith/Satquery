"""The frozen SAR rendering pipeline. Raw SAR is never fed to a model.

Sentinel-1 at training time and RISAT at evaluation time go through this identical
function, in this identical order:

1. Refined Lee speckle filter, 7x7 window
2. Linear amplitude to decibels, ``10 * log10(x + 1e-6)``
3. Per-image percentile stretch, 2nd to 98th, clipped to ``[0, 255]``
4. Pseudo-RGB layout: R = VV, G = VH, B = VV/VH ratio in dB
5. Single-polarisation input: R = G = the available polarisation, B = zeros, warning

Step ordering note: the ratio channel is built from the dB values produced by step 2,
before the stretch of step 3 is applied. A ratio of two already-stretched channels
would be a ratio of display values rather than of backscatter, which is physically
meaningless. All three channels are then stretched independently.

RISAT commonly delivers HH and HV rather than VV and VH. They occupy the same slots:
the co-polarised channel goes to R, the cross-polarised channel to G. That mapping is
made in `io/modality.py`; this module just takes the two arrays.

`render_sar` returns a channels-last ``(height, width, 3)`` uint8 image, because its
output is a picture handed to a model or written to disk, not a band stack. Everything
else in `preprocess/` is channels-first.
"""

from __future__ import annotations

import numpy as np

from satquery.preprocess.constants import (
    DB_EPS,
    DB_SCALE,
    SAR_SINGLE_POL_WARNING,
    SPECKLE_FILTER_NUM_LOOKS,
    SPECKLE_FILTER_WINDOW,
    SPECKLE_SUBWINDOW,
    STRETCH_PERCENTILES,
)
from satquery.preprocess.optical import percentile_stretch

__all__ = [
    "db_to_linear",
    "directional_masks",
    "linear_to_db",
    "refined_lee",
    "render_sar",
]


def directional_masks(window: int = SPECKLE_FILTER_WINDOW) -> np.ndarray:
    """Return the eight edge-aligned half-window masks used by the Refined Lee filter.

    The Refined Lee filter's whole idea is that a square window straddling an edge
    mixes two populations. It instead detects the dominant edge orientation and
    filters using only the half of the window on the centre pixel's side.

    Masks are derived geometrically rather than hardcoded: for each of four edge
    orientations there is a unit normal, and a window offset belongs to a half-window
    when its projection onto that normal is non-negative. Four orientations times two
    signs gives the eight directions, and every mask contains the centre pixel.

    Returns:
        Boolean array of shape ``(8, window, window)``. Index ``orientation * 2 + side``,
        where side 0 is the positive-normal half and side 1 the negative-normal half.
    """
    if window % 2 == 0:
        raise ValueError(f"speckle window must be odd, got {window}")

    radius = window // 2
    offsets_y, offsets_x = np.mgrid[-radius : radius + 1, -radius : radius + 1]

    # Four edge orientations, as the normal to the edge. Order matches _GRADIENT_PAIRS.
    normals = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (1.0, -1.0))

    masks = []
    for normal_y, normal_x in normals:
        projection = offsets_y * normal_y + offsets_x * normal_x
        masks.append(projection >= 0.0)
        masks.append(projection <= 0.0)
    return np.stack(masks)


# For each orientation, the (negative-side, positive-side) sub-block coordinates in the
# 3x3 grid of sub-block means. Aligned with the `normals` tuple above.
_GRADIENT_PAIRS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((1, 0), (1, 2)),  # normal (0, +1):  horizontal gradient, vertical edge
    ((0, 0), (2, 2)),  # normal (+1, +1): main-diagonal gradient
    ((0, 1), (2, 1)),  # normal (+1, 0):  vertical gradient, horizontal edge
    ((0, 2), (2, 0)),  # normal (+1, -1): anti-diagonal gradient
)


def _shift(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift with edge replication, so sampling never wraps around the raster."""
    shifted = np.roll(array, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        shifted[:dy, :] = shifted[dy : dy + 1, :]
    elif dy < 0:
        shifted[dy:, :] = shifted[dy - 1 : dy, :]
    if dx > 0:
        shifted[:, :dx] = shifted[:, dx : dx + 1]
    elif dx < 0:
        shifted[:, dx:] = shifted[:, dx - 1 : dx]
    return shifted


def refined_lee(
    image: np.ndarray,
    *,
    window: int = SPECKLE_FILTER_WINDOW,
    num_looks: float = SPECKLE_FILTER_NUM_LOOKS,
) -> np.ndarray:
    """Refined Lee speckle filter (Lee 1981), as implemented in ESA SNAP.

    Speckle is multiplicative noise with a known coefficient of variation
    ``Cu = 1/sqrt(L)`` for ``L`` looks. A plain square window straddling an edge mixes
    two populations, so the filter first finds the dominant edge direction from a 3x3
    grid of sub-block means, selects the half-window on the centre pixel's own side of
    that edge, and forms a minimum-mean-square-error estimate from that half alone::

          varX = (varZ - meanZ^2 * Cu^2) / (1 + Cu^2)
          b    = clip(varX / varZ, 0, 1)
          out  = meanZ + b * (centre - meanZ)

    In a homogeneous neighbourhood the observed variance is entirely speckle, ``varX``
    collapses to zero, ``b`` goes to zero and the output is the local mean -- full
    smoothing. Across an edge the selected half-window is single-population, so the
    step survives. Both behaviours fall out of the same expression; there is
    deliberately no separate flat-versus-edge branch, matching the widely used SNAP
    and Google Earth Engine formulations. An explicit branch keyed on the local
    coefficient of variation classifies even a 1-to-100 step as flat whenever the
    window sits mostly on one side of it, and smears exactly the edges this filter
    exists to preserve.

    Fully vectorised: the per-direction statistics are nine correlations, not a Python
    loop over pixels.

    Args:
        image: 2D array of linear amplitude or intensity. Must be non-negative.
        window: Filter window, odd. Frozen at 7.
        num_looks: Equivalent number of looks. Frozen; see `constants.py`.

    Returns:
        Filtered array, float64, same shape as the input.
    """
    from scipy.ndimage import correlate, uniform_filter

    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"refined_lee operates on a single 2D band, got shape {values.shape}")
    if window % 2 == 0:
        raise ValueError(f"speckle window must be odd, got {window}")
    if num_looks <= 0.0:
        raise ValueError(f"number of looks must be positive, got {num_looks}")

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    cu_squared = 1.0 / num_looks

    # -- dominant edge direction from a 3x3 grid of sub-block means
    step = SPECKLE_SUBWINDOW - 1 if window >= 2 * SPECKLE_SUBWINDOW - 1 else 1
    sub_mean = uniform_filter(values, size=SPECKLE_SUBWINDOW, mode="reflect")
    grid = {
        (i, j): _shift(sub_mean, (1 - i) * step, (1 - j) * step) for i in range(3) for j in range(3)
    }
    centre_block = grid[(1, 1)]

    gradients = np.stack(
        [np.abs(grid[positive] - grid[negative]) for negative, positive in _GRADIENT_PAIRS]
    )
    orientation = np.argmax(gradients, axis=0)

    # Side selection: keep the half whose sub-block mean is closer to the centre block,
    # which is the half the centre pixel actually belongs to.
    side_is_negative = np.zeros_like(orientation, dtype=bool)
    for index, (negative, positive) in enumerate(_GRADIENT_PAIRS):
        closer_to_negative = np.abs(grid[negative] - centre_block) < np.abs(
            grid[positive] - centre_block
        )
        side_is_negative = np.where(orientation == index, closer_to_negative, side_is_negative)
    direction = orientation * 2 + side_is_negative.astype(np.int64)

    # -- per-direction masked statistics
    masks = directional_masks(window)
    means = np.empty((masks.shape[0], *values.shape), dtype=np.float64)
    variances = np.empty_like(means)
    for index, mask in enumerate(masks):
        kernel = mask.astype(np.float64)
        kernel /= kernel.sum()
        mean = correlate(values, kernel, mode="reflect")
        mean_sq = correlate(values * values, kernel, mode="reflect")
        means[index] = mean
        variances[index] = np.maximum(mean_sq - mean * mean, 0.0)

    selector = direction[np.newaxis, ...]
    mean_z = np.take_along_axis(means, selector, axis=0)[0]
    var_z = np.take_along_axis(variances, selector, axis=0)[0]

    # -- minimum mean square error estimate on the selected half-window
    safe_var_z = np.where(var_z > 0.0, var_z, 1.0)
    var_x = (var_z - mean_z * mean_z * cu_squared) / (1.0 + cu_squared)
    weight = np.clip(var_x / safe_var_z, 0.0, 1.0)
    weight = np.where(var_z > 0.0, weight, 0.0)
    return mean_z + weight * (values - mean_z)


def linear_to_db(values: np.ndarray, *, eps: float = DB_EPS, scale: float = DB_SCALE) -> np.ndarray:
    """Convert linear amplitude or intensity to decibels: ``scale * log10(x + eps)``.

    The epsilon is inside the logarithm, not added afterwards, so a zero-backscatter
    pixel produces a large negative finite number rather than ``-inf``.
    """
    array = np.asarray(values, dtype=np.float64)
    return scale * np.log10(np.maximum(array, 0.0) + eps)


def db_to_linear(values: np.ndarray, *, eps: float = DB_EPS, scale: float = DB_SCALE) -> np.ndarray:
    """Invert `linear_to_db`: recover linear backscatter from decibels.

    Some sources deliver SAR already converted to dB -- reBEN's Sentinel-1 patches do,
    while raw Sentinel-1 GRD and RISAT do not. The frozen pipeline in `render_sar`
    begins at linear amplitude, so a dB source must be inverted before entering it.

    Inverting is the correct move rather than skipping step 2, and the reason is
    physical, not cosmetic. Speckle is *multiplicative* noise in the linear domain,
    which is the model the Refined Lee filter is derived from; taking the logarithm
    makes it additive. Running that filter on dB values would apply an
    minimum-mean-square-error estimator built for one noise model to data following
    another, quietly degrading exactly the edges the filter exists to preserve.

    The `eps` added inside the logarithm by `linear_to_db` is subtracted here, so the
    round trip is exact to floating-point precision rather than off by `eps`. Results
    are clamped at zero because a negative backscatter is not physical.

    Args:
        values: Array in decibels.
        eps: Must match the epsilon used on the forward conversion.
        scale: Must match the scale used on the forward conversion.

    Returns:
        Linear array, float64, same shape as the input.
    """
    array = np.asarray(values, dtype=np.float64)
    return np.maximum(np.power(10.0, array / scale) - eps, 0.0)


def render_sar(
    vv: np.ndarray | None,
    vh: np.ndarray | None = None,
    *,
    apply_speckle_filter: bool = True,
    percentiles: tuple[float, float] = STRETCH_PERCENTILES,
) -> tuple[np.ndarray, list[str]]:
    """Render raw SAR polarisations to the frozen pseudo-RGB image.

    This is the only sanctioned way to turn SAR into something a model sees. Both
    Sentinel-1 at training time and RISAT at evaluation time go through it unchanged.

    Args:
        vv: Co-polarised channel (VV, or HH on RISAT) in linear amplitude or intensity.
        vh: Cross-polarised channel (VH, or HV). ``None`` triggers the single-pol path.
        apply_speckle_filter: Run step 1. Only ever disabled in tests that isolate a
            later step; never disable it on a real inference path.
        percentiles: Stretch percentiles. Frozen; exposed only for tests.

    Returns:
        ``(image, warnings)`` with ``image`` of shape ``(height, width, 3)``, dtype
        uint8, channels ordered R, G, B per the frozen layout.

    Raises:
        ValueError: Neither polarisation was supplied, or the two differ in shape.
    """
    warnings: list[str] = []

    if vv is None and vh is None:
        raise ValueError("render_sar needs at least one polarisation; both were None")

    single_pol = vv is None or vh is None
    if single_pol:
        available = vv if vv is not None else vh
        assert available is not None  # narrowed by the check above
        band = np.asarray(available, dtype=np.float64)
        if band.ndim != 2:
            raise ValueError(f"expected a 2D SAR band, got shape {band.shape}")
        warnings.append(SAR_SINGLE_POL_WARNING)

        if apply_speckle_filter:
            band = refined_lee(band)
        band_db = linear_to_db(band)
        stretched = percentile_stretch(band_db, percentiles=percentiles)
        channel = np.clip(np.rint(stretched), 0, 255).astype(np.uint8)
        blue = np.zeros_like(channel)
        return np.stack([channel, channel, blue], axis=-1), warnings

    co_pol = np.asarray(vv, dtype=np.float64)
    cross_pol = np.asarray(vh, dtype=np.float64)
    if co_pol.ndim != 2 or cross_pol.ndim != 2:
        raise ValueError(
            f"expected two 2D SAR bands, got shapes {co_pol.shape} and {cross_pol.shape}"
        )
    if co_pol.shape != cross_pol.shape:
        raise ValueError(
            f"polarisations must share a shape, got {co_pol.shape} and {cross_pol.shape}"
        )

    # Step 1
    if apply_speckle_filter:
        co_pol = refined_lee(co_pol)
        cross_pol = refined_lee(cross_pol)

    # Step 2
    co_db = linear_to_db(co_pol)
    cross_db = linear_to_db(cross_pol)

    # Step 4 channel construction, in the dB domain where a ratio is a difference
    ratio_db = co_db - cross_db

    # Step 3, applied independently to each of the three assembled channels
    channels = [
        percentile_stretch(co_db, percentiles=percentiles),
        percentile_stretch(cross_db, percentiles=percentiles),
        percentile_stretch(ratio_db, percentiles=percentiles),
    ]
    image = np.clip(np.rint(np.stack(channels, axis=-1)), 0, 255).astype(np.uint8)
    return image, warnings
