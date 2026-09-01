"""Optical preprocessing: canonical band order, pan-sharpening, contrast stretch.

Three responsibilities, all deterministic and all CPU-only:

- reorder an arbitrary sensor's bands into the frozen internal order, by name,
- sharpen a Cartosat-2S style panchromatic plus multispectral stack onto one grid,
- stretch a band to display range.

`percentile_stretch` lives here and is imported by `preprocess/sar.py`. There is
deliberately only one implementation: if the optical and SAR paths each had their own,
they would drift, and the whole point of this package is that they cannot.

Arrays are channels-first `(bands, height, width)`, matching how rasterio reads.
"""

from __future__ import annotations

import numpy as np

from satquery.preprocess.constants import (
    OPTICAL_BAND_ORDER_4,
    OPTICAL_BAND_ORDER_10,
    PANSHARPEN_METHOD_DEFAULT,
    PANSHARPEN_METHOD_FALLBACK,
    PANSHARPEN_VARIANCE_EPS,
    STRETCH_OUTPUT_RANGE,
    STRETCH_PERCENTILES,
)

__all__ = [
    "brovey_pansharpen",
    "canonical_order_for",
    "gram_schmidt_pansharpen",
    "pansharpen",
    "percentile_stretch",
    "reorder_bands",
    "stretch_to_uint8",
    "upsample_to",
]


# --------------------------------------------------------------------------------------
# Contrast stretch
# --------------------------------------------------------------------------------------


def percentile_stretch(
    band: np.ndarray,
    *,
    percentiles: tuple[float, float] = STRETCH_PERCENTILES,
    out_range: tuple[float, float] = STRETCH_OUTPUT_RANGE,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Linearly rescale `band` so the given percentiles map to the output range.

    Per-image, not per-dataset: the hidden evaluation imagery has a different
    radiometric calibration from the training imagery, and a fixed global scaling
    would map it into the wrong part of the range.

    Args:
        band: Array of any shape. Non-finite values are treated as invalid.
        percentiles: `(low, high)` percentiles mapped to the ends of `out_range`.
        out_range: `(min, max)` of the returned array.
        valid_mask: Optional boolean array, same shape as `band`, marking usable
            pixels. Combined with the finite check.

    Returns:
        A float64 array of `band`'s shape, clipped to `out_range`. Invalid pixels
        take the low end of the range.
    """
    values = np.asarray(band, dtype=np.float64)
    finite = np.isfinite(values)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)

    out_min, out_max = out_range
    if not finite.any():
        return np.full(values.shape, out_min, dtype=np.float64)

    low, high = np.percentile(values[finite], percentiles)
    if high <= low:
        # A constant band carries no contrast to stretch; collapse it deterministically
        # rather than dividing by zero.
        return np.full(values.shape, out_min, dtype=np.float64)

    scaled = (values - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled = out_min + scaled * (out_max - out_min)
    return np.where(finite, scaled, out_min)


def stretch_to_uint8(
    stack: np.ndarray,
    *,
    percentiles: tuple[float, float] = STRETCH_PERCENTILES,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Stretch every channel of a `(bands, height, width)` stack independently to uint8.

    Channels are stretched independently because they carry different physical
    quantities: in the SAR pseudo-RGB layout the third channel is a dB ratio whose
    range has nothing to do with the two backscatter channels.
    """
    array = np.asarray(stack)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"expected a 2D band or 3D (bands, h, w) stack, got shape {array.shape}")

    stretched = np.stack(
        [
            percentile_stretch(channel, percentiles=percentiles, valid_mask=valid_mask)
            for channel in array
        ]
    )
    return np.clip(np.rint(stretched), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Band ordering
# --------------------------------------------------------------------------------------


def canonical_order_for(band_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Pick the frozen band order this set of bands can satisfy.

    Returns the 10-band order when every one of its bands is present, otherwise the
    4-band order. Raises when even the 4-band order cannot be met, because silently
    returning fewer bands would make every downstream positional access wrong.
    """
    present = set(band_names)
    if set(OPTICAL_BAND_ORDER_10).issubset(present):
        return OPTICAL_BAND_ORDER_10
    if set(OPTICAL_BAND_ORDER_4).issubset(present):
        return OPTICAL_BAND_ORDER_4
    missing = sorted(set(OPTICAL_BAND_ORDER_4) - present)
    raise ValueError(
        f"cannot reach the 4-band canonical order {OPTICAL_BAND_ORDER_4}: missing {missing}. "
        f"Available bands: {sorted(present)}"
    )


def reorder_bands(
    array: np.ndarray,
    band_names: list[str] | tuple[str, ...],
    target_order: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Reorder a `(bands, h, w)` stack into the frozen internal order, by name.

    Never by position. A sensor that happens to store red first would otherwise be
    read as blue, and every index computed from it would be quietly wrong.

    Args:
        array: Stack whose first axis is bands, in `band_names` order.
        band_names: Canonical names, already mapped by `io/modality.py`.
        target_order: Desired order. Defaults to whichever frozen order the input
            can satisfy.

    Returns:
        A view-backed stack whose first axis follows `target_order`.
    """
    stack = np.asarray(array)
    if stack.ndim != 3:
        raise ValueError(f"expected a 3D (bands, h, w) stack, got shape {stack.shape}")
    if stack.shape[0] != len(band_names):
        raise ValueError(
            f"band count mismatch: array has {stack.shape[0]} bands but {len(band_names)} "
            f"names were given ({list(band_names)})"
        )

    order = canonical_order_for(band_names) if target_order is None else target_order

    index_of: dict[str, int] = {}
    for position, name in enumerate(band_names):
        index_of.setdefault(name, position)

    missing = [name for name in order if name not in index_of]
    if missing:
        raise ValueError(
            f"cannot reorder into {order}: missing bands {missing}. Available: {sorted(index_of)}"
        )

    return stack[[index_of[name] for name in order]]


# --------------------------------------------------------------------------------------
# Pan-sharpening
# --------------------------------------------------------------------------------------


def upsample_to(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a `(bands, h, w)` stack onto `(height, width)`.

    Used to put lower-resolution multispectral bands on the panchromatic grid before
    sharpening. Falls back to nearest-neighbour repetition when SciPy is unavailable
    only for exact integer factors; otherwise SciPy is required.
    """
    stack = np.asarray(array, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"expected a 3D (bands, h, w) stack, got shape {stack.shape}")

    target_h, target_w = shape
    if stack.shape[1:] == (target_h, target_w):
        return stack

    from scipy.ndimage import zoom

    factors = (1.0, target_h / stack.shape[1], target_w / stack.shape[2])
    resampled = zoom(stack, factors, order=1, mode="nearest", grid_mode=True)
    # `zoom` can land one pixel off on non-integer factors; trim or pad-by-edge to be exact.
    resampled = resampled[:, :target_h, :target_w]
    if resampled.shape[1:] != (target_h, target_w):
        pad_h = target_h - resampled.shape[1]
        pad_w = target_w - resampled.shape[2]
        resampled = np.pad(resampled, ((0, 0), (0, max(pad_h, 0)), (0, max(pad_w, 0))), mode="edge")
    return resampled


def _simulate_pan(ms: np.ndarray) -> np.ndarray:
    """Approximate the panchromatic response as the equal-weight mean of the MS bands.

    Equal weights rather than sensor-specific spectral response weights: the response
    curves differ between Sentinel-2 at training time and Cartosat-2S at evaluation
    time, and a train-only weighting is exactly the drift this package forbids.
    """
    return ms.mean(axis=0)


def brovey_pansharpen(ms: np.ndarray, pan: np.ndarray) -> np.ndarray:
    """Brovey transform: scale each band by the ratio of real to simulated pan.

    The fallback method. Cheap, always defined, and mildly distorts colour balance.
    """
    ms_f = np.asarray(ms, dtype=np.float64)
    pan_f = np.asarray(pan, dtype=np.float64)
    if ms_f.ndim != 3:
        raise ValueError(f"expected a 3D (bands, h, w) MS stack, got shape {ms_f.shape}")

    ms_f = upsample_to(ms_f, pan_f.shape)
    simulated = _simulate_pan(ms_f)
    ratio = pan_f / np.where(np.abs(simulated) < PANSHARPEN_VARIANCE_EPS, np.nan, simulated)
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    return ms_f * ratio[np.newaxis, ...]


def gram_schmidt_pansharpen(ms: np.ndarray, pan: np.ndarray) -> np.ndarray:
    """Gram-Schmidt spectral sharpening (Laben and Brower, US patent 6,011,875).

    The default method, because it preserves the spectral relationships the indices
    depend on far better than Brovey does -- and NDWI, NDBI and NDVI are the safety
    net that has to stay trustworthy on unseen sensors.

    Procedure:
      1. Simulate a low-resolution panchromatic band from the MS stack.
      2. Gram-Schmidt orthogonalise `[simulated_pan, *ms_bands]`, keeping the
         projection coefficients.
      3. Histogram-match the real panchromatic band to the first GS component.
      4. Substitute it for that component and invert the transform with the stored
         coefficients.

    Raises:
        ValueError: A GS component is degenerate, so the transform is not invertible.
            `pansharpen` catches this and falls back to Brovey.
    """
    ms_f = np.asarray(ms, dtype=np.float64)
    pan_f = np.asarray(pan, dtype=np.float64)
    if ms_f.ndim != 3:
        raise ValueError(f"expected a 3D (bands, h, w) MS stack, got shape {ms_f.shape}")
    if pan_f.ndim != 2:
        raise ValueError(f"expected a 2D panchromatic band, got shape {pan_f.shape}")

    ms_f = upsample_to(ms_f, pan_f.shape)
    n_bands, height, width = ms_f.shape

    components = np.vstack([_simulate_pan(ms_f)[np.newaxis, ...], ms_f]).reshape(n_bands + 1, -1)
    means = components.mean(axis=1, keepdims=True)

    # Forward Gram-Schmidt, retaining the projection coefficients for the inverse.
    gs = np.empty_like(components)
    phi: list[list[float]] = [[] for _ in range(n_bands + 1)]
    gs[0] = components[0] - means[0, 0]

    for t in range(1, n_bands + 1):
        vector = components[t] - means[t, 0]
        for j in range(t):
            denominator = float(gs[j] @ gs[j])
            if denominator < PANSHARPEN_VARIANCE_EPS:
                raise ValueError(
                    f"Gram-Schmidt component {j} is degenerate (norm^2={denominator:.3e}); "
                    "the transform is not invertible for this stack"
                )
            coefficient = float(vector @ gs[j]) / denominator
            phi[t].append(coefficient)
            vector = vector - coefficient * gs[j]
        gs[t] = vector

    # Histogram-match the real pan to the first GS component.
    gs0_std = float(gs[0].std())
    pan_flat = pan_f.reshape(-1)
    pan_std = float(pan_flat.std())
    if pan_std < PANSHARPEN_VARIANCE_EPS:
        raise ValueError("panchromatic band is constant; nothing to sharpen with")
    matched = (pan_flat - pan_flat.mean()) * (gs0_std / pan_std)

    # Inverse transform with the substituted first component.
    sharpened_gs = gs.copy()
    sharpened_gs[0] = matched

    out = np.empty_like(components)
    out[0] = sharpened_gs[0] + means[0, 0]
    for t in range(1, n_bands + 1):
        reconstructed = sharpened_gs[t] + means[t, 0]
        for j, coefficient in enumerate(phi[t]):
            reconstructed = reconstructed + coefficient * sharpened_gs[j]
        out[t] = reconstructed

    return out[1:].reshape(n_bands, height, width)


def pansharpen(
    ms: np.ndarray,
    pan: np.ndarray,
    *,
    method: str = PANSHARPEN_METHOD_DEFAULT,
) -> tuple[np.ndarray, list[str]]:
    """Sharpen `ms` with `pan`, falling back to Brovey when Gram-Schmidt is degenerate.

    Cartosat-2S delivers a high-resolution panchromatic band alongside coarser
    multispectral bands. Sharpening happens before anything else sees the stack, so
    the fusion tool always receives one consistent resolution.

    Args:
        ms: `(bands, h, w)` multispectral stack, already in canonical band order.
        pan: `(H, W)` panchromatic band at the target resolution.
        method: `"gram_schmidt"` or `"brovey"`.

    Returns:
        `(sharpened, warnings)` where `sharpened` is `(bands, H, W)` float64.
    """
    warnings: list[str] = []

    if method == PANSHARPEN_METHOD_FALLBACK:
        return brovey_pansharpen(ms, pan), warnings

    if method != PANSHARPEN_METHOD_DEFAULT:
        raise ValueError(
            f"unknown pan-sharpening method {method!r}; expected "
            f"{PANSHARPEN_METHOD_DEFAULT!r} or {PANSHARPEN_METHOD_FALLBACK!r}"
        )

    try:
        return gram_schmidt_pansharpen(ms, pan), warnings
    except ValueError as exc:
        warnings.append(
            f"Gram-Schmidt pan-sharpening failed ({exc}); fell back to {PANSHARPEN_METHOD_FALLBACK}"
        )
        return brovey_pansharpen(ms, pan), warnings
