"""The frozen SAR pipeline: speckle filtering, dB conversion, layout, single-pol path."""

from __future__ import annotations

import numpy as np
import pytest
from tests.fixtures.synthetic import sar_stack

from satquery.preprocess.constants import (
    DB_EPS,
    SPECKLE_FILTER_NUM_LOOKS,
    SPECKLE_FILTER_WINDOW,
)
from satquery.preprocess.sar import directional_masks, linear_to_db, refined_lee, render_sar


def _cv(array: np.ndarray) -> float:
    """Coefficient of variation, the quantity a speckle filter is meant to reduce."""
    return float(array.std() / array.mean())


# -- directional masks ------------------------------------------------------------------


def test_masks_have_the_expected_shape() -> None:
    masks = directional_masks(SPECKLE_FILTER_WINDOW)
    assert masks.shape == (8, SPECKLE_FILTER_WINDOW, SPECKLE_FILTER_WINDOW)


def test_every_mask_contains_the_centre_pixel() -> None:
    masks = directional_masks(7)
    assert masks[:, 3, 3].all()


def test_opposing_masks_cover_the_whole_window() -> None:
    # Each orientation's two half-windows must tile the window, or some pixel is
    # unreachable by the filter for that edge direction.
    masks = directional_masks(7)
    for orientation in range(4):
        both = masks[orientation * 2] | masks[orientation * 2 + 1]
        assert both.all()


def test_even_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="odd"):
        directional_masks(6)


# -- refined Lee ------------------------------------------------------------------------


def test_refined_lee_preserves_shape_and_dtype_contract() -> None:
    image = sar_stack(size=32)[0][0]
    filtered = refined_lee(image)
    assert filtered.shape == image.shape
    assert filtered.dtype == np.float64


def test_refined_lee_reduces_speckle_in_homogeneous_regions() -> None:
    rng = np.random.default_rng(3)
    speckle = rng.gamma(
        shape=SPECKLE_FILTER_NUM_LOOKS, scale=1.0 / SPECKLE_FILTER_NUM_LOOKS, size=(64, 64)
    )
    noisy = np.full((64, 64), 5.0) * speckle
    filtered = refined_lee(noisy)
    # Interior only, so border handling does not dominate the statistic.
    assert _cv(filtered[8:-8, 8:-8]) < 0.5 * _cv(noisy[8:-8, 8:-8])


def test_refined_lee_preserves_regional_means() -> None:
    # Smoothing must not shift the radiometry; the filter is meant to reduce variance
    # around the mean, not to move the mean.
    rng = np.random.default_rng(4)
    clean = np.ones((64, 64))
    clean[:, 32:] = 10.0
    noisy = clean * rng.gamma(shape=4.0, scale=0.25, size=clean.shape)
    filtered = refined_lee(noisy)
    assert filtered[8:-8, 8:24].mean() == pytest.approx(noisy[8:-8, 8:24].mean(), rel=0.15)
    assert filtered[8:-8, 40:56].mean() == pytest.approx(noisy[8:-8, 40:56].mean(), rel=0.15)


def test_refined_lee_preserves_a_high_contrast_edge() -> None:
    # The whole point of the "refined" variant: a plain box filter would smear the step
    # across the window width. Measure the step across the boundary after filtering.
    clean = np.ones((64, 64))
    clean[:, 32:] = 100.0
    filtered = refined_lee(clean)
    step = filtered[32, 34] - filtered[32, 29]
    assert step > 90.0


def test_refined_lee_is_deterministic() -> None:
    image = sar_stack(size=32)[0][0]
    assert np.array_equal(refined_lee(image), refined_lee(image))


def test_refined_lee_rejects_bad_arguments() -> None:
    image = np.ones((8, 8))
    with pytest.raises(ValueError, match="2D"):
        refined_lee(np.ones((2, 8, 8)))
    with pytest.raises(ValueError, match="odd"):
        refined_lee(image, window=4)
    with pytest.raises(ValueError, match="positive"):
        refined_lee(image, num_looks=0.0)


# -- dB conversion ----------------------------------------------------------------------


def test_linear_to_db_is_finite_at_zero() -> None:
    # The epsilon is inside the log, so zero backscatter is a large negative finite
    # number rather than -inf, which would poison every downstream percentile.
    value = linear_to_db(np.array([0.0]))[0]
    assert np.isfinite(value)
    assert value == pytest.approx(10.0 * np.log10(DB_EPS))


def test_linear_to_db_known_values() -> None:
    assert linear_to_db(np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-5)
    assert linear_to_db(np.array([10.0]))[0] == pytest.approx(10.0, abs=1e-5)


# -- rendering --------------------------------------------------------------------------


def test_render_dual_pol_produces_a_uint8_rgb_image() -> None:
    stack, _ = sar_stack(size=32)
    image, warnings = render_sar(stack[0], stack[1])
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8
    assert image.min() >= 0 and image.max() <= 255
    assert warnings == []


def test_render_single_pol_duplicates_and_zeroes_blue_with_a_warning() -> None:
    stack, _ = sar_stack(size=32)
    image, warnings = render_sar(stack[0])
    assert np.array_equal(image[..., 0], image[..., 1])
    assert (image[..., 2] == 0).all()
    assert any("single-polarisation" in warning for warning in warnings)


def test_render_accepts_cross_pol_only_as_the_single_polarisation() -> None:
    stack, _ = sar_stack(size=32)
    image, warnings = render_sar(None, stack[1])
    assert (image[..., 2] == 0).all()
    assert warnings


def test_render_is_deterministic() -> None:
    stack, _ = sar_stack(size=32)
    first, _ = render_sar(stack[0], stack[1])
    second, _ = render_sar(stack[0], stack[1])
    assert np.array_equal(first, second)


def test_render_water_is_darker_than_land_in_the_co_pol_channel() -> None:
    # Ground truth from the fixture: the left eight columns are smooth water, which is
    # a specular reflector and must come out dark.
    stack, _ = sar_stack(size=32, water_cols=8)
    image, _ = render_sar(stack[0], stack[1])
    assert image[:, :6, 0].mean() < image[:, 12:, 0].mean()


def test_render_rejects_empty_and_mismatched_input() -> None:
    with pytest.raises(ValueError, match="at least one polarisation"):
        render_sar(None, None)
    with pytest.raises(ValueError, match="share a shape"):
        render_sar(np.ones((8, 8)), np.ones((8, 9)))
