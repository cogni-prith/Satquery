"""Spectral and backscatter indices: correctness, eps guards, and the agreement signal."""

from __future__ import annotations

import numpy as np
import pytest

from satquery.preprocess.constants import SAR_WATER_DB_THRESHOLD
from satquery.preprocess.indices import (
    agreement_score,
    builtup_mask,
    compute_indices,
    ndbi,
    ndvi,
    ndwi,
    normalised_difference,
    sar_water_mask,
    vegetation_mask,
    water_mask,
)


def test_normalised_difference_known_value() -> None:
    assert normalised_difference(np.array([3.0]), np.array([1.0]))[0] == pytest.approx(0.5)


def test_zero_denominator_yields_zero_not_nan() -> None:
    result = normalised_difference(np.zeros(3), np.zeros(3))
    assert np.isfinite(result).all()
    assert (result == 0.0).all()


def test_output_is_clipped_to_the_valid_range() -> None:
    # A negative reflectance from over-correction must not produce an index of 40.
    result = normalised_difference(np.array([-5.0]), np.array([1.0]))
    assert -1.0 <= result[0] <= 1.0


def test_ndwi_is_high_over_water() -> None:
    # Water: high green, near-zero NIR.
    assert ndwi(np.array([0.22]), np.array([0.02]))[0] > 0.5


def test_ndvi_is_high_over_vegetation() -> None:
    assert ndvi(np.array([0.45]), np.array([0.06]))[0] > 0.5


def test_ndbi_is_positive_when_swir_exceeds_nir() -> None:
    assert ndbi(np.array([0.30]), np.array([0.10]))[0] > 0.0


def test_index_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        normalised_difference(np.zeros((2, 2)), np.zeros((3, 3)))


def test_sar_water_mask_thresholds_dark_backscatter() -> None:
    # 0.001 linear is about -60 dB, well below the threshold; 0.5 is about -3 dB.
    mask = sar_water_mask(np.array([0.001, 0.5]))
    assert mask.tolist() == [True, False]


def test_sar_water_mask_accepts_decibels_directly() -> None:
    below = SAR_WATER_DB_THRESHOLD - 1.0
    above = SAR_WATER_DB_THRESHOLD + 1.0
    assert sar_water_mask(np.array([below, above]), in_db=True).tolist() == [True, False]


def test_compute_indices_looks_bands_up_by_name() -> None:
    stack = np.stack([np.full((4, 4), 0.22), np.full((4, 4), 0.02)])
    maps, warnings = compute_indices(stack, ["B03", "B08"], ("ndwi",))
    assert "ndwi" in maps
    assert maps["ndwi"].mean() > 0.5
    assert warnings == []


def test_compute_indices_omits_rather_than_fakes_an_uncomputable_index() -> None:
    stack = np.stack([np.full((4, 4), 0.22), np.full((4, 4), 0.02)])
    maps, warnings = compute_indices(stack, ["B03", "B08"], ("ndwi", "ndbi"))
    assert "ndbi" not in maps  # never a zero-filled placeholder
    assert any("NDBI not computed" in warning for warning in warnings)


def test_compute_indices_rejects_an_unknown_index_name() -> None:
    with pytest.raises(ValueError, match="unknown index"):
        compute_indices(np.zeros((2, 4, 4)), ["B03", "B08"], ("ndxx",))


def test_masks_apply_their_thresholds() -> None:
    assert water_mask(np.array([0.5, -0.5])).tolist() == [True, False]
    assert builtup_mask(np.array([0.5, -0.5])).tolist() == [True, False]
    assert vegetation_mask(np.array([0.5, 0.1])).tolist() == [True, False]


def test_agreement_is_iou() -> None:
    a = np.array([True, True, False, False])
    b = np.array([True, False, False, False])
    assert agreement_score(a, b) == pytest.approx(0.5)


def test_agreement_of_two_empty_masks_is_one() -> None:
    # Neither found anything, which is consistent, not uninformative.
    assert agreement_score(np.zeros(4, bool), np.zeros(4, bool)) == 1.0


def test_agreement_of_disjoint_masks_is_zero() -> None:
    assert agreement_score(np.array([True, False]), np.array([False, True])) == 0.0


def test_agreement_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        agreement_score(np.zeros(4, bool), np.zeros(5, bool))
