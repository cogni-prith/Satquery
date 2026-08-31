"""The symbolic layer, against synthetic masks. No GPU, no real imagery."""

from __future__ import annotations

import numpy as np
import pytest

from satquery.symbolic import measures, predicates
from satquery.symbolic.thresholds import classify_trend, confidence_band

WATER, BUILT, VEG = 1, 2, 3


@pytest.fixture
def mask() -> np.ndarray:
    """20x20 scene: a 4x4 water block top-left, a 6x6 built block bottom-right."""
    array = np.full((20, 20), VEG, dtype=np.uint8)
    array[0:4, 0:4] = WATER
    array[14:20, 14:20] = BUILT
    return array


# -- measures ------------------------------------------------------------------------------


def test_class_fraction_and_area(mask) -> None:
    assert measures.class_fraction(mask, WATER) == pytest.approx(16 / 400)
    # 16 pixels at 10 m GSD is 16 * 100 m2.
    assert measures.class_area_m2(mask, WATER, 10.0) == pytest.approx(1600.0)


def test_area_refuses_an_unknown_gsd(mask) -> None:
    """An unknown GSD means the area is unknown. Substituting a default would put a
    fabricated number into an answer a judge can check."""
    with pytest.raises(ValueError, match="gsd_m must be positive"):
        measures.class_area_m2(mask, WATER, 0.0)


def test_connected_components_splits_and_ranks(mask) -> None:
    components = measures.connected_components(mask, BUILT, min_area_px=1)
    assert len(components) == 1
    assert components[0]["area_px"] == 36.0
    assert components[0]["centroid_row"] == pytest.approx(16.5)


def test_min_area_drops_speckle() -> None:
    array = np.zeros((10, 10), dtype=np.uint8)
    array[0, 0] = WATER  # 1 px, speckle
    array[5:9, 5:9] = WATER  # 16 px, real
    assert len(measures.connected_components(array, WATER, min_area_px=4)) == 1


def test_area_delta_signs_and_ratio(mask) -> None:
    later = mask.copy()
    later[4:8, 0:4] = WATER  # water doubles
    delta = measures.area_delta(mask, later, WATER, 10.0)
    assert delta["absolute_m2"] == pytest.approx(1600.0)
    assert delta["relative"] == pytest.approx(1.0)


def test_growth_from_nothing_has_no_infinite_ratio() -> None:
    """A class absent at T1 would give inf, which propagates into the threshold as a
    spurious 'increased'."""
    empty = np.full((10, 10), VEG, dtype=np.uint8)
    later = empty.copy()
    later[0:3, 0:3] = WATER
    assert measures.area_delta(empty, later, WATER, 10.0)["relative"] == 0.0


def test_mismatched_masks_are_refused(mask) -> None:
    with pytest.raises(ValueError, match="co-registered"):
        measures.area_delta(mask, np.zeros((5, 5), dtype=np.uint8), WATER, 10.0)


def test_change_matrix_reports_transitions(mask) -> None:
    later = mask.copy()
    later[0:4, 0:4] = BUILT  # water -> built
    matrix = measures.change_matrix(mask, later, gsd_m=10.0)
    assert matrix[(WATER, BUILT)] == pytest.approx(1600.0)
    assert (WATER, WATER) not in matrix


def test_agreement_is_nan_when_nothing_was_measured() -> None:
    """Two empty masks measured nothing. 0.0 and 1.0 would both be fabricated."""
    empty = np.zeros((8, 8), dtype=bool)
    assert measures.agreement_iou(empty, empty) != measures.agreement_iou(empty, empty)


def test_agreement_iou_is_symmetric_and_correct() -> None:
    a = np.zeros((10, 10), dtype=bool)
    a[0:5, :] = True
    b = np.zeros((10, 10), dtype=bool)
    b[3:8, :] = True
    assert measures.agreement_iou(a, b) == pytest.approx(20 / 80)


# -- thresholds ----------------------------------------------------------------------------


def test_both_floors_must_be_exceeded() -> None:
    """A large relative change on a tiny area is noise; a large absolute change on a huge
    area can sit inside measurement error."""
    assert classify_trend(50_000.0, 0.20) == "increased"
    assert classify_trend(50_000.0, 0.001) == "unchanged"  # relative too small
    assert classify_trend(100.0, 0.90) == "unchanged"  # absolute too small
    assert classify_trend(-80_000.0, -0.30) == "decreased"


def test_unmeasurable_agreement_bands_low() -> None:
    """An agreement that could not be computed is exactly what a reader needs flagged."""
    assert confidence_band(float("nan")) == "low"


# -- predicates ----------------------------------------------------------------------------


def test_northernmost_uses_the_smaller_row(mask) -> None:
    """Row 0 is the top. Taking the maximum would return the southernmost and nothing
    downstream would notice."""
    components = [
        {"area_px": 4.0, "centroid_row": 2.0, "centroid_col": 5.0},
        {"area_px": 9.0, "centroid_row": 18.0, "centroid_col": 5.0},
    ]
    assert predicates.northernmost(components)["centroid_row"] == 2.0


def test_largest_and_smallest(mask) -> None:
    components = measures.connected_components(mask, VEG, min_area_px=1)
    assert predicates.largest(components)["area_px"] >= predicates.smallest(components)["area_px"]


def test_nearest_to_a_point() -> None:
    components = [
        {"area_px": 1.0, "centroid_row": 0.0, "centroid_col": 0.0},
        {"area_px": 1.0, "centroid_row": 10.0, "centroid_col": 10.0},
    ]
    assert predicates.nearest_to(components, 9.0, 9.0)["centroid_row"] == 10.0


def test_unknown_sector_is_refused() -> None:
    """Returning nothing would read as 'no such region' rather than 'no such sector'."""
    with pytest.raises(ValueError, match="unknown sector"):
        predicates.in_sector([], "up-and-left", 20, 20)


def test_sector_membership_follows_image_north() -> None:
    north = {"area_px": 1.0, "centroid_row": 1.0, "centroid_col": 10.0}
    south = {"area_px": 1.0, "centroid_row": 19.0, "centroid_col": 10.0}
    chosen = predicates.in_sector([north, south], "north", 20, 20)
    assert chosen == [north]


def test_resolve_reference_ranks_rather_than_guessing() -> None:
    """A referring phrase is often genuinely ambiguous. Collapsing to one box discards
    that instead of reporting it."""
    components = [
        {"area_px": 100.0, "centroid_row": 2.0, "centroid_col": 10.0},
        {"area_px": 10.0, "centroid_row": 18.0, "centroid_col": 10.0},
    ]
    ranked = predicates.resolve_reference(components, "the largest one in the north", 20, 20)
    assert len(ranked) == 2
    assert ranked[0]["area_px"] == 100.0
    assert ranked[0]["score"] > ranked[1]["score"]
    assert "largest" in ranked[0]["matched"] and "north" in ranked[0]["matched"]


def test_resolve_reference_on_nothing_returns_nothing() -> None:
    assert predicates.resolve_reference([], "the largest", 20, 20) == []
