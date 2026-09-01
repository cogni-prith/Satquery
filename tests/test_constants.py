"""The frozen constants must stay frozen, and self-consistent."""

from __future__ import annotations

from satquery.preprocess import constants as C


def test_fingerprint_is_stable_within_a_process() -> None:
    assert C.constants_fingerprint() == C.constants_fingerprint()


def test_fingerprint_is_a_sha256_hex_digest() -> None:
    digest = C.constants_fingerprint()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_four_band_order_is_a_prefix_of_the_ten_band_order() -> None:
    # Positional access after reordering is only safe if the shorter order is a prefix.
    assert C.OPTICAL_BAND_ORDER_10[:4] == C.OPTICAL_BAND_ORDER_4


def test_band_roles_are_present_in_the_canonical_orders() -> None:
    for role in (C.BAND_ROLE_BLUE, C.BAND_ROLE_GREEN, C.BAND_ROLE_RED, C.BAND_ROLE_NIR):
        assert role in C.OPTICAL_BAND_ORDER_4
    assert C.BAND_ROLE_SWIR in C.OPTICAL_BAND_ORDER_10


def test_cdvqa_answer_set_is_the_full_19_value_vocabulary() -> None:
    """Verified against the published annotations, not against the project's prose.

    the architecture describes a six-class answer set. The actual data has 19 values across all
    four splits: the six land-cover classes cover only 23.4% of answers, with yes/no at
    52.2% and change-ratio buckets at 24.4%. A six-way head could not answer three
    quarters of the benchmark, so the ordering pinned here is the 19-value one, and it is
    this head's output-layer ordering.
    """
    assert len(C.CDVQA_ANSWERS) == 19
    assert C.CDVQA_ANSWERS[:2] == ("no", "yes")
    assert set(C.CDVQA_LAND_COVER_ANSWERS) < set(C.CDVQA_ANSWERS)
    assert len(C.CDVQA_LAND_COVER_ANSWERS) == 6
    # Dataset token spelling, not the project's prose spelling.
    assert "NVG_surface" in C.CDVQA_ANSWERS
    assert "non-vegetated ground surface" not in C.CDVQA_ANSWERS
    # No duplicates: a repeated label would collapse two output units.
    assert len(set(C.CDVQA_ANSWERS)) == len(C.CDVQA_ANSWERS)


def test_canonical_gsd_ladder_is_sorted_and_positive() -> None:
    assert list(C.CANONICAL_GSD_SCALES_M) == sorted(C.CANONICAL_GSD_SCALES_M)
    assert all(scale > 0 for scale in C.CANONICAL_GSD_SCALES_M)


def test_speckle_window_is_odd() -> None:
    assert C.SPECKLE_FILTER_WINDOW % 2 == 1
    assert C.SPECKLE_SUBWINDOW % 2 == 1
