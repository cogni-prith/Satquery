"""GSD is computed from geometry, formatted exactly, and never guessed."""

from __future__ import annotations

import math

import pytest

from satquery.preprocess.constants import GSD_TOKEN_UNKNOWN
from satquery.preprocess.gsd import (
    format_gsd_token,
    gsd_from_transform,
    nearest_canonical_gsd,
    prefix_instruction,
    resample_scale_factor,
    strip_gsd_token,
)
from tests.fixtures.synthetic import north_up_transform


def test_gsd_from_a_north_up_transform() -> None:
    gsd, warnings = gsd_from_transform(north_up_transform(10.0))
    assert gsd == pytest.approx(10.0)
    assert warnings == []


def test_gsd_is_invariant_under_rotation() -> None:
    # A 30 degree rotated transform with 2 m pixels. Reading `a` alone would give 1.73.
    angle = math.radians(30.0)
    scale = 2.0
    transform = (
        scale * math.cos(angle),
        -scale * math.sin(angle),
        0.0,
        scale * math.sin(angle),
        scale * math.cos(angle),
        0.0,
    )
    gsd, _ = gsd_from_transform(transform)
    assert gsd == pytest.approx(2.0)


def test_missing_transform_yields_none_not_a_guess() -> None:
    gsd, warnings = gsd_from_transform(None)
    assert gsd is None
    assert warnings


def test_zero_pixel_extent_yields_none() -> None:
    gsd, warnings = gsd_from_transform((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert gsd is None
    assert warnings


def test_geographic_crs_is_converted_to_metres() -> None:
    # 0.0001 degrees at the equator is roughly 11 m. Read as metres it would be 0.0001.
    transform = (0.0001, 0.0, 77.0, 0.0, -0.0001, 0.0)
    gsd, warnings = gsd_from_transform(transform, is_geographic=True, centre_latitude=0.0)
    assert gsd is not None
    assert 10.0 < gsd < 12.0
    assert any("geographic" in warning for warning in warnings)


def test_geographic_crs_without_latitude_refuses_to_guess() -> None:
    transform = (0.0001, 0.0, 77.0, 0.0, -0.0001, 0.0)
    gsd, warnings = gsd_from_transform(transform, is_geographic=True, centre_latitude=None)
    assert gsd is None
    assert any("centre latitude" in warning for warning in warnings)


def test_anisotropic_pixels_warn_and_return_the_geometric_mean() -> None:
    gsd, warnings = gsd_from_transform((10.0, 0.0, 0.0, 0.0, -40.0, 0.0))
    assert gsd == pytest.approx(20.0)
    assert any("anisotropic" in warning for warning in warnings)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(10.0, "<gsd:10.0m>"), (0.6, "<gsd:0.6m>"), (0.65, "<gsd:0.7m>"), (2.0, "<gsd:2.0m>")],
)
def test_token_formatting(value: float, expected: str) -> None:
    assert format_gsd_token(value) == expected


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 0.0, 0.04, -1.0])
def test_unrepresentable_values_yield_the_unknown_token(value: float | None) -> None:
    # 0.04 would render as "<gsd:0.0m>", which reads as a measurement. Refuse instead.
    assert format_gsd_token(value) == GSD_TOKEN_UNKNOWN


def test_nearest_canonical_scale_uses_log_distance() -> None:
    # 3.0 is equidistant from 2 and 5 linearly, but closer to 2 as a ratio.
    assert nearest_canonical_gsd(3.0) == 2.0
    assert nearest_canonical_gsd(9.0) == 10.0
    assert nearest_canonical_gsd(None) is None


def test_prefix_is_idempotent() -> None:
    once = prefix_instruction("Describe the scene.", 10.0)
    assert once == "<gsd:10.0m> Describe the scene."
    assert prefix_instruction(once, 10.0) == once
    assert prefix_instruction(once, 0.5) == once


def test_prefix_falls_back_to_unknown() -> None:
    assert prefix_instruction("Describe the scene.", None).startswith(GSD_TOKEN_UNKNOWN)


def test_strip_removes_only_the_leading_token() -> None:
    assert strip_gsd_token("<gsd:10.0m> Describe it.") == "Describe it."
    assert strip_gsd_token("Describe it.") == "Describe it."


def test_resample_scale_factor() -> None:
    assert resample_scale_factor(10.0, 2.0) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="positive"):
        resample_scale_factor(10.0, 0.0)
