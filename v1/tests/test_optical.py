"""Optical preprocessing: band order by name, pan-sharpening, contrast stretch."""

from __future__ import annotations

import numpy as np
import pytest

from satquery.preprocess.constants import OPTICAL_BAND_ORDER_4, OPTICAL_BAND_ORDER_10
from satquery.preprocess.optical import (
    brovey_pansharpen,
    canonical_order_for,
    gram_schmidt_pansharpen,
    pansharpen,
    percentile_stretch,
    reorder_bands,
    stretch_to_uint8,
    upsample_to,
)


def test_stretch_maps_to_the_full_output_range() -> None:
    stretched = percentile_stretch(np.linspace(0, 1, 100).reshape(10, 10))
    assert stretched.min() == pytest.approx(0.0)
    assert stretched.max() == pytest.approx(255.0)


def test_stretch_of_a_constant_band_does_not_divide_by_zero() -> None:
    stretched = percentile_stretch(np.full((8, 8), 42.0))
    assert np.isfinite(stretched).all()


def test_stretch_ignores_non_finite_values() -> None:
    band = np.array([[0.0, 1.0], [np.nan, np.inf]])
    stretched = percentile_stretch(band)
    assert np.isfinite(stretched).all()


def test_stretch_respects_a_valid_mask() -> None:
    band = np.array([[0.0, 1.0], [1000.0, 1.0]])
    mask = np.array([[True, True], [False, True]])
    # The 1000 outlier is masked out, so it must not set the upper end of the range.
    stretched = percentile_stretch(band, valid_mask=mask)
    assert stretched[1, 0] == pytest.approx(255.0) or stretched[1, 0] == pytest.approx(0.0)


def test_stretch_to_uint8_handles_a_stack() -> None:
    out = stretch_to_uint8(np.random.default_rng(0).random((3, 8, 8)))
    assert out.shape == (3, 8, 8)
    assert out.dtype == np.uint8


def test_canonical_order_prefers_ten_bands_when_available() -> None:
    assert canonical_order_for(list(OPTICAL_BAND_ORDER_10)) == OPTICAL_BAND_ORDER_10
    assert canonical_order_for(list(OPTICAL_BAND_ORDER_4)) == OPTICAL_BAND_ORDER_4


def test_canonical_order_raises_when_the_four_band_set_is_incomplete() -> None:
    with pytest.raises(ValueError, match="missing"):
        canonical_order_for(["B02", "B03"])


def test_reorder_is_by_name_not_position() -> None:
    # A sensor storing red first must not be read as blue.
    stack = np.stack([np.full((4, 4), float(i)) for i in range(4)])
    names = ["B04", "B03", "B02", "B08"]
    reordered = reorder_bands(stack, names)
    assert np.array_equal(reordered[0], stack[2])  # B02 was stored third
    assert np.array_equal(reordered[2], stack[0])  # B04 was stored first


def test_reorder_rejects_a_name_count_mismatch() -> None:
    with pytest.raises(ValueError, match="band count mismatch"):
        reorder_bands(np.zeros((3, 4, 4)), ["B02", "B03"])


def test_reorder_rejects_a_missing_target_band() -> None:
    with pytest.raises(ValueError, match="missing bands"):
        reorder_bands(np.zeros((2, 4, 4)), ["B02", "B03"], ("B02", "B03", "B04"))


def _ms() -> np.ndarray:
    return np.random.default_rng(0).random((4, 16, 16)) * 1000.0


def test_gram_schmidt_is_an_identity_when_pan_equals_the_simulated_pan() -> None:
    # The strongest available correctness check: substituting the simulated pan back
    # into the transform must reconstruct the input exactly.
    ms = _ms()
    out = gram_schmidt_pansharpen(ms, ms.mean(axis=0))
    assert np.abs(out - ms).max() < 1e-8


def test_brovey_is_an_identity_when_pan_equals_the_simulated_pan() -> None:
    ms = _ms()
    out = brovey_pansharpen(ms, ms.mean(axis=0))
    assert np.abs(out - ms).max() < 1e-8


def test_pansharpen_upsamples_the_multispectral_stack_to_the_pan_grid() -> None:
    ms = np.random.default_rng(1).random((4, 8, 8))
    pan = np.random.default_rng(2).random((32, 32))
    out, _ = pansharpen(ms, pan)
    assert out.shape == (4, 32, 32)


def test_pansharpen_falls_back_to_brovey_on_a_degenerate_pan() -> None:
    out, warnings = pansharpen(_ms(), np.zeros((16, 16)))
    assert out.shape == (4, 16, 16)
    assert any("fell back to brovey" in warning for warning in warnings)


def test_pansharpen_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown pan-sharpening method"):
        pansharpen(_ms(), np.ones((16, 16)), method="wavelet")


def test_upsample_hits_the_requested_shape_exactly() -> None:
    out = upsample_to(np.random.default_rng(0).random((2, 7, 5)), (23, 19))
    assert out.shape == (2, 23, 19)


# -- VLM tiling ---------------------------------------------------------------------------
# `dynamic_tiles` lives in models/vlm/backbone.py but is a pure preprocessing function,
# so it is tested here alongside the other deterministic image maths.


def test_dynamic_tiles_square_image_is_a_single_tile() -> None:
    from satquery.models.vlm.backbone import dynamic_tiles

    tiles = dynamic_tiles(np.zeros((448, 448, 3), dtype=np.uint8))
    assert tiles.shape == (1, 3, 448, 448)  # no thumbnail is added for a lone tile


def test_dynamic_tiles_wide_image_adds_a_thumbnail() -> None:
    from satquery.models.vlm.backbone import dynamic_tiles

    # A 2:1 image fills a 2x1 grid, and the whole-image thumbnail makes three.
    tiles = dynamic_tiles(np.zeros((448, 896, 3), dtype=np.uint8))
    assert tiles.shape == (3, 3, 448, 448)


def test_dynamic_tiles_respects_the_tile_budget() -> None:
    from satquery.models.vlm.backbone import dynamic_tiles

    tiles = dynamic_tiles(np.zeros((300, 4000, 3), dtype=np.uint8), max_tiles=6)
    assert tiles.shape[0] <= 6 + 1  # budget plus the thumbnail


def test_dynamic_tiles_applies_imagenet_normalisation() -> None:
    from satquery.models.vlm.backbone import dynamic_tiles
    from satquery.preprocess.constants import IMAGENET_MEAN, IMAGENET_STD

    # A mid-grey image must map to (0.5 - mean) / std on each channel.
    tiles = dynamic_tiles(np.full((448, 448, 3), 128, dtype=np.uint8))
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)):
        expected = (128 / 255.0 - mean) / std
        assert tiles[0, channel].mean() == pytest.approx(expected, abs=1e-3)


def test_dynamic_tiles_accepts_a_rendered_sar_scene() -> None:
    # render_sar emits (H, W, 3) uint8, which must feed straight into the backbone.
    from satquery.models.vlm.backbone import dynamic_tiles
    from satquery.preprocess.sar import render_sar
    from tests.fixtures.synthetic import sar_stack

    stack, _ = sar_stack(size=64)
    rendered, _ = render_sar(stack[0], stack[1])
    tiles = dynamic_tiles(rendered)
    assert tiles.shape[1:] == (3, 448, 448)
    assert tiles.dtype == np.float32


def test_dynamic_tiles_rejects_non_rgb_input() -> None:
    from satquery.models.vlm.backbone import dynamic_tiles

    with pytest.raises(ValueError, match="expected a"):
        dynamic_tiles(np.zeros((448, 448), dtype=np.uint8))
