"""Tests for the reBEN LMDB reader and the fusion loader's pure parts.

Everything here builds its own tiny LMDB from synthetic arrays, so the suite stays
runnable on a machine that has never seen the 162 GB store.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.data.datasets.bigearthnet_fusion import corine_extraction_mask
from satquery.io.reben import REFERENCE_MAP_SUFFIX, ReBENStore, patch_gsd_m
from satquery.preprocess.constants import (
    FUSION_EXTRACTION_CLASSES,
    FUSION_MASK_BACKGROUND,
    REBEN_PATCH_EXTENT_M,
)
from satquery.preprocess.sar import db_to_linear, linear_to_db

pytest.importorskip("lmdb")
pytest.importorskip("safetensors")


# -- fixtures ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """A three-key reBEN LMDB: one S2 patch, its S1 pair, and a reference map."""
    import lmdb
    from safetensors.numpy import save

    path = tmp_path / "lmdb"
    rng = np.random.default_rng(0)

    optical = {
        band: rng.integers(100, 4000, size=(120, 120), dtype=np.uint16)
        for band in ("B02", "B03", "B04", "B08")
    }
    # A 60 m band, stored at its native 20x20 like the real corpus.
    optical["B01"] = rng.integers(100, 900, size=(20, 20), dtype=np.uint16)

    # Sentinel-1 in decibels, as reBEN actually ships it.
    sar = {
        "VV": rng.uniform(-25.0, -5.0, size=(120, 120)).astype(np.float32),
        "VH": rng.uniform(-32.0, -12.0, size=(120, 120)).astype(np.float32),
    }
    reference = np.full((120, 120), 311, dtype=np.uint16)
    reference[:10, :10] = 112  # built-up
    reference[20:25, 20:25] = 512  # water
    reference[40:45, 40:45] = 411  # wetland, deliberately neither

    env = lmdb.open(str(path), map_size=64 * 1024**2)
    with env.begin(write=True) as transaction:
        transaction.put(b"S2_PATCH", save(optical))
        transaction.put(b"S1_PATCH", save(sar))
        transaction.put(f"S2_PATCH{REFERENCE_MAP_SUFFIX}".encode(), save({"Data": reference}))
    env.close()
    return ReBENStore(path)


# -- GSD --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "expected"),
    [(120, 10.0), (60, 20.0), (20, 60.0)],
)
def test_patch_gsd_matches_sentinel2_native_resolutions(width, expected):
    """The 1200 m patch invariant must reproduce Sentinel-2's real band resolutions."""
    assert patch_gsd_m(width) == pytest.approx(expected)


def test_patch_gsd_rejects_nonpositive_width():
    with pytest.raises(ValueError, match="must be positive"):
        patch_gsd_m(0)


def test_patch_extent_is_the_only_source_of_scale():
    """GSD scales with the extent constant, proving nothing is hardcoded per sensor."""
    assert patch_gsd_m(120, extent_m=REBEN_PATCH_EXTENT_M * 2) == pytest.approx(20.0)


# -- store ------------------------------------------------------------------------------


def test_optical_returns_stack_in_requested_order(store):
    stack, gsd = store.optical("S2_PATCH", ("B04", "B03", "B02"))
    assert stack.shape == (3, 120, 120)
    assert stack.dtype == np.float32
    assert gsd == pytest.approx(10.0)


def test_optical_refuses_to_mix_resolutions(store):
    """Silently resampling here would hide a real caller error behind plausible pixels."""
    with pytest.raises(ValueError, match="more than one resolution"):
        store.optical("S2_PATCH", ("B02", "B01"))


def test_optical_reports_missing_bands(store):
    with pytest.raises(KeyError, match="B12"):
        store.optical("S2_PATCH", ("B02", "B12"))


def test_missing_patch_raises_keyerror(store):
    with pytest.raises(KeyError, match="not in the reBEN LMDB"):
        store.read("NOT_A_PATCH")


def test_sar_is_converted_from_db_to_linear(store):
    """reBEN stores dB; the frozen pipeline starts at linear, so the reader inverts."""
    stored = store.read("S1_PATCH")
    vv, _vh, gsd, warnings = store.sar("S1_PATCH")

    assert warnings == []
    assert gsd == pytest.approx(10.0)
    # Every dB value was negative, so every linear value must be a small positive.
    assert (vv > 0.0).all() and (vv < 1.0).all()
    # And the inversion must be the exact inverse of the forward conversion.
    np.testing.assert_allclose(linear_to_db(vv), stored["VV"], rtol=1e-6, atol=1e-6)


def test_sar_warns_when_units_look_wrong(tmp_path):
    """If the source ever ships linear values, inverting them must not pass silently."""
    import lmdb
    from safetensors.numpy import save

    path = tmp_path / "linear_lmdb"
    env = lmdb.open(str(path), map_size=16 * 1024**2)
    with env.begin(write=True) as transaction:
        transaction.put(
            b"S1_LINEAR",
            save(
                {
                    "VV": np.full((8, 8), 0.5, dtype=np.float32),
                    "VH": np.full((8, 8), 0.2, dtype=np.float32),
                }
            ),
        )
    env.close()

    _, _, _, warnings = ReBENStore(path).sar("S1_LINEAR")
    assert warnings, "linear-looking values must produce a units warning"
    assert "likely linear sigma0" in warnings[0]


def test_reference_map_round_trips(store):
    reference, gsd = store.reference_map("S2_PATCH")
    assert reference.shape == (120, 120)
    assert gsd == pytest.approx(10.0)
    assert set(np.unique(reference).tolist()) == {112, 311, 411, 512}


def test_missing_lmdb_names_the_build_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="convert_reben"):
        ReBENStore(tmp_path / "absent").read("anything")


def test_store_len_counts_all_keys(store):
    assert len(store) == 3


# -- dB round trip ----------------------------------------------------------------------


def test_db_round_trip_is_exact():
    """The epsilon must cancel, or every SAR pixel drifts by a constant."""
    values = np.array([[0.0, 1e-6, 1e-3, 0.5, 1.0, 25.0]])
    np.testing.assert_allclose(db_to_linear(linear_to_db(values)), values, rtol=0, atol=1e-12)


def test_db_to_linear_never_returns_negative():
    assert (db_to_linear(np.array([-300.0, -60.0, 0.0])) >= 0.0).all()


# -- CORINE mapping ---------------------------------------------------------------------


def test_corine_mask_groups_on_level1_digit():
    codes = np.array([[111, 124, 231, 311, 512, 523]], dtype=np.uint16)
    mask = corine_extraction_mask(codes)
    built_up = FUSION_EXTRACTION_CLASSES.index("built_up") + 1
    water = FUSION_EXTRACTION_CLASSES.index("water") + 1
    assert mask.tolist() == [
        [built_up, built_up, FUSION_MASK_BACKGROUND, FUSION_MASK_BACKGROUND, water, water]
    ]


def test_corine_mask_excludes_wetlands_from_water():
    """Wetlands are only partially inundated; calling them water corrupts the
    index-agreement confidence signal."""
    mask = corine_extraction_mask(np.array([[411, 412, 421]], dtype=np.uint16))
    assert (mask == FUSION_MASK_BACKGROUND).all()


def test_corine_mask_rejects_non_2d():
    with pytest.raises(ValueError, match="2D reference map"):
        corine_extraction_mask(np.zeros((2, 4, 4), dtype=np.uint16))


def test_corine_mask_dtype_is_uint8():
    assert corine_extraction_mask(np.array([[512]], dtype=np.uint16)).dtype == np.uint8
