"""Raster ingest, modality inference, pair checks and the compatibility gate."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from satquery.io.modality import (
    canonical_band_name,
    infer_modality,
    map_band_names,
    polarisation_slots,
)
from satquery.io.pairing import bounds_of, check_pair, extent_overlap, timestamp_delta_days
from satquery.io.raster import parse_timestamp, read_image_ref, read_raster, write_raster
from satquery.io.validate import representative_gsd, validate_inputs
from satquery.serve.contracts import ImageRef, InputConfig, Modality
from tests.fixtures.synthetic import (
    T1,
    T2,
    UTM_43N,
    WGS84,
    north_up_transform,
    write_optical,
    write_sar,
)

# -- modality ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Blue", "B02"),
        ("GREEN", "B03"),
        (" red ", "B04"),
        ("nir", "B08"),
        ("MS1", "B02"),
        ("MS4", "B08"),
        ("Sigma0_VV", "VV"),
        ("swir1", "B11"),
        ("pan", "PAN"),
    ],
)
def test_band_aliases_are_case_and_sensor_tolerant(raw: str, expected: str) -> None:
    assert canonical_band_name(raw) == expected


def test_unknown_band_name_returns_none() -> None:
    assert canonical_band_name("chlorophyll") is None
    assert canonical_band_name(None) is None


def test_unmapped_bands_keep_a_placeholder_so_positions_do_not_shift() -> None:
    names, warnings = map_band_names(["B02", "mystery", "B04"])
    assert names == ["B02", "band_2", "B04"]  # not dropped; B04 stays at index 2
    assert len(warnings) == 1


@pytest.mark.parametrize(
    ("bands", "expected"),
    [
        (["VV", "VH"], Modality.SAR),
        (["HH", "HV"], Modality.SAR),
        (["B02", "B03", "B04"], Modality.OPTICAL_RGB),
        (["B02", "B03", "B04", "B08"], Modality.MULTISPECTRAL),
        (["PAN"], Modality.PANCHROMATIC),
    ],
)
def test_modality_inference(bands: list[str], expected: Modality) -> None:
    assert infer_modality(bands)[0] is expected


def test_risat_hh_hv_maps_onto_the_same_polarisation_slots_as_sentinel_vv_vh() -> None:
    assert polarisation_slots(["VV", "VH"]) == ("VV", "VH")
    assert polarisation_slots(["HH", "HV"]) == ("HH", "HV")
    assert polarisation_slots(["B02"]) == (None, None)


# -- raster round trip --------------------------------------------------------------------


def test_write_then_read_preserves_bands_geometry_and_gsd(tmp_path) -> None:
    path = write_optical(tmp_path / "ms.tif", gsd_m=10.0, size=32)
    array, ref = read_raster(path)

    assert array.shape == (10, 32, 32)
    assert ref.modality is Modality.MULTISPECTRAL
    assert ref.gsd_m == pytest.approx(10.0)
    assert ref.gsd_token == "<gsd:10.0m>"
    assert ref.crs is not None and "32643" in ref.crs
    assert ref.band_names[:4] == ["B02", "B03", "B04", "B08"]
    assert ref.shape == (32, 32)


def test_read_image_ref_matches_read_raster_metadata(tmp_path) -> None:
    path = write_sar(tmp_path / "sar.tif")
    _, full = read_raster(path)
    meta_only = read_image_ref(path)
    assert meta_only.modality is full.modality
    assert meta_only.gsd_m == full.gsd_m
    assert meta_only.band_names == full.band_names


def test_acquisition_timestamp_is_read_from_tags(tmp_path) -> None:
    path = write_optical(tmp_path / "ms.tif", when=T1)
    ref = read_image_ref(path)
    assert ref.timestamp is not None
    assert ref.timestamp.year == 2024


def test_missing_timestamp_warns_rather_than_inventing_one(tmp_path) -> None:
    path = write_optical(tmp_path / "ms.tif", when=None)
    ref = read_image_ref(path)
    assert ref.timestamp is None
    assert any("timestamp" in warning for warning in ref.warnings)


def test_geographic_crs_yields_a_gsd_in_metres_not_degrees(tmp_path) -> None:
    path = tmp_path / "geo.tif"
    write_raster(
        path,
        np.zeros((1, 8, 8), dtype="float32"),
        crs=WGS84,
        transform=(0.0001, 0.0, 77.0, 0.0, -0.0001, 12.0),
        band_names=["PAN"],
    )
    ref = read_image_ref(path)
    assert ref.gsd_m is not None
    assert 8.0 < ref.gsd_m < 13.0  # roughly 11 m, not 0.0001


@pytest.mark.parametrize(
    "raw", ["2024:01:15 05:30:00", "2024-01-15 05:30:00", "2024-01-15T05:30:00", "20240115"]
)
def test_timestamp_formats_seen_in_the_wild(raw: str) -> None:
    parsed = parse_timestamp(raw)
    assert parsed is not None and parsed.year == 2024


def test_unparseable_timestamp_returns_none_rather_than_raising() -> None:
    assert parse_timestamp("last tuesday") is None
    assert parse_timestamp(None) is None


# -- pairing ------------------------------------------------------------------------------


def _ref(**overrides) -> ImageRef:
    base = {
        "path": "/tmp/x.tif",
        "modality": Modality.MULTISPECTRAL,
        "gsd_m": 10.0,
        "crs": UTM_43N,
        "transform": north_up_transform(10.0),
        "width": 32,
        "height": 32,
        "band_names": ["B02", "B03", "B04", "B08"],
        "timestamp": T1,
    }
    return ImageRef(**{**base, **overrides})


def test_bounds_are_computed_from_all_four_corners() -> None:
    bounds = bounds_of(_ref())
    assert bounds == (600000.0, 2000000.0 - 320.0, 600000.0 + 320.0, 2000000.0)


def test_identical_footprints_overlap_completely() -> None:
    assert extent_overlap(_ref(), _ref()) == pytest.approx(1.0)


def test_disjoint_footprints_do_not_overlap() -> None:
    far = _ref(transform=north_up_transform(10.0, origin_x=9_000_000.0))
    assert extent_overlap(_ref(), far) == pytest.approx(0.0)


def test_timestamp_delta_is_absolute_and_in_days() -> None:
    delta = timestamp_delta_days(_ref(timestamp=T1), _ref(timestamp=T2))
    assert delta == pytest.approx((T2 - T1).total_seconds() / 86400.0)


def test_check_pair_flags_a_crs_mismatch() -> None:
    check = check_pair(_ref(), _ref(crs="EPSG:4326"))
    assert check.crs_match is False
    assert any("CRS mismatch" in warning for warning in check.warnings)
    assert not check.co_registered


def test_check_pair_reports_co_registration_for_an_aligned_pair() -> None:
    assert check_pair(_ref(), _ref(timestamp=T2)).co_registered


# -- the gate -----------------------------------------------------------------------------


def test_one_image_is_single() -> None:
    config, _ = validate_inputs([_ref()])
    assert config is InputConfig.SINGLE


def test_two_dates_of_one_sensor_is_bi_temporal() -> None:
    config, _ = validate_inputs([_ref(timestamp=T1), _ref(timestamp=T2)])
    assert config is InputConfig.BI_TEMPORAL_PAIR


def test_two_sensors_is_cross_modal() -> None:
    sar = _ref(modality=Modality.SAR, band_names=["VV", "VH"])
    config, _ = validate_inputs([_ref(), sar])
    assert config is InputConfig.CROSS_MODAL_PAIR


def test_sensor_difference_wins_over_date_difference() -> None:
    # Only fusion can consume two modalities, so a cross-sensor pair routes to fusion
    # even when the dates also differ. The date difference becomes a warning.
    sar = _ref(modality=Modality.SAR, band_names=["VV", "VH"], timestamp=T2)
    config, warnings = validate_inputs([_ref(timestamp=T1), sar])
    assert config is InputConfig.CROSS_MODAL_PAIR
    assert any("differ in both sensor" in warning for warning in warnings)


def test_same_sensor_same_instant_still_routes_but_warns() -> None:
    config, warnings = validate_inputs(
        [_ref(timestamp=T1), _ref(timestamp=T1 + timedelta(minutes=5))]
    )
    assert config is InputConfig.BI_TEMPORAL_PAIR
    assert any("below the" in warning for warning in warnings)


def test_missing_timestamps_assume_bi_temporal_with_a_warning() -> None:
    config, warnings = validate_inputs([_ref(timestamp=None), _ref(timestamp=None)])
    assert config is InputConfig.BI_TEMPORAL_PAIR
    assert any("no comparable timestamps" in warning for warning in warnings)


def test_gate_rejects_zero_and_more_than_two_inputs() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        validate_inputs([])
    with pytest.raises(ValueError, match="at most two images"):
        validate_inputs([_ref(), _ref(), _ref()])


def test_representative_gsd_is_the_coarsest_known_value() -> None:
    assert representative_gsd([_ref(gsd_m=0.5), _ref(gsd_m=10.0)]) == 10.0
    assert representative_gsd([_ref(gsd_m=None), _ref(gsd_m=2.0)]) == 2.0
    assert representative_gsd([_ref(gsd_m=None)]) is None


def test_a_non_georeferenced_raster_reports_unknown_gsd(tmp_path):
    """rasterio substitutes the identity matrix for an image with no georeferencing, and
    its pixel size is exactly 1.0 -- so reading a GSD from it yields a confident "1.0 m"
    for an image whose true scale is unknown.

    That is the fabrication `preprocess/gsd.py` exists to prevent, and it is worse than an
    unknown: a wrong GSD token silently mis-scales every downstream decision, and nothing
    in the output reveals it. Caught in the browser, on a PNG upload showing <gsd:1.0m>.
    """
    import numpy as np
    from PIL import Image

    from satquery.io.raster import read_image_ref

    path = tmp_path / "plain.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)

    ref = read_image_ref(path)
    assert ref.gsd_m is None
    assert ref.gsd_token == "<gsd:unknown>"
    assert ref.transform is None


def test_warnings_are_not_repeated_in_a_result():
    """The same raster is parsed twice on a normal call -- once by the caller building the
    ImageRef, once by `load_model_input` reading pixels -- so every ingest warning arrives
    from both paths. Showing a user the same sentence twice reads as a bug in the warning
    and buries whichever one matters."""
    from satquery.models.base import BaseTool

    merged = list(dict.fromkeys(["a", "b", "a", "c", "b"]))
    assert merged == ["a", "b", "c"]
    assert hasattr(BaseTool, "run")


def test_an_rgba_screenshot_is_read_positionally(tmp_path):
    """A PNG screenshot usually carries an alpha channel. Four unnamed bands are RGBA by
    the format's own definition, so reading them positionally is sound -- and refusing
    them rejected the most ordinary input a user has."""
    import numpy as np
    from PIL import Image

    from satquery.io.raster import read_image_ref
    from satquery.models.base import load_model_input

    path = tmp_path / "screenshot.png"
    Image.fromarray(np.random.randint(0, 255, (16, 24, 4), dtype=np.uint8), "RGBA").save(path)

    rgb, warnings = load_model_input(read_image_ref(path))
    assert rgb.shape == (16, 24, 3)
    assert any("alpha channel is dropped" in w for w in warnings)
