"""The deterministic index tool, end to end on synthetic georeferenced rasters.

The only tool v2 can answer with today, so it gets the test that proves the whole chain:
raster -> index -> mask -> measured area -> AnswerRecord -> English.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.io.raster import read_image_ref
from satquery.models.indices.deterministic import DeterministicIndexTool
from satquery.models.registry import REGISTRY
from satquery.serve.contracts import ToolRequest

GSD = 10.0
SIDE = 100
#: A 20x20 lake at date 1 growing to 40x40 at date 2, at 10 m GSD.
LAKE_T1_M2 = 20 * 20 * GSD**2
LAKE_T2_M2 = 40 * 40 * GSD**2


def _write(path, lake_size: int):
    """A 4-band multispectral tile with a square of water in the top-left corner.

    Water is bright in green and dark in NIR, which is what makes NDWI positive; land is
    the reverse. Written with a real affine transform so the GSD is read, not guessed.
    """
    green = np.full((SIDE, SIDE), 0.10, dtype=np.float32)
    nir = np.full((SIDE, SIDE), 0.40, dtype=np.float32)
    green[:lake_size, :lake_size] = 0.30
    nir[:lake_size, :lake_size] = 0.05
    red = np.full((SIDE, SIDE), 0.15, dtype=np.float32)
    swir = np.full((SIDE, SIDE), 0.20, dtype=np.float32)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIDE,
        width=SIDE,
        count=4,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 2000000.0, GSD, GSD),
    ) as dst:
        for index, (band, name) in enumerate(
            [(red, "B04"), (green, "B03"), (nir, "B08"), (swir, "B11")], start=1
        ):
            dst.write(band, index)
            dst.set_band_description(index, name)
    return path


@pytest.fixture
def tool() -> DeterministicIndexTool:
    return DeterministicIndexTool(REGISTRY.get_spec("indices.deterministic"))


def test_single_image_area_matches_the_geometry(tmp_path, tool) -> None:
    ref = read_image_ref(_write(tmp_path / "t1.tif", 20))
    assert ref.gsd_m == pytest.approx(GSD), "GSD must come from the transform"

    result = tool.run(ToolRequest(query="how much water is in this scene?", images=[ref]))
    assert result.ok, result.error

    area = next(f for f in result.answer_record["facts"] if f["key"] == "water_area")
    assert area["value"] == pytest.approx(LAKE_T1_M2)
    assert area["provenance"] == "symbolic.measures.class_area_m2"


def test_change_between_two_dates_is_measured_not_guessed(tmp_path, tool) -> None:
    refs = [
        read_image_ref(_write(tmp_path / "t1.tif", 20)),
        read_image_ref(_write(tmp_path / "t2.tif", 40)),
    ]
    result = tool.run(ToolRequest(query="how has the water changed?", images=refs))
    assert result.ok, result.error

    delta = result.answer_record["area_deltas"]["water"]
    assert delta["area_t1_m2"] == pytest.approx(LAKE_T1_M2)
    assert delta["area_t2_m2"] == pytest.approx(LAKE_T2_M2)
    assert delta["trend"] == "increased"
    assert "increased" in result.answer


def test_a_query_naming_one_class_does_not_answer_about_three(tmp_path, tool) -> None:
    ref = read_image_ref(_write(tmp_path / "t1.tif", 20))
    result = tool.run(ToolRequest(query="how much water?", images=[ref]))
    assert set(result.answer_record["class_proportions"]) == {"water"}


def test_an_rgb_screenshot_is_refused_rather_than_answered(tmp_path, tool) -> None:
    """No NIR band means no index applies. Answering anyway would be inventing one."""
    path = tmp_path / "screenshot.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=SIDE, width=SIDE, count=3, dtype="uint8"
    ) as dst:
        dst.write(np.full((3, SIDE, SIDE), 120, dtype=np.uint8))

    result = tool.run(ToolRequest(query="how much water?", images=[read_image_ref(path)]))
    assert not result.ok
    assert "NIR" in (result.error or "")


def test_single_optical_image_claims_no_confidence(tmp_path, tool) -> None:
    """One signal cannot agree with anything. A number here would be fabricated."""
    ref = read_image_ref(_write(tmp_path / "t1.tif", 20))
    result = tool.run(ToolRequest(query="how much water?", images=[ref]))
    assert result.confidence is None


def test_open_water_is_not_counted_as_built_up(tmp_path, tool) -> None:
    """NDBI is high over water as well as concrete: both are bright in SWIR relative to
    NIR. Unmasked, a growing lake reads as a construction boom -- which is exactly what
    this tool did before the water mask was subtracted."""
    refs = [
        read_image_ref(_write(tmp_path / "t1.tif", 20)),
        read_image_ref(_write(tmp_path / "t2.tif", 40)),
    ]
    result = tool.run(ToolRequest(query="how much has the built-up area changed?", images=refs))
    assert result.ok, result.error

    delta = result.answer_record["area_deltas"]["built_up"]
    assert delta["absolute_m2"] == pytest.approx(0.0), (
        "the synthetic scene has no built-up change; any delta here is the lake leaking "
        "through NDBI"
    )
    assert delta["trend"] == "unchanged"
    assert "water" not in result.answer_record["area_deltas"], "only built-up was asked about"


def _write_no_blue(path, lake: int):
    """A four-band stack with no blue band — the ordinary B04/B03/B08/B11 case."""
    green = np.full((SIDE, SIDE), 0.10, dtype=np.float32)
    nir = np.full((SIDE, SIDE), 0.40, dtype=np.float32)
    green[:lake, :lake] = 0.30
    nir[:lake, :lake] = 0.05
    with rasterio.open(
        path, "w", driver="GTiff", height=SIDE, width=SIDE, count=4, dtype="float32",
        crs="EPSG:32643", transform=from_origin(500000.0, 2000000.0, GSD, GSD),
    ) as dst:
        for index, (band, name) in enumerate(
            [(np.full((SIDE, SIDE), 0.15, np.float32), "B04"), (green, "B03"),
             (nir, "B08"), (np.full((SIDE, SIDE), 0.20, np.float32), "B11")],
            start=1,
        ):
            dst.write(band, index)
            dst.set_band_description(index, name)
    return path


def test_highlight_survives_a_stack_with_no_blue_band(tmp_path, tool) -> None:
    """The highlight is a transparent layer over the imagery and needs no backdrop.

    It once shared a try block with the tiles, so a raster with no B02 -- which only stops
    the *backdrop* rendering -- discarded the highlight too, and the feature appeared to
    work only on files that happened to carry blue.
    """
    refs = [
        read_image_ref(_write_no_blue(tmp_path / "a.tif", 20)),
        read_image_ref(_write_no_blue(tmp_path / "b.tif", 40)),
    ]
    result = tool.run(ToolRequest(query="what changed?", images=refs))
    assert result.ok, result.error
    assert result.evidence.highlight_path is not None, "highlight must not need a blue band"
    assert result.evidence.highlight_path.exists()
    # The false-colour fallback means the tiles render too, rather than being skipped.
    assert result.evidence.mask_path is not None


def test_highlight_marks_the_class_that_actually_changed(tmp_path, tool) -> None:
    """Choosing the class alphabetically highlighted built-up on a scene where only the
    water moved, producing an empty overlay on a visibly changed pair."""
    refs = [
        read_image_ref(_write(tmp_path / "a.tif", 20)),
        read_image_ref(_write(tmp_path / "b.tif", 40)),
    ]
    result = tool.run(ToolRequest(query="what changed?", images=refs))
    assert result.ok, result.error

    from PIL import Image

    highlight = np.array(Image.open(result.evidence.highlight_path))
    assert highlight.shape[2] == 4, "the highlight must be RGBA to composite over imagery"
    marked = int(np.count_nonzero(highlight[:, :, 3]))
    expected = 40 * 40 - 20 * 20  # the water square growing
    assert marked == pytest.approx(expected, rel=0.02), (
        f"highlight marked {marked} px; the water change is {expected} px"
    )
