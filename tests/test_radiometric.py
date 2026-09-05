"""Radiometric change on RGB pairs: the path that answers when no spectral index can."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from satquery.io.raster import read_image_ref
from satquery.models.indices.radiometric import RadiometricChangeTool
from satquery.models.registry import REGISTRY
from satquery.preprocess.rgb_change import rgb_change_mask
from satquery.serve.contracts import ToolRequest

SIDE = 120


@pytest.fixture
def tool() -> RadiometricChangeTool:
    return RadiometricChangeTool(REGISTRY.get_spec("change.radiometric"))


def _write_rgb(path, array: np.ndarray):
    """A plain three-band RGB raster: no NIR, no transform. A screenshot, in effect."""
    with rasterio.open(
        path, "w", driver="GTiff", height=SIDE, width=SIDE, count=3, dtype="uint8"
    ) as dst:
        dst.write(array.transpose(2, 0, 1))
    return path


def _scene(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(60, 180, (SIDE, SIDE, 3)).astype(np.uint8)


def test_exposure_change_alone_is_not_reported_as_change() -> None:
    """Two screenshots of one place differ in exposure before anything has moved.

    Differencing raw values measures the camera as loudly as the ground, which is why each
    image is normalised into its own statistics first.
    """
    base = _scene()
    brighter = np.clip(base.astype(int) * 1.25 + 18, 0, 255).astype(np.uint8)
    mask, _ = rgb_change_mask(base, brighter)
    assert mask.sum() == 0, "a pure exposure shift must not register as change"


def test_a_real_block_is_found_at_roughly_its_true_size() -> None:
    base = _scene()
    changed = base.copy()
    changed[30:70, 30:70] = [220, 40, 40]
    mask, _ = rgb_change_mask(base, changed)
    assert mask.sum() == pytest.approx(40 * 40, rel=0.25)


def test_normalisation_is_robust_to_the_change_it_is_looking_for() -> None:
    """Mean/std normalisation was dragged by the changed region itself: a 1,600 px block
    was reported as 11,340 because every unchanged pixel also shifted. Median/MAD barely
    moves."""
    base = _scene()
    changed = base.copy()
    changed[30:70, 30:70] = [220, 40, 40]
    mask, _ = rgb_change_mask(base, changed)
    assert mask.mean() < 0.25, f"{mask.mean():.0%} of the scene flagged for an 11% block"


def _write_sized(path, side: int, crs: str | None = None):
    """A raster of an arbitrary size, optionally georeferenced."""
    extra = (
        {"crs": crs, "transform": rasterio.transform.from_origin(0, 0, 10, 10)}
        if crs is not None
        else {}
    )
    with rasterio.open(
        path, "w", driver="GTiff", height=side, width=side, count=3, dtype="uint8", **extra
    ) as dst:
        dst.write(np.zeros((3, side, side), dtype="uint8"))
    return path


def test_two_hand_cropped_screenshots_are_fitted_and_say_so(tmp_path, tool) -> None:
    """The imagery people actually upload: two crops of one place, sizes not equal.

    Refusing was correct and useless -- it hands the alignment back to the person who came
    here to avoid doing it. Neither raster is georeferenced, so there is no measured
    alignment to protect, and the fit is done and declared.
    """
    refs = [
        read_image_ref(_write_rgb(tmp_path / "a.tif", _scene())),
        read_image_ref(_write_sized(tmp_path / "small.tif", 40)),
    ]
    result = tool.run(ToolRequest(query="what changed", images=refs))
    assert result.ok, result.error
    assert any("scaled onto" in w for w in result.warnings), result.warnings


def test_a_georeferenced_mismatch_is_still_refused(tmp_path, tool) -> None:
    """With a CRS the misalignment is measurable, so fitting frames would discard it."""
    refs = [
        read_image_ref(_write_sized(tmp_path / "geo.tif", 120, crs="EPSG:32643")),
        read_image_ref(_write_sized(tmp_path / "small.tif", 40)),
    ]
    result = tool.run(ToolRequest(query="what changed", images=refs))
    assert not result.ok
    assert "one grid" in (result.error or "")


def test_an_rgb_pair_is_answered_and_carries_its_limits(tmp_path, tool) -> None:
    """The whole point: a screenshot pair is no longer a dead end -- but the answer must
    never read as a land-cover claim."""
    base = _scene()
    changed = base.copy()
    changed[20:60, 20:90] = [30, 30, 200]
    refs = [
        read_image_ref(_write_rgb(tmp_path / "a.tif", base)),
        read_image_ref(_write_rgb(tmp_path / "b.tif", changed)),
    ]
    result = tool.run(ToolRequest(query="what changed", images=refs))
    assert result.ok, result.error
    assert "changed appearance" in (result.answer or "")
    assert "cannot say what it changed into" in (result.answer or "")
    assert result.confidence is None, "one signal cannot agree with anything"
    assert result.evidence.highlight_path is not None
    # No land-cover class may appear: this tool cannot name one.
    record = result.answer_record
    assert set(record["class_proportions"]) == {"changed"}
    assert any("WHERE" in w for w in record["warnings"])


def test_threshold_is_found_per_pair_not_fixed() -> None:
    """A frozen distance cannot transfer between pairs.

    On real SECOND imagery the median per-pixel distance ranges from 0.8 to 2.2 across
    pairs. The original fixed 0.35 sat below the median on most of them and reported 90%
    of every scene as changed. Otsu asks where THIS pair's distribution splits.
    """
    from satquery.preprocess.rgb_change import change_distance, otsu_threshold

    base = _scene()
    changed = base.copy()
    changed[10:50, 10:50] = [240, 20, 20]

    # Scale one image's contrast: the absolute distances move, the split should follow.
    stretched = np.clip(base.astype(int) * 2 - 120, 0, 255).astype(np.uint8)
    stretched_changed = stretched.copy()
    stretched_changed[10:50, 10:50] = [240, 20, 20]

    plain = otsu_threshold(change_distance(base, changed))
    scaled = otsu_threshold(change_distance(stretched, stretched_changed))
    assert plain > 0 and scaled > 0

    # The measured share is what must stay stable, not the raw distance.
    a, _ = rgb_change_mask(base, changed)
    b, _ = rgb_change_mask(stretched, stretched_changed)
    assert a.mean() == pytest.approx(b.mean(), abs=0.05)


def test_the_floor_stops_otsu_splitting_pure_noise() -> None:
    """Otsu always finds a split, including in a distribution that is entirely noise, so
    an unchanged pair would come back with a confident partition of its own sensor noise
    were it not for the floor."""
    base = _scene()
    identical, _ = rgb_change_mask(base, base.copy())
    assert identical.sum() == 0, "an identical pair must report no change"


def test_the_tool_uses_the_adaptive_threshold_by_default(tmp_path, tool) -> None:
    """The tool once defaulted its own threshold parameter to the frozen constant, which
    silently disabled Otsu and put every scene back at ~90% changed."""
    base = _scene()
    changed = base.copy()
    changed[20:50, 20:50] = [230, 30, 30]
    refs = [
        read_image_ref(_write_rgb(tmp_path / "a.tif", base)),
        read_image_ref(_write_rgb(tmp_path / "b.tif", changed)),
    ]
    result = tool.run(ToolRequest(query="what changed", images=refs))
    assert result.ok, result.error
    assert result.params_used["threshold"] == "otsu (per pair)"
    assert result.answer_record["class_proportions"]["changed"] < 0.35
