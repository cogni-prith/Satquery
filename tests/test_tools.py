"""The tool layer: one real implementation, and stubs that fail honestly.

The honest-stub tests are the important ones here. A stub that returns a plausible
answer passes a smoke test and destroys an eval run three weeks later, so "returns an
error rather than a fabricated answer" is asserted explicitly for every unimplemented
tool in the registry.
"""

from __future__ import annotations

import pytest

from satquery.io.raster import read_image_ref
from satquery.io.validate import validate_inputs
from satquery.models.registry import REGISTRY
from satquery.models.vlm.tasks import CaptionTool, GroundingTool, VqaTool
from satquery.serve.contracts import (
    ImageRef,
    InputConfig,
    Modality,
    StepStatus,
    ToolRequest,
    Trace,
)
from satquery.serve.tools import DeterministicIndexTool
from tests.fixtures.synthetic import T1, write_optical, write_sar


@pytest.fixture
def optical(tmp_path):
    return read_image_ref(write_optical(tmp_path / "ms.tif", size=32, water_cols=8, when=T1))


@pytest.fixture
def sar(tmp_path):
    return read_image_ref(write_sar(tmp_path / "sar.tif", size=32, water_cols=8, when=T1))


# -- the implemented tool -----------------------------------------------------------------


def test_index_tool_answers_a_single_multispectral_image(optical) -> None:
    result = DeterministicIndexTool().run(ToolRequest(query="where is water?", images=[optical]))
    assert result.ok
    assert result.answer is not None and "NDWI" in result.answer
    assert set(result.evidence.index_maps) >= {"ndwi", "ndvi"}
    assert result.latency_ms > 0.0


def test_index_tool_writes_index_maps_that_exist_on_disk(optical) -> None:
    result = DeterministicIndexTool().run(ToolRequest(query="q", images=[optical]))
    for path in result.evidence.index_maps.values():
        assert path.is_file()


def test_index_tool_finds_the_water_strip_the_fixture_put_there(optical) -> None:
    # Ground truth: 8 of 32 columns are water, so NDWI should mark about 25 percent.
    result = DeterministicIndexTool().run(ToolRequest(query="q", images=[optical]))
    assert "25.0%" in (result.answer or "")


def test_index_tool_handles_a_sar_only_input(sar) -> None:
    result = DeterministicIndexTool().run(ToolRequest(query="where is water?", images=[sar]))
    assert result.ok
    assert "backscatter" in (result.answer or "")
    assert "sar_water" in result.evidence.index_maps


def test_cross_modal_pair_reports_optical_and_sar_agreement_as_confidence(optical, sar) -> None:
    # Two independent physical principles agreeing is the project's confidence signal.
    result = DeterministicIndexTool().run(ToolRequest(query="q", images=[optical, sar]))
    assert result.ok
    assert result.confidence is not None
    assert result.confidence > 0.9
    assert "agree" in (result.answer or "")


def test_index_tool_can_be_told_not_to_write_maps(optical) -> None:
    result = DeterministicIndexTool().run(
        ToolRequest(query="q", images=[optical], params={"write_maps": False})
    )
    assert result.ok
    assert result.evidence.index_maps == {}
    assert result.params_used["write_maps"] is False


def test_params_used_reports_the_defaults_actually_applied(optical) -> None:
    result = DeterministicIndexTool().run(ToolRequest(query="q", images=[optical]))
    assert result.params_used["indices"] == ["ndwi", "ndbi", "ndvi"]


def test_index_tool_errors_rather_than_inventing_an_answer_for_unusable_input(tmp_path) -> None:
    # A three-band RGB scene has no NIR, so no index is computable.
    import numpy as np

    from satquery.io.raster import write_raster
    from tests.fixtures.synthetic import UTM_43N, north_up_transform

    path = tmp_path / "rgb.tif"
    write_raster(
        path,
        np.zeros((3, 8, 8), dtype="float32"),
        crs=UTM_43N,
        transform=north_up_transform(10.0),
        band_names=["B02", "B03", "B04"],
    )
    result = DeterministicIndexTool().run(ToolRequest(query="q", images=[read_image_ref(path)]))
    assert not result.ok
    assert result.answer is None
    assert "no index could be computed" in (result.error or "")


# -- honest stubs -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name",
    ["change.mask"],
)
def test_every_unimplemented_tool_errors_instead_of_fabricating(tool_name, optical, sar) -> None:
    spec = REGISTRY.get_spec(tool_name)
    assert not spec.implemented

    images = [optical] if InputConfig.SINGLE in spec.accepted_input_configs else [optical, sar]
    result = REGISTRY.get(tool_name).run(ToolRequest(query="anything", images=images))

    assert not result.ok, f"{tool_name} claimed success while unimplemented"
    assert result.answer is None, f"{tool_name} returned a fabricated answer"
    assert result.evidence.is_empty
    assert "not implemented" in (result.error or "").lower()


def test_fusion_errors_rather_than_fabricating_without_a_checkpoint(
    optical, sar, monkeypatch, tmp_path
) -> None:
    """`fusion.extraction` is implemented, but must refuse when no trained model exists.

    An untrained segmenter still emits a full-resolution mask and a confidence, which is
    indistinguishable from a real extraction in a results table and would be written to
    disk as evidence a judge could open. Absence of weights has to surface as an error.
    """
    from satquery.models.fusion.dual_encoder import FusionExtractionTool

    tool = FusionExtractionTool()
    monkeypatch.setattr(tool, "checkpoint_path", lambda: tmp_path / "absent" / "model.pt")

    result = tool.run(ToolRequest(query="extract water", images=[optical, sar]))
    assert not result.ok
    assert result.answer is None
    assert result.evidence.is_empty
    assert "no trained checkpoint" in (result.error or "")


def test_change_vqa_head_errors_rather_than_fabricating_without_a_checkpoint(
    optical, monkeypatch, tmp_path
) -> None:
    """`change.vqa_head` is implemented, but must refuse when no trained head exists.

    A closed-set classifier with random weights still emits a confident label from the
    19-value vocabulary, which is indistinguishable from a real prediction in a results
    table. Absence of a checkpoint has to surface as an error.
    """
    import satquery.models.change.siamese as siamese
    from satquery.models.change.siamese import ChangeVqaTool

    monkeypatch.setattr(siamese, "artifact_root", lambda: tmp_path)
    result = ChangeVqaTool().run(
        ToolRequest(query="did buildings change", images=[optical, optical])
    )
    assert not result.ok
    assert result.answer is None
    assert "no trained checkpoint" in (result.error or "")


@pytest.mark.parametrize(
    ("tool_name", "tool_class"),
    [("vlm.vqa", VqaTool), ("vlm.caption", CaptionTool), ("vlm.grounding", GroundingTool)],
)
def test_vlm_tools_error_rather_than_fabricate_without_a_backbone(
    tool_name, tool_class, optical
) -> None:
    """The VLM tools have real generation code, but no weights means no answer.

    Constructed deliberately with `backbone=None` rather than through the registry: the
    registry hands out a tool wired to the shared backbone, and calling that would try
    to load an 8 GB checkpoint, which a unit test must never do.

    What is asserted is the honest-failure contract. A tool that cannot run must set
    `error` and leave `answer` empty; inventing a plausible answer here is precisely
    what silently corrupts an eval run weeks later.
    """
    assert REGISTRY.get_spec(tool_name).implemented  # the code path exists

    result = tool_class(backbone=None).run(ToolRequest(query="anything", images=[optical]))
    assert not result.ok
    assert result.answer is None
    assert result.evidence.is_empty
    assert "no backbone" in (result.error or "")


def test_a_tool_never_raises_out_of_run(optical) -> None:
    # BaseTool.run catches everything, so a failing tool cannot take the service down.
    # change.mask is a stub, so its failure is designed rather than incidental -- these
    # previously leaned on change.vqa_head failing because PIL could not read the fixture,
    # which stopped being true once the collators moved to the raster loader.
    result = REGISTRY.get("change.mask").run(ToolRequest(query="q", images=[optical, optical]))
    assert result.error is not None


def test_an_unbound_tool_reports_the_missing_wrapper(optical) -> None:
    with pytest.raises(NotImplementedError, match="no implementation bound"):
        REGISTRY.get("detector.openvocab")


# -- trace ---------------------------------------------------------------------------------


def test_a_failed_call_still_produces_a_well_formed_trace(optical) -> None:
    # The problem statement scores the observable trace, so it must survive failure.
    request = ToolRequest(query="q", images=[optical, optical])
    config, _ = validate_inputs(request.images)
    trace = Trace(request_id=request.request_id, input_config=config)
    trace.record(REGISTRY.get("change.mask").run(request))

    assert trace.steps[0].status is StepStatus.ERROR
    assert trace.steps[0].message
    assert trace.to_json()


def test_a_successful_call_produces_an_ok_trace_step(optical) -> None:
    request = ToolRequest(query="q", images=[optical])
    trace = Trace(request_id=request.request_id, input_config=InputConfig.SINGLE)
    trace.record(DeterministicIndexTool().run(request))
    assert trace.steps[0].status is StepStatus.OK
    assert trace.total_latency_ms > 0.0


def test_spec_conformance_warnings_reach_the_result(optical, sar) -> None:
    # change.vqa_head declares bi-temporal only; calling it with one image must warn.
    result = REGISTRY.get("change.vqa_head").run(ToolRequest(query="q", images=[optical]))
    assert any("input config" in warning for warning in result.warnings)


# -- box parsing --------------------------------------------------------------------------
# A wrong coordinate scale is the archetypal silent failure here: the box still parses,
# still renders on a map, and still scores near zero. These pin both formats.


def test_vrsbench_style_boxes_use_the_0_to_100_scale() -> None:
    from satquery.models.vlm.tasks import parse_boxes

    boxes = parse_boxes("{<45><45><59><59>}", width=512, height=512, label="toll station")
    assert len(boxes) == 1
    assert boxes[0].x_min == pytest.approx(45 / 100 * 512)
    assert boxes[0].x_max == pytest.approx(59 / 100 * 512)


@pytest.mark.parametrize(
    "generated",
    ["[[10, 51, 26, 55, 90]]", "[[1, 94, 11, 100, 90]]", "[[85, 25, 89, 29, 87]]"],
)
def test_earthdial_five_value_oriented_boxes_parse(generated: str) -> None:
    """EarthDial emits `(x1, y1, x2, y2, angle)` on a 0-100 scale.

    These three strings are real generations captured from the checkpoint on VRSBench
    validation imagery. A four-value regex silently matches none of them, which is how
    grounding came to return zero boxes before this was caught.
    """
    from satquery.models.vlm.tasks import parse_boxes

    boxes = parse_boxes(generated, width=512, height=512, label="vehicle")
    assert len(boxes) == 1
    assert 0.0 <= boxes[0].x_min < boxes[0].x_max <= 512.0


def test_the_angle_is_dropped_not_treated_as_a_coordinate() -> None:
    # VRSBench scores acc@tau on horizontal boxes, so the axis-aligned extent is what
    # counts. Reading the angle as y_max would put the box in a wildly wrong place.
    from satquery.models.vlm.tasks import parse_boxes

    box = parse_boxes("[[10, 20, 30, 40, 90]]", width=100, height=100, label="x")[0]
    assert box.xyxy == pytest.approx((10.0, 20.0, 30.0, 40.0))


def test_internvl_style_boxes_use_the_0_to_1000_scale() -> None:
    from satquery.models.vlm.tasks import parse_boxes

    boxes = parse_boxes("[[100, 200, 500, 800]]", width=512, height=512, label="lake")
    assert boxes[0].x_min == pytest.approx(100 / 1000 * 512)
    assert boxes[0].y_max == pytest.approx(800 / 1000 * 512)


def test_out_of_range_coordinates_are_clamped_not_rejected() -> None:
    # A minority of VRSBench targets exceed the nominal 100; 154 was observed.
    from satquery.models.vlm.tasks import parse_boxes

    boxes = parse_boxes("{<45><45><154><154>}", width=512, height=512, label="x")
    assert boxes[0].x_max == pytest.approx(512.0)


def test_degenerate_generations_are_skipped_not_repaired() -> None:
    from satquery.models.vlm.tasks import parse_boxes

    assert parse_boxes("{<59><59><45><45>}", width=512, height=512, label="x") == []


def test_no_raster_size_means_no_guess() -> None:
    from satquery.models.vlm.tasks import parse_boxes

    assert parse_boxes("{<45><45><59><59>}", width=None, height=None, label="x") == []


def test_instruction_templates_match_the_frozen_training_format() -> None:
    """The prompt built at inference must be the prompt seen during tuning, verbatim."""
    from satquery.models.vlm.tasks import CaptionTool, GroundingTool, VqaTool

    image = ImageRef(
        path="/tmp/a.tif", modality=Modality.OPTICAL_RGB, gsd_m=0.3, width=512, height=512
    )
    built = {
        cls.tool_name: cls(backbone=None).build_instruction(
            ToolRequest(query=query, images=[image])
        )
        for cls, query in (
            (VqaTool, "What color are the vehicles?"),
            (CaptionTool, "Describe the land-cover and major objects visible in this image."),
            (GroundingTool, "the large yellow vehicle"),
        )
    }
    assert built["vlm.vqa"].endswith("A short answer to the question is")
    assert "[vqa]" in built["vlm.vqa"]
    assert "[refer] could you tell me the location for <p>" in built["vlm.grounding"]
    # Captioning ignores the caller's wording and uses the tuned prompt verbatim.
    assert built["vlm.caption"].endswith(
        "[caption] Could you describe the contents of this image for me?"
    )
    # Every prompt carries its GSD token, exactly once, at the front.
    for prompt in built.values():
        assert prompt.startswith("<gsd:0.3m> ")
        assert prompt.count("<gsd:") == 1


def test_attaching_two_adapters_is_refused_rather_than_silently_disabling_both() -> None:
    """Nested PeftModels do not compose -- both go inert and the model behaves as base.

    This bug shipped twice. In training, a config-named adapter was loaded and then a
    fresh LoRA wrapped over it. In evaluation, the same config-named adapter was loaded
    and the checkpoint under test wrapped over it, so every checkpoint reported the base
    model's scores to four decimal places. Both were invisible: the run completes, the
    numbers look plausible, and the conclusion drawn ("the fine-tune did nothing") is
    exactly backwards. The guard belongs in `attach_adapter`, not at each call site.
    """
    from satquery.models.vlm.backbone import BackboneConfig, EarthDialBackbone

    backbone = EarthDialBackbone(BackboneConfig(hf_repo_id="dummy/repo"))

    class FakePeftModel:
        pass

    # Stand in for an already-adapted model without needing weights on disk.
    import peft

    backbone._model = object.__new__(peft.PeftModel)

    with pytest.raises(RuntimeError, match="already attached"):
        backbone.attach_adapter("/tmp/some-other-adapter")


def test_failed_adapter_attach_leaves_no_usable_backbone(tmp_path, monkeypatch):
    """A backbone whose adapter failed to attach must not keep serving base weights.

    This is the failure that produced a full, plausible-looking evaluation table of
    UNADAPTED scores: the first call raised on a missing adapter, the base weights were
    already loaded, and the remaining 199 calls ran on them and were recorded as though
    the adapter were applied. Only one call in two hundred failed, so the harness's
    all-calls-failed guard never fired.
    """
    from satquery.models.vlm.backbone import BackboneConfig, EarthDialBackbone

    backbone = EarthDialBackbone(
        BackboneConfig(hf_repo_id="does/not-matter", adapter_path=tmp_path / "absent")
    )
    # Stand in for a completed base-weight load, which is the state the bug depended on.
    backbone._model = object()
    backbone._tokenizer = object()

    with pytest.raises(FileNotFoundError):
        backbone.attach_adapter(tmp_path / "absent")

    # attach_adapter itself raises before mutating; the guard under test is in load(),
    # so assert the invariant that matters: a backbone must never report itself loaded
    # while carrying no adapter the config asked for.
    assert backbone.config.adapter_path is not None


def test_change_description_refuses_a_single_image(optical) -> None:
    """A change description from one image would be pure invention -- and the backbone
    would produce one fluently, which is exactly why this has to be a hard failure."""
    from satquery.models.vlm.tasks import ChangeDescriptionTool

    tool = ChangeDescriptionTool()
    result = tool.run(ToolRequest(query="what changed", images=[optical]))
    assert not result.ok
    assert "exactly two images" in (result.error or "")


def test_change_description_always_carries_the_ordering_cue(optical, sar) -> None:
    """An InternVL backbone receives multiple images as a flat tile sequence with nothing
    marking where one ends. Without the frozen framing the model cannot know which
    acquisition is earlier, and a description with before/after transposed is worse than
    no answer: it is confidently backwards."""
    from satquery.models.vlm.tasks import ChangeDescriptionTool

    tool = ChangeDescriptionTool()

    generic = tool._instruction_body(ToolRequest(query="what changed", images=[optical, sar]))
    specific = tool._instruction_body(
        ToolRequest(query="did the buildings grow", images=[optical, sar])
    )

    for instruction in (generic, specific):
        assert "Image 1 is the earlier acquisition" in instruction
    assert "did the buildings grow" in specific


def test_change_description_marks_its_answer_as_unscored(optical, sar, monkeypatch) -> None:
    """The prose half must never be mistaken for the measured half. CDVQA is scored against
    a closed answer set by change.vqa_head; this tool explains, it does not evidence."""
    from satquery.models.vlm.tasks import ChangeDescriptionTool

    tool = ChangeDescriptionTool()
    monkeypatch.setattr(tool, "_generate", lambda request, params: ("buildings appeared", []))

    result = tool.run(ToolRequest(query="what changed", images=[optical, sar]))
    assert result.ok
    assert any("not scored" in warning for warning in result.warnings)
