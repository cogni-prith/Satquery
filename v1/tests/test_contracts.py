"""The wire contract. These tests are what the backend team relies on staying true."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from satquery.serve.contracts import (
    CONTRACT_VERSION,
    BoundingBox,
    Evidence,
    ImageRef,
    InputConfig,
    Modality,
    StepStatus,
    TaskType,
    ToolRequest,
    ToolResult,
    ToolSpec,
    Trace,
)


def _image(**overrides) -> ImageRef:
    base = {"path": "/tmp/a.tif", "modality": Modality.MULTISPECTRAL, "gsd_m": 10.0}
    return ImageRef(**{**base, **overrides})


def test_contract_version_is_exported() -> None:
    assert CONTRACT_VERSION


def test_enum_members_are_stable_strings() -> None:
    # The backend serialises these; renaming a value is a breaking change.
    assert {m.value for m in Modality} == {"optical_rgb", "multispectral", "sar", "panchromatic"}
    assert {c.value for c in InputConfig} == {"single", "cross_modal_pair", "bi_temporal_pair"}
    assert {t.value for t in TaskType} == {
        "vqa",
        "caption",
        "grounding",
        "change_description",
        "change_vqa",
        "change_mask",
        "fusion_extraction",
    }
    assert {s.value for s in StepStatus} == {"ok", "warning", "error", "skipped"}


def test_degenerate_boxes_are_rejected() -> None:
    with pytest.raises(ValidationError, match="degenerate box"):
        BoundingBox(x_min=10, y_min=0, x_max=5, y_max=10, label="x")


def test_box_geometry_helpers() -> None:
    box = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=4, label="ship", score=0.9)
    assert box.xyxy == (0.0, 0.0, 10.0, 4.0)
    assert box.area == pytest.approx(40.0)


def test_image_ref_exposes_the_gsd_token_and_never_a_guess() -> None:
    assert _image(gsd_m=0.6).gsd_token == "<gsd:0.6m>"
    assert _image(gsd_m=None).gsd_token == "<gsd:unknown>"


def test_tool_request_requires_one_or_two_images() -> None:
    with pytest.raises(ValidationError):
        ToolRequest(query="q", images=[])
    with pytest.raises(ValidationError):
        ToolRequest(query="q", images=[_image(), _image(), _image()])


def test_tool_request_generates_a_request_id() -> None:
    assert len(ToolRequest(query="q", images=[_image()]).request_id) == 32


def test_extra_fields_are_rejected_so_a_typo_is_loud() -> None:
    with pytest.raises(ValidationError):
        ToolRequest(query="q", images=[_image()], tsak=TaskType.VQA)


def test_a_successful_result_must_carry_an_answer_or_evidence() -> None:
    with pytest.raises(ValidationError, match="answer or evidence"):
        ToolResult(request_id="r", tool_name="t", tool_version="0.1.0")


def test_a_failed_result_needs_neither() -> None:
    result = ToolResult(request_id="r", tool_name="t", tool_version="0.1.0", error="boom")
    assert not result.ok


def test_evidence_only_result_is_valid() -> None:
    evidence = Evidence(boxes=[BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1, label="x")])
    result = ToolResult(request_id="r", tool_name="t", tool_version="0.1.0", evidence=evidence)
    assert result.ok
    assert not evidence.is_empty


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        ToolResult(request_id="r", tool_name="t", tool_version="0.1.0", answer="a", confidence=1.5)


# -- trace ------------------------------------------------------------------------------


def _trace() -> Trace:
    return Trace(request_id="r", input_config=InputConfig.SINGLE, task=TaskType.VQA)


def test_trace_indexes_steps_and_sums_latency() -> None:
    trace = _trace()
    trace.add_step(tool_name="a", tool_version="1", latency_ms=10.0)
    trace.add_step(tool_name="b", tool_version="1", latency_ms=5.5)
    assert [step.step for step in trace.steps] == [0, 1]
    assert trace.total_latency_ms == pytest.approx(15.5)


def test_record_derives_status_from_the_result() -> None:
    trace = _trace()
    trace.record(ToolResult(request_id="r", tool_name="t", tool_version="1", answer="a"))
    trace.record(
        ToolResult(request_id="r", tool_name="t", tool_version="1", answer="a", warnings=["w"])
    )
    trace.record(ToolResult(request_id="r", tool_name="t", tool_version="1", error="boom"))
    assert [step.status for step in trace.steps] == [
        StepStatus.OK,
        StepStatus.WARNING,
        StepStatus.ERROR,
    ]


def test_trace_serialises_to_valid_json_including_latency() -> None:
    trace = _trace()
    trace.add_step(tool_name="a", tool_version="1", latency_ms=3.0)
    payload = json.loads(trace.to_json())
    assert payload["request_id"] == "r"
    assert payload["total_latency_ms"] == pytest.approx(3.0)
    assert payload["steps"][0]["tool_name"] == "a"


# -- tool spec --------------------------------------------------------------------------


def _spec(**overrides) -> ToolSpec:
    base = {
        "name": "t",
        "version": "0.1.0",
        "task": TaskType.VQA,
        "accepted_modalities": [Modality.MULTISPECTRAL],
        "accepted_input_configs": [InputConfig.SINGLE],
        "min_gsd_m": 1.0,
        "max_gsd_m": 20.0,
    }
    return ToolSpec(**{**base, **overrides})


def test_spec_rejects_an_inverted_gsd_range() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        _spec(min_gsd_m=30.0, max_gsd_m=1.0)


def test_spec_sorts_and_dedupes_its_lists_for_deterministic_json() -> None:
    spec = _spec(accepted_modalities=[Modality.SAR, Modality.MULTISPECTRAL, Modality.SAR])
    assert spec.accepted_modalities == [Modality.MULTISPECTRAL, Modality.SAR]


def test_gsd_acceptance_bounds() -> None:
    spec = _spec()
    assert spec.accepts_gsd(10.0)
    assert not spec.accepts_gsd(0.5)
    assert not spec.accepts_gsd(50.0)


def test_unknown_gsd_never_excludes_a_tool() -> None:
    # Refusing to run on the hidden evaluation set because a transform was unreadable
    # is a worse failure than running and saying so.
    assert _spec().accepts_gsd(None)


def test_accepts_combines_every_constraint() -> None:
    spec = _spec()
    assert spec.accepts(
        input_config=InputConfig.SINGLE, modalities=[Modality.MULTISPECTRAL], gsd_m=10.0
    )
    assert not spec.accepts(
        input_config=InputConfig.BI_TEMPORAL_PAIR, modalities=[Modality.MULTISPECTRAL]
    )
    assert not spec.accepts(input_config=InputConfig.SINGLE, modalities=[Modality.SAR])
    assert not spec.accepts(
        input_config=InputConfig.SINGLE, modalities=[Modality.MULTISPECTRAL], task=TaskType.CAPTION
    )


def test_spec_key_is_name_at_version() -> None:
    assert _spec().key == "t@0.1.0"


# -- round trip -------------------------------------------------------------------------


def test_image_ref_round_trips_through_json() -> None:
    # `gsd_token` is emitted by model_dump but derived, so it must be accepted and
    # ignored on the way back in. Otherwise the backend cannot replay a logged request.
    original = _image(gsd_m=0.6)
    assert ImageRef.model_validate(original.model_dump(mode="json")).gsd_m == original.gsd_m


def test_tool_request_round_trips_through_json() -> None:
    original = ToolRequest(query="Highlight the water body.", images=[_image()])
    restored = ToolRequest.model_validate(json.loads(original.model_dump_json()))
    assert restored.request_id == original.request_id
    assert restored.images[0].gsd_token == original.images[0].gsd_token


def test_tool_result_round_trips_through_json() -> None:
    original = ToolResult(
        request_id="r", tool_name="t", tool_version="0.1.0", answer="a", latency_ms=12.5
    )
    assert ToolResult.model_validate(json.loads(original.model_dump_json())) == original


def test_trace_round_trips_through_json() -> None:
    # A persisted trace must be re-readable, or the audit trail is write-only.
    original = _trace()
    original.add_step(tool_name="a", tool_version="1", latency_ms=7.25)
    restored = Trace.model_validate(json.loads(original.to_json()))
    assert restored.total_latency_ms == original.total_latency_ms
    assert restored.steps == original.steps


def test_forbidden_extras_are_still_rejected_after_the_computed_field_fix() -> None:
    # The round-trip fix must not have turned into a blanket extra="ignore".
    with pytest.raises(ValidationError):
        ImageRef(path="/tmp/a.tif", modality=Modality.SAR, gsdm=10.0)
