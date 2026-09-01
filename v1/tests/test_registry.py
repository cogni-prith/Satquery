"""The registry is the router's lookup table. It must be deterministic and honest."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from satquery.models.registry import BUILTIN_SPECS, REGISTRY, ToolRegistry
from satquery.serve.contracts import InputConfig, Modality, TaskType, ToolSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_builtin_spec_is_registered() -> None:
    assert len(REGISTRY.list_specs()) == len(BUILTIN_SPECS)


def test_specs_are_listed_in_a_stable_order() -> None:
    names = [spec.name for spec in REGISTRY.list_specs()]
    assert names == sorted(names)


def test_the_mandatory_tasks_all_have_a_tool() -> None:
    covered = {spec.task for spec in REGISTRY.list_specs()}
    for required in (TaskType.VQA, TaskType.CHANGE_VQA, TaskType.FUSION_EXTRACTION):
        assert required in covered


def test_duplicate_registration_is_rejected_unless_replacing() -> None:
    registry = ToolRegistry()
    spec = BUILTIN_SPECS[0]
    registry.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)
    registry.register(spec, replace=True)


def test_get_spec_defaults_to_the_highest_version() -> None:
    registry = ToolRegistry()
    base = BUILTIN_SPECS[0]
    registry.register(base.model_copy(update={"version": "0.1.0"}))
    registry.register(base.model_copy(update={"version": "0.10.0"}))
    # String comparison would pick 0.1.0; numeric comparison picks 0.10.0.
    assert registry.get_spec(base.name).version == "0.10.0"


def test_unknown_tool_raises_key_error() -> None:
    with pytest.raises(KeyError, match="no tool named"):
        REGISTRY.get_spec("nope.nothing")


def test_candidates_for_a_single_image_offer_the_single_image_tasks() -> None:
    specs = REGISTRY.candidates(
        input_config=InputConfig.SINGLE, modalities=[Modality.MULTISPECTRAL], gsd_m=10.0
    )
    tasks = {spec.task for spec in specs}
    assert TaskType.VQA in tasks
    assert TaskType.CAPTION in tasks
    assert TaskType.CHANGE_VQA not in tasks


def test_candidates_for_a_bi_temporal_pair_offer_only_change_tasks() -> None:
    specs = REGISTRY.candidates(
        input_config=InputConfig.BI_TEMPORAL_PAIR,
        modalities=[Modality.OPTICAL_RGB, Modality.OPTICAL_RGB],
        gsd_m=0.5,
    )
    assert specs
    assert all(
        spec.task in {TaskType.CHANGE_VQA, TaskType.CHANGE_MASK, TaskType.CHANGE_DESCRIPTION}
        for spec in specs
    )


def test_candidates_for_a_cross_modal_pair_offer_fusion() -> None:
    specs = REGISTRY.candidates(
        input_config=InputConfig.CROSS_MODAL_PAIR,
        modalities=[Modality.MULTISPECTRAL, Modality.SAR],
    )
    assert "fusion.extraction" in {spec.name for spec in specs}


def test_a_tool_must_accept_every_input_modality() -> None:
    # change.vqa_head does not accept SAR, so a SAR-containing pair must exclude it.
    specs = REGISTRY.candidates(
        input_config=InputConfig.BI_TEMPORAL_PAIR,
        modalities=[Modality.OPTICAL_RGB, Modality.SAR],
    )
    assert "change.vqa_head" not in {spec.name for spec in specs}


def test_candidates_are_ordered_deterministically() -> None:
    args = {"input_config": InputConfig.SINGLE, "modalities": [Modality.MULTISPECTRAL]}
    assert REGISTRY.candidates(**args) == REGISTRY.candidates(**args)


def test_unknown_gsd_does_not_shrink_the_candidate_set() -> None:
    known = REGISTRY.candidates(
        input_config=InputConfig.SINGLE, modalities=[Modality.MULTISPECTRAL], gsd_m=10.0
    )
    unknown = REGISTRY.candidates(
        input_config=InputConfig.SINGLE, modalities=[Modality.MULTISPECTRAL], gsd_m=None
    )
    assert {s.name for s in known} <= {s.name for s in unknown}


def test_an_unbound_tool_raises_not_implemented_rather_than_faking() -> None:
    registry = ToolRegistry()
    registry.register(BUILTIN_SPECS[0])
    with pytest.raises(NotImplementedError, match="no implementation bound"):
        registry.get(BUILTIN_SPECS[0].name)


def test_implemented_flags_match_reality() -> None:
    implemented = [spec.name for spec in REGISTRY.list_specs(implemented_only=True)]
    assert implemented == [
        "change.vqa_head",
        "fusion.extraction",
        "indices.deterministic",
        "vlm.caption",
        "vlm.change_description",
        "vlm.grounding",
        "vlm.vqa",
    ]


def test_json_cli_emits_parseable_specs() -> None:
    # This is literally what `make registry` runs and what the backend consumes.
    result = subprocess.run(
        [sys.executable, "-m", "satquery.models.registry", "--json"],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["tool_count"] == len(BUILTIN_SPECS)
    assert payload["contract_version"]
    for entry in payload["tools"]:
        ToolSpec.model_validate(entry)  # round-trips back into the model
