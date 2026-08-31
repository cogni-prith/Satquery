"""Dataset loaders. Real-data tests are skipped unless the corpus is present.

The suite must pass on a clean machine with nothing downloaded, so anything that needs
a real corpus is guarded by a skip. What is NOT skipped is the pure mapping logic:
coordinate conversion and schema conformance are tested against synthetic annotations
written into a tmp_path, so a regression is caught with no data on disk.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from satquery.data.datasets.vrsbench import VRSBenchDataset
from satquery.data.schema import AnswerType
from satquery.serve.contracts import TaskType
from satquery.utils.paths import dataset_dir

_HAS_VRSBENCH = (dataset_dir("vrsbench") / "VRSBench_train.json").is_file()
requires_vrsbench = pytest.mark.skipif(_HAS_VRSBENCH is False, reason="VRSBench not downloaded")


def _fake_corpus(root, *, width: int = 200, height: int = 100) -> None:
    """Write a minimal VRSBench-shaped corpus: one image and one record per subset."""
    images = root / "images" / "Images_val"
    images.mkdir(parents=True)
    Image.new("RGB", (width, height)).save(images / "x.png")

    (root / "VRSBench_EVAL_Cap.json").write_text(
        json.dumps([{"image_id": "x.png", "question": "Describe", "ground_truth": "a field"}])
    )
    (root / "VRSBench_EVAL_vqa.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "x.png",
                    "question": "What colour?",
                    "ground_truth": "Yellow",
                    "type": "object color",
                    "question_id": 7,
                }
            ]
        )
    )
    (root / "VRSBench_EVAL_referring.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "x.png",
                    "question": "the vehicle",
                    "ground_truth": "{<25><40><50><80>}",
                    "type": "ref",
                    "question_id": 3,
                }
            ]
        )
    )


def test_referring_boxes_convert_from_0_to_100_into_pixels(tmp_path) -> None:
    # The scale is per-axis: a non-square image would expose an x/y mix-up that a
    # 512x512 fixture cannot.
    _fake_corpus(tmp_path, width=200, height=100)
    sample = VRSBenchDataset(root=tmp_path, split="val", subset="referring")[0]

    assert sample.task is TaskType.GROUNDING
    assert sample.answer_type is AnswerType.BOXES
    box = sample.boxes[0]
    assert box.x_min == pytest.approx(0.25 * 200)
    assert box.y_min == pytest.approx(0.40 * 100)
    assert box.x_max == pytest.approx(0.50 * 200)
    assert box.y_max == pytest.approx(0.80 * 100)


def test_vqa_is_open_text_not_closed_set(tmp_path) -> None:
    # VRSBench VQA is judged by an LLM, not exact-matched. The wrong answer_type here
    # would silently select the wrong metric.
    _fake_corpus(tmp_path)
    sample = VRSBenchDataset(root=tmp_path, split="val", subset="vqa")[0]
    assert sample.answer_type is AnswerType.OPEN_TEXT
    assert sample.answer_text == "Yellow"
    assert sample.metadata["question_type"] == "object color"


def test_prompts_use_the_frozen_templates_and_an_unknown_gsd(tmp_path) -> None:
    # VRSBench PNGs carry no affine transform, so the honest token is <gsd:unknown>.
    _fake_corpus(tmp_path)
    for subset, marker in (("caption", "[caption]"), ("vqa", "[vqa]"), ("referring", "[refer]")):
        prompt = VRSBenchDataset(root=tmp_path, split="val", subset=subset)[0].prompt()
        assert prompt.startswith("<gsd:unknown> ")
        assert marker in prompt


def test_missing_corpus_names_what_is_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="VRSBench annotations not found"):
        len(VRSBenchDataset(root=tmp_path / "nope", split="val", subset="vqa"))


def test_unknown_split_or_subset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown VRSBench split"):
        VRSBenchDataset(split="nope")
    with pytest.raises(ValueError, match="unknown VRSBench subset"):
        VRSBenchDataset(subset="nope")


@requires_vrsbench
@pytest.mark.parametrize(
    ("split", "subset", "expected"),
    [
        ("train", "caption", 20264),
        ("train", "vqa", 85813),
        ("train", "referring", 36313),
        ("val", "caption", 9350),
        ("val", "vqa", 37409),
        ("val", "referring", 16159),
    ],
)
def test_real_corpus_record_counts(split: str, subset: str, expected: int) -> None:
    """Pins the counts actually present, so a re-download that truncates is caught."""
    assert len(VRSBenchDataset(split=split, subset=subset)) == expected


@requires_vrsbench
def test_real_samples_validate_against_the_unified_schema() -> None:
    for subset in ("caption", "vqa", "referring"):
        dataset = VRSBenchDataset(split="val", subset=subset)
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            sample = dataset[index]  # Sample construction validates on every access
            assert sample.images[0].path.is_file()
            assert sample.source == "vrsbench"
