"""Dataset loaders. Real-data tests are skipped unless the corpus is present.

The suite must pass on a clean machine with nothing downloaded, so anything that needs
a real corpus is guarded by a skip. What is NOT skipped is the pure mapping logic:
coordinate conversion and schema conformance are tested against synthetic annotations
written into a tmp_path, so a regression is caught with no data on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from satquery.data.datasets.vrsbench import VRSBenchDataset
from satquery.data.schema import AnswerType
from satquery.serve.contracts import ImageRef, Modality, TaskType
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


# -- BigEarthNet.txt ----------------------------------------------------------------------


def test_choices_split_on_markers_not_commas() -> None:
    """An option can itself contain a comma, and the first marker follows the question
    mark rather than a comma. Splitting on commas found one option where there were four,
    which the loader then rejected loudly rather than training on a wrong target."""
    from satquery.data.datasets.bigearthnet_txt import parse_choices

    question = (
        "Which classes share a boundary? a) Broad-leaved forest and Pastures, "
        "b) Coastal wetlands and Coniferous forest, c) Coniferous forest and Mixed forest, "
        "d) Arable land and Pastures"
    )
    choices = parse_choices(question)
    assert len(choices) == 4
    assert choices[3] == "Arable land and Pastures"


def test_choices_survive_a_comma_inside_an_option() -> None:
    from satquery.data.datasets.bigearthnet_txt import parse_choices

    choices = parse_choices("Pick one: a) Arable land, pastures and forest, b) Water bodies")
    assert choices == ["Arable land, pastures and forest", "Water bodies"]


def test_no_markers_yields_no_choices() -> None:
    """An empty list is the honest answer; the caller must treat it as unusable rather
    than inventing options."""
    from satquery.data.datasets.bigearthnet_txt import parse_choices

    assert parse_choices("Is there water in this image?") == []


def test_normalised_boxes_become_absolute_pixels() -> None:
    """The corpus writes the unit square; `BoundingBox` is defined in raster pixels because
    that is how grounding is scored. Passing the normalised numbers through unchanged would
    put every box inside the top-left pixel -- a silent zero on every grounding metric, with
    data that looks well-formed at every intermediate step."""
    from satquery.data.datasets.bigearthnet_txt import parse_normalised_box

    box = parse_normalised_box("[0.64 0.0, 1.0 0.71]", 120, 120, "point")
    assert box is not None
    assert box.x_min == pytest.approx(76.8)
    assert box.x_max == pytest.approx(120.0)
    assert box.y_max == pytest.approx(85.2)


def test_a_degenerate_box_is_dropped_rather_than_emitted() -> None:
    """A zero-area target trains on nothing; better to drop the row."""
    from satquery.data.datasets.bigearthnet_txt import parse_normalised_box

    assert parse_normalised_box("[0.5 0.5, 0.5 0.5]", 120, 120, "point") is None
    assert parse_normalised_box("not a box", 120, 120, "point") is None


def test_unknown_split_is_rejected() -> None:
    from satquery.data.datasets.bigearthnet_txt import BigEarthNetTxtDataset

    with pytest.raises(ValueError, match=r"unknown BigEarthNet\.txt split"):
        BigEarthNetTxtDataset(split="nope")


def test_missing_annotations_name_the_download() -> None:
    from satquery.data.datasets.bigearthnet_txt import BigEarthNetTxtDataset

    dataset = BigEarthNetTxtDataset(root=Path("/nonexistent"), split="test")
    with pytest.raises(FileNotFoundError, match="huggingface-cli download"):
        len(dataset)


# -- the weighted mix ---------------------------------------------------------------------


class _FakeComponent:
    """A component whose samples say which component they came from."""

    def __init__(self, name: str, size: int) -> None:
        self.name, self.size = name, size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        from satquery.data.schema import Sample

        return Sample(
            sample_id=f"{self.name}:{index}",
            task=TaskType.VQA,
            answer_type=AnswerType.BINARY,
            images=[ImageRef(path="/tmp/x.tif", modality=Modality.OPTICAL_RGB, gsd_m=10.0)],
            instruction="q",
            answer_text="yes",
            source=self.name,
        )


def _spec_with(weights: dict[str, float]):
    from satquery.data.mixer import MixComponent, MixSpec

    return MixSpec(
        name="test_mix",
        components=[
            MixComponent(name=name, weight=weight, config=f"{name}.yaml", splits=("train",))
            for name, weight in weights.items()
        ],
        seed=7,
    )


def test_mix_draws_in_the_declared_proportions() -> None:
    """The 40/40/20 blend is specified in the architecture; a mixer that silently drew uniformly
    would train on the wrong distribution while every log line still looked correct."""
    import collections

    from satquery.data.mixer import MixedDataset

    spec = _spec_with({"a": 0.5, "b": 0.3, "c": 0.2})
    mix = MixedDataset(
        spec, components={n: _FakeComponent(n, 1000) for n in ("a", "b", "c")}
    )

    drawn = collections.Counter(mix[i].metadata["mix_component"] for i in range(3000))
    assert drawn["a"] / 3000 == pytest.approx(0.5, abs=0.05)
    assert drawn["b"] / 3000 == pytest.approx(0.3, abs=0.05)
    assert drawn["c"] / 3000 == pytest.approx(0.2, abs=0.05)


def test_mix_draws_are_deterministic_for_an_index() -> None:
    """A DataLoader worker and a resumed run must see the same sample for the same index,
    so the draw is seeded from (mix seed, index) rather than global RNG state."""
    from satquery.data.mixer import MixedDataset

    spec = _spec_with({"a": 0.5, "b": 0.5})
    build = lambda: MixedDataset(  # noqa: E731 - a one-line factory reads better here
        spec, components={n: _FakeComponent(n, 500) for n in ("a", "b")}
    )
    first, second = build(), build()
    assert [first[i].sample_id for i in range(50)] == [second[i].sample_id for i in range(50)]


def test_mix_without_components_says_what_is_missing() -> None:
    from satquery.data.mixer import MixedDataset

    mix = MixedDataset(_spec_with({"a": 1.0}))
    with pytest.raises(RuntimeError, match="no component datasets are built"):
        len(mix)
