"""The unified training record. Every loader must be able to produce exactly this."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from satquery.data.schema import SCHEMA_VERSION, AnswerType, Sample
from satquery.serve.contracts import BoundingBox, ImageRef, Modality, TaskType


def _image(gsd_m: float | None = 10.0) -> ImageRef:
    return ImageRef(path="/tmp/a.tif", modality=Modality.MULTISPECTRAL, gsd_m=gsd_m)


def _sample(**overrides) -> Sample:
    base = {
        "sample_id": "s1",
        "task": TaskType.VQA,
        "answer_type": AnswerType.CLOSED_SET,
        "images": [_image()],
        "instruction": "Is there water?",
        "answer_text": "yes",
        "source": "rsvqa",
    }
    return Sample(**{**base, **overrides})


def test_schema_version_is_exported() -> None:
    assert SCHEMA_VERSION


def test_prompt_carries_the_frozen_gsd_token() -> None:
    assert _sample().prompt() == "<gsd:10.0m> Is there water?"


def test_prompt_degrades_to_unknown_rather_than_guessing() -> None:
    assert _sample(images=[_image(None)]).prompt().startswith("<gsd:unknown>")


def test_primary_gsd_is_the_coarsest_of_a_pair() -> None:
    # The coarsest bounds what is resolvable once the pair is on a common grid.
    sample = _sample(
        task=TaskType.CHANGE_VQA,
        images=[_image(0.5), _image(10.0)],
        answer_type=AnswerType.CLOSED_SET,
        answer_text="water",
    )
    assert sample.primary_gsd_m == 10.0
    assert sample.is_paired


def test_boxes_answer_type_requires_boxes() -> None:
    with pytest.raises(ValidationError, match="requires `boxes`"):
        _sample(task=TaskType.GROUNDING, answer_type=AnswerType.BOXES, answer_text=None)


def test_mask_answer_type_requires_a_mask_path() -> None:
    with pytest.raises(ValidationError, match="requires `mask_path`"):
        _sample(
            task=TaskType.CHANGE_MASK,
            answer_type=AnswerType.MASK,
            answer_text=None,
            images=[_image(), _image()],
        )


def test_text_answer_types_require_answer_text() -> None:
    with pytest.raises(ValidationError, match="requires `answer_text`"):
        _sample(answer_text=None)


def test_multiple_choice_requires_the_answer_to_be_among_the_choices() -> None:
    with pytest.raises(ValidationError, match="not among choices"):
        _sample(
            answer_type=AnswerType.MULTIPLE_CHOICE,
            answer_text="maybe",
            choices=["yes", "no"],
        )


def test_multiple_choice_accepts_a_valid_answer() -> None:
    sample = _sample(
        answer_type=AnswerType.MULTIPLE_CHOICE, answer_text="yes", choices=["yes", "no"]
    )
    assert sample.answer_text in (sample.choices or [])


def test_paired_tasks_reject_a_single_image() -> None:
    with pytest.raises(ValidationError, match="needs two images"):
        _sample(task=TaskType.CHANGE_VQA, answer_text="water")


def test_single_image_tasks_reject_a_pair() -> None:
    with pytest.raises(ValidationError, match="takes one image"):
        _sample(task=TaskType.VQA, images=[_image(), _image()])


def test_box_image_index_must_be_in_range() -> None:
    box = BoundingBox(x_min=0, y_min=0, x_max=4, y_max=4, label="x", image_index=1)
    with pytest.raises(ValidationError, match="out of range"):
        _sample(
            task=TaskType.GROUNDING, answer_type=AnswerType.BOXES, answer_text=None, boxes=[box]
        )
