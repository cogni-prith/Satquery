"""The unified training record every dataset loader emits.

BigEarthNet.txt, VRSBench, RSVQA and CDVQA have four different on-disk layouts and
four different notions of an answer. They all normalise to `Sample` here, so nothing
downstream -- mixer, collator, model, loss, eval harness -- needs to know which
dataset a record came from.

`Sample.images` deliberately reuses `ImageRef` from the serve contract rather than
defining a parallel type. A training sample and an inference request then describe
imagery through exactly the same code path, which is what keeps the GSD token
identical on both sides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from satquery.preprocess.gsd import prefix_instruction
from satquery.serve.contracts import BoundingBox, ImageRef, StrEnum, TaskType

__all__ = ["PAIRED_TASKS", "SCHEMA_VERSION", "SINGLE_IMAGE_TASKS", "AnswerType", "Sample"]

#: Bumped when `Sample` changes shape. Cached mixes record it and are rebuilt on a bump.
SCHEMA_VERSION = "1.0.0"

SINGLE_IMAGE_TASKS: frozenset[TaskType] = frozenset(
    {TaskType.VQA, TaskType.CAPTION, TaskType.GROUNDING}
)
PAIRED_TASKS: frozenset[TaskType] = frozenset(
    {
        TaskType.CHANGE_DESCRIPTION,
        TaskType.CHANGE_VQA,
        TaskType.CHANGE_MASK,
        TaskType.FUSION_EXTRACTION,
    }
)


class AnswerType(StrEnum):
    """How a sample's target is scored, which decides both the loss and the metric."""

    OPEN_TEXT = "open_text"
    """Free-form generation. VRSBench captions and open-set VQA, scored by an LLM judge."""

    CLOSED_SET = "closed_set"
    """Exact match against a fixed vocabulary. RSVQA and CDVQA."""

    MULTIPLE_CHOICE = "multiple_choice"
    """One of `choices`. BigEarthNet.txt multiple-choice VQA."""

    BINARY = "binary"
    """Yes or no. BigEarthNet.txt binary VQA and RSVQA presence questions."""

    BOXES = "boxes"
    """One or more bounding boxes. VRSBench referring expressions, scored acc@tau."""

    MASK = "mask"
    """A pixel mask on disk. LEVIR-CD and SECOND change masks."""


class Sample(BaseModel):
    """One training or evaluation record, independent of its source dataset."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(description="Stable identifier, unique within `source`.")
    task: TaskType
    answer_type: AnswerType
    images: list[ImageRef] = Field(
        min_length=1,
        max_length=2,
        description="One image, or an ordered pair. Bi-temporal pairs are earlier image first.",
    )
    instruction: str = Field(
        min_length=1,
        description="The question or directive, without its GSD token. Use `prompt()` to add it.",
    )

    answer_text: str | None = Field(
        default=None, description="Target for OPEN_TEXT, CLOSED_SET, MULTIPLE_CHOICE and BINARY."
    )
    choices: list[str] | None = Field(
        default=None, description="Candidate answers for MULTIPLE_CHOICE, in presentation order."
    )
    boxes: list[BoundingBox] = Field(
        default_factory=list, description="Target boxes for BOXES, in image pixel coordinates."
    )
    mask_path: Path | None = Field(default=None, description="Target mask raster for MASK.")

    source: str = Field(
        description=(
            "Originating dataset, e.g. 'vrsbench'. For logging, mix accounting and per-dataset "
            "eval slicing only. Never branch on this in a collator, model or loss."
        )
    )
    split: str = Field(default="train", description="Dataset split, e.g. 'train', 'test_1'.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Loader-specific extras kept out of the typed fields."
    )

    # -- derived ------------------------------------------------------------------

    @property
    def is_paired(self) -> bool:
        """True when this sample carries two images."""
        return len(self.images) == 2

    @property
    def primary_gsd_m(self) -> float | None:
        """GSD used for the prompt token: the coarsest known GSD across the images.

        The coarsest, not the finest, because it bounds what is actually resolvable in
        the pair once they are co-registered onto a common grid.
        """
        known = [image.gsd_m for image in self.images if image.gsd_m is not None]
        return max(known) if known else None

    def prompt(self) -> str:
        """Return the instruction prefixed with its frozen GSD token.

        This is the exact string the model sees. Building it here rather than in each
        loader is what stops the training and inference prompts from drifting apart.
        """
        return prefix_instruction(self.instruction, self.primary_gsd_m)

    # -- validation ---------------------------------------------------------------

    @model_validator(mode="after")
    def _check_target_matches_answer_type(self) -> Sample:
        if self.answer_type is AnswerType.BOXES:
            if not self.boxes:
                raise ValueError(f"sample {self.sample_id}: answer_type BOXES requires `boxes`")
        elif self.answer_type is AnswerType.MASK:
            if self.mask_path is None:
                raise ValueError(f"sample {self.sample_id}: answer_type MASK requires `mask_path`")
        elif self.answer_text is None:
            raise ValueError(
                f"sample {self.sample_id}: answer_type {self.answer_type.value} requires "
                "`answer_text`"
            )

        if self.answer_type is AnswerType.MULTIPLE_CHOICE:
            if not self.choices:
                raise ValueError(
                    f"sample {self.sample_id}: answer_type MULTIPLE_CHOICE requires `choices`"
                )
            if self.answer_text not in self.choices:
                raise ValueError(
                    f"sample {self.sample_id}: answer_text {self.answer_text!r} is not among "
                    f"choices {self.choices!r}"
                )
        return self

    @model_validator(mode="after")
    def _check_image_count_matches_task(self) -> Sample:
        if self.task in PAIRED_TASKS and not self.is_paired:
            raise ValueError(
                f"sample {self.sample_id}: task {self.task.value} needs two images, got "
                f"{len(self.images)}"
            )
        if self.task in SINGLE_IMAGE_TASKS and self.is_paired:
            raise ValueError(
                f"sample {self.sample_id}: task {self.task.value} takes one image, got 2"
            )
        return self

    @model_validator(mode="after")
    def _check_box_image_indices(self) -> Sample:
        limit = len(self.images)
        for box in self.boxes:
            if box.image_index >= limit:
                raise ValueError(
                    f"sample {self.sample_id}: box image_index {box.image_index} is out of range "
                    f"for {limit} image(s)"
                )
        return self
