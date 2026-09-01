"""CDVQA loader: bi-temporal change visual question answering.

CDVQA (Yuan et al., "Change Detection Meets Visual Question Answering") is built on
the public subset of SECOND: 2,968 co-registered bi-temporal pairs at 512x512 with
122k+ automatically generated question-answer pairs and two official test splits.
Canonical source: https://github.com/YZHJessica/CDVQA

The answers are **closed** over the six land-cover classes in
`satquery.preprocess.constants.CDVQA_ANSWERS`, which is imported here and never
retyped: that tuple is also the output-layer ordering of the discriminative change
head, so a divergent copy would silently scramble a reloaded checkpoint.

That closed answer set is also why CDVQA is routed two ways. The scored
answer comes from the Siamese classification head, which beats a generative VLM on
a six-way choice and runs in milliseconds; the VLM handles free-form change
description. This loader serves both -- it emits `TaskType.CHANGE_VQA` with
`AnswerType.CLOSED_SET` records, and the description route reads the same pairs.

Every record is a pair, earlier acquisition first, matching `Sample.images`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from satquery.data.datasets import SampleDataset, SampleTransform
from satquery.data.schema import AnswerType, Sample
from satquery.preprocess.constants import CDVQA_ANSWERS, CDVQA_NOMINAL_GSD_M
from satquery.serve.contracts import ImageRef, Modality, TaskType
from satquery.utils.logging import get_logger
from satquery.utils.paths import dataset_dir

__all__ = ["CDVQA_ANSWERS", "CDVQA_IMAGE_SIZE", "CDVQA_SPLITS", "CDVQADataset"]

logger = get_logger(__name__)

#: Upstream splits. Both official test splits are scored and reported separately.
CDVQA_SPLITS: tuple[str, ...] = ("train", "val", "test_1", "test_2")

#: Annotation file prefix per split. `test_1` and `test_2` are two DIFFERENT question
#: sets over the SAME 968 image pairs, which is what "two official test splits" means.
#: They are scored separately and never averaged.
_SPLIT_PREFIX: dict[str, str] = {
    "train": "Train",
    "val": "Val",
    "test_1": "Test",
    "test_2": "Test2",
}

#: Every SECOND-derived pair is square at this side length.
CDVQA_IMAGE_SIZE: int = 512


class CDVQADataset(SampleDataset):
    """Map-style view over one CDVQA split.

    Single responsibility: turn one CDVQA question record into one unified `Sample`
    carrying an ordered bi-temporal image pair and a closed-set answer drawn from
    `CDVQA_ANSWERS`. It owns no batching, no tensor conversion and no pixel work.
    """

    #: Value written to `Sample.source`. Never branched on downstream.
    SOURCE = "cdvqa"

    def __init__(
        self,
        root: Path | None = None,
        split: str = "train",
        transform: SampleTransform | None = None,
    ) -> None:
        """Configure the view without touching disk.

        Args:
            root: Corpus directory. Defaults to `dataset_dir("cdvqa")`.
            split: One of `CDVQA_SPLITS`.
            transform: Optional `Sample`-to-`Sample` rewrite applied last.

        Raises:
            ValueError: If `split` is not an upstream name.
        """
        if split not in CDVQA_SPLITS:
            raise ValueError(f"unknown CDVQA split {split!r}; expected one of {CDVQA_SPLITS!r}")

        self.root = Path(root) if root is not None else dataset_dir(self.SOURCE)
        self.split = split
        self.transform = transform
        self._index: list[dict[str, Any]] | None = None
        logger.debug("CDVQADataset configured: root=%s split=%s", self.root, self.split)

    # -- index ------------------------------------------------------------------------

    def _annotation_path(self) -> Path:
        """Path of the questions file for this split. Pure, safe without data."""
        return self.root / f"{_SPLIT_PREFIX[self.split]}_questions.json"

    def _image_dirs(self) -> tuple[Path, Path]:
        """The `(im1, im2)` directories of the bi-temporal SECOND imagery."""
        base = self.root / "images"
        return base / "im1", base / "im2"

    def _load_index(self) -> list[dict[str, Any]]:
        """Join questions, answers and images into one flat record list, cached.

        The three files are relational: `questions[i].img_id` indexes `images`, and
        `answers` is keyed by `question_id`. They are joined here so `__getitem__` is a
        plain lookup rather than three searches.
        """
        if self._index is not None:
            return self._index

        prefix = _SPLIT_PREFIX[self.split]
        paths = {
            key: self.root / f"{prefix}_{key}.json" for key in ("questions", "answers", "images")
        }
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"CDVQA annotations not found: {missing}. Nothing is downloaded "
                "automatically; see scripts/download_datasets.py. The annotations come "
                "from github.com/YZHJessica/CDVQA and the imagery from the SECOND "
                "dataset (captain-whu.github.io/SCD)."
            )

        with paths["questions"].open(encoding="utf-8") as handle:
            questions = json.load(handle)["questions"]
        with paths["answers"].open(encoding="utf-8") as handle:
            answers = {a["question_id"]: a["answer"] for a in json.load(handle)["answers"]}
        with paths["images"].open(encoding="utf-8") as handle:
            images = {i["id"]: i["file_name"] for i in json.load(handle)["images"]}

        # SECOND imagery is a separate manual download from the annotations, so a split
        # can legitimately have every question and no pictures. Skipping the rows whose
        # imagery is absent lets a partial corpus train on what it has instead of dying
        # thousands of steps in; the count is logged so a silently tiny split is visible.
        first, _second = self._image_dirs()
        records: list[dict[str, Any]] = []
        unknown: set[str] = set()
        absent = 0
        for question in questions:
            if not question.get("active", True):
                continue
            answer = answers.get(question["id"])
            file_name = images.get(question["img_id"])
            if answer is None or file_name is None:
                continue
            if not (first / file_name).is_file():
                absent += 1
                continue
            if answer not in CDVQA_ANSWERS:
                # Never silently drop: an answer outside the closed set means the frozen
                # vocabulary is wrong, and that must surface rather than shrink the split.
                unknown.add(answer)
            records.append(
                {
                    "question": question["question"],
                    "answer": answer,
                    "type": question.get("type", ""),
                    "file_name": file_name,
                }
            )

        if absent:
            logger.warning(
                "CDVQA %s: %d of %d questions reference imagery that is not on disk and "
                "were skipped. The SECOND imagery is a separate download; see "
                "scripts/download_datasets.py.",
                self.split,
                absent,
                absent + len(records),
            )

        if unknown:
            raise ValueError(
                f"CDVQA {self.split} contains answers outside the frozen closed set: "
                f"{sorted(unknown)}. Update CDVQA_ANSWERS in preprocess/constants.py "
                "-- do not drop the records."
            )

        logger.info("CDVQA %s: %d records from %s", self.split, len(records), prefix)
        self._index = records
        return records

    def __len__(self) -> int:
        """Number of question-answer records in this split."""
        return len(self._load_index())

    def __getitem__(self, index: int) -> Sample:
        """Return record `index` as a bi-temporal `Sample`.

        Both dates are carried as two `ImageRef`s in acquisition order (im1 then im2),
        which is what `TaskType.CHANGE_VQA` requires.

        GSD note: the annotations claim `res_x: .1524m`, which is exactly six inches and
        looks like an inherited template value. SECOND is documented at 0.53 m/pixel, so
        `CDVQA_NOMINAL_GSD_M` is used and the conflict is recorded in constants.py. The
        images carry no affine transform, so nothing better is available.
        """
        record = self._load_index()[index]
        im1_dir, im2_dir = self._image_dirs()

        images = [
            ImageRef(
                path=directory / record["file_name"],
                modality=Modality.OPTICAL_RGB,
                gsd_m=CDVQA_NOMINAL_GSD_M,
                width=CDVQA_IMAGE_SIZE,
                height=CDVQA_IMAGE_SIZE,
            )
            for directory in (im1_dir, im2_dir)
        ]

        sample = Sample(
            sample_id=f"{self.SOURCE}-{self.split}-{index}",
            task=TaskType.CHANGE_VQA,
            answer_type=AnswerType.CLOSED_SET,
            images=images,
            instruction=record["question"].strip(),
            answer_text=record["answer"],
            source=self.SOURCE,
            split=self.split,
            metadata={"question_type": record["type"], "question": record["question"].strip()},
        )
        return self.transform(sample) if self.transform else sample
