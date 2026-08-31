"""VRSBench loader: single-image captioning, VQA and visual grounding.

VRSBench ("A Versatile Vision-Language Benchmark for Remote Sensing Image
Understanding", Li et al.) publishes 29,614 aerial images at 512x512 with
human-verified detailed captions, 52,472 referring expressions and 123,221 QA
pairs. Canonical source: https://github.com/lx709/VRSBench

Scoring, which is why the three sub-splits stay separate here:

- captioning is scored with standard caption metrics against the verified caption,
- VQA is **open-set** and scored by an LLM judge, not by exact match,
- grounding is scored ``acc@tau`` on horizontal boxes, the same box convention as
  `satquery.serve.contracts.BoundingBox` (absolute pixels, xyxy).

VRSBench supplies two of the judged rows on its own: the second single-image task
(captioning and grounding) and a large share of the mandatory single-image VQA
score, so this is the loader the VQA and grounding registry entries train against.

The imagery is sub-metre-to-few-metre aerial, not Sentinel, which makes VRSBench
the only part of the default mix that natively sits near the Cartosat-2S end of the
resolution range. Its true GSD varies per source scene and is read from the raster
transform; it is never assumed from the 512x512 tile size.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar

from satquery.data.datasets import SampleDataset, SampleTransform
from satquery.data.schema import AnswerType, Sample
from satquery.preprocess.constants import (
    BOX_COORDINATE_SCALE,
    INSTRUCTION_CAPTION,
    INSTRUCTION_REFER_TEMPLATE,
    INSTRUCTION_VQA_TEMPLATE,
)
from satquery.serve.contracts import BoundingBox, ImageRef, Modality, TaskType
from satquery.utils.logging import get_logger
from satquery.utils.paths import dataset_dir

#: Referring targets look like `{<25><40><33><60>}`, normalised to 0-100.
_BOX_TOKEN_RE = re.compile(r"<(-?\d+(?:\.\d+)?)>")

__all__ = ["VRSBENCH_IMAGE_SIZE", "VRSBENCH_SPLITS", "VRSBENCH_SUBSETS", "VRSBenchDataset"]

logger = get_logger(__name__)

#: The three logical annotation files, each a different task and a different metric.
VRSBENCH_SUBSETS: tuple[str, ...] = ("caption", "vqa", "referring")

#: Upstream split names.
VRSBENCH_SPLITS: tuple[str, ...] = ("train", "val", "test")

#: Every published image is square at this side length.
VRSBENCH_IMAGE_SIZE: int = 512


class VRSBenchDataset(SampleDataset):
    """Map-style view over one VRSBench sub-split.

    Single responsibility: turn one VRSBench annotation record into one unified
    `Sample`, with the answer type the sub-split's metric implies -- OPEN_TEXT for
    caption and VQA, BOXES for referring. It owns no batching and no pixel work.
    """

    #: Value written to `Sample.source`. Never branched on downstream.
    SOURCE = "vrsbench"

    def __init__(
        self,
        root: Path | None = None,
        split: str = "train",
        subset: str = "vqa",
        transform: SampleTransform | None = None,
    ) -> None:
        """Configure the view without touching disk.

        Args:
            root: Corpus directory. Defaults to `dataset_dir("vrsbench")`.
            split: One of `VRSBENCH_SPLITS`.
            subset: One of `VRSBENCH_SUBSETS`, selecting task and metric.
            transform: Optional `Sample`-to-`Sample` rewrite applied last.

        Raises:
            ValueError: If `split` or `subset` is not an upstream name.
        """
        if split not in VRSBENCH_SPLITS:
            raise ValueError(
                f"unknown VRSBench split {split!r}; expected one of {VRSBENCH_SPLITS!r}"
            )
        if subset not in VRSBENCH_SUBSETS:
            raise ValueError(
                f"unknown VRSBench subset {subset!r}; expected one of {VRSBENCH_SUBSETS!r}"
            )

        self.root = Path(root) if root is not None else dataset_dir(self.SOURCE)
        self.split = split
        self.subset = subset
        self.transform = transform
        self._index: list[dict[str, Any]] | None = None
        self._sizes: dict[Path, tuple[int, int]] = {}
        logger.debug(
            "VRSBenchDataset configured: root=%s split=%s subset=%s",
            self.root,
            self.split,
            self.subset,
        )

    # -- index ------------------------------------------------------------------------

    #: Annotation file per (split, subset). Train is one file carrying all three tasks,
    #: tagged inline; the eval splits are three separate files.
    _TRAIN_FILE = "VRSBench_train.json"
    _EVAL_FILES: ClassVar[dict[str, str]] = {
        "caption": "VRSBench_EVAL_Cap.json",
        "vqa": "VRSBench_EVAL_vqa.json",
        "referring": "VRSBench_EVAL_referring.json",
    }
    #: Inline task tag in the training conversations, per subset.
    _TRAIN_TAGS: ClassVar[dict[str, str]] = {
        "caption": "[caption]",
        "vqa": "[vqa]",
        "referring": "[refer]",
    }

    _TASK_TYPES: ClassVar[dict[str, TaskType]] = {
        "caption": TaskType.CAPTION,
        "vqa": TaskType.VQA,
        "referring": TaskType.GROUNDING,
    }
    #: VRSBench VQA is open-set and judged by an LLM, not exact-matched, so it is
    #: OPEN_TEXT rather than CLOSED_SET. Getting this wrong would pick the wrong metric.
    _ANSWER_TYPES: ClassVar[dict[str, AnswerType]] = {
        "caption": AnswerType.OPEN_TEXT,
        "vqa": AnswerType.OPEN_TEXT,
        "referring": AnswerType.BOXES,
    }

    def _annotation_path(self) -> Path:
        """Path of the annotation file this sub-split reads. Pure, safe without data."""
        if self.split == "train":
            return self.root / self._TRAIN_FILE
        return self.root / self._EVAL_FILES[self.subset]

    def _image_dir(self) -> Path:
        """Directory holding this split's images.

        Upstream ships `Images_train.zip` and `Images_val.zip`; there is no separate
        test archive, so `test` reads the validation images -- which is what the
        `VRSBench_EVAL_*.json` files reference.
        """
        return self.root / "images" / ("Images_train" if self.split == "train" else "Images_val")

    def _load_index(self) -> list[dict[str, Any]]:
        """Read and filter the annotation file, caching the result on the instance.

        For `train`, the single annotation file carries all three tasks distinguished by
        an inline tag, so it is filtered down to this subset. For the eval splits each
        subset already has its own file.
        """
        if self._index is not None:
            return self._index

        path = self._annotation_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"VRSBench annotations not found at {path}. Nothing is downloaded "
                f"automatically; see scripts/download_datasets.py, and make sure "
                f"$SATQUERY_DATA_ROOT/{self.SOURCE} is populated."
            )

        with path.open(encoding="utf-8") as handle:
            records: list[dict[str, Any]] = json.load(handle)

        if self.split == "train":
            tag = self._TRAIN_TAGS[self.subset]
            records = [r for r in records if tag in r["conversations"][0]["value"]]

        logger.info(
            "VRSBench %s/%s: %d records from %s", self.split, self.subset, len(records), path.name
        )
        self._index = records
        return records

    def _image_size(self, path: Path) -> tuple[int, int]:
        """Return `(width, height)`, reading only the file header, memoised per path.

        VRSBench nominally ships 512x512 tiles, but the size is read rather than assumed:
        referring targets are normalised coordinates, so a wrong size silently misplaces
        every box.
        """
        cached = self._sizes.get(path)
        if cached is None:
            from PIL import Image

            with Image.open(path) as image:
                cached = (int(image.width), int(image.height))
            self._sizes[path] = cached
        return cached

    def _parse_boxes(self, raw: str, width: int, height: int, sample_id: str) -> list[BoundingBox]:
        """Convert a `{<x1><y1><x2><y2>}` target into absolute pixel boxes.

        Coordinates are normalised to `BOX_COORDINATE_SCALE` (100), not the 0-1000 grid
        InternVL uses. They are clamped because a minority of upstream targets exceed
        the nominal range.
        """
        values = [float(v) for v in _BOX_TOKEN_RE.findall(raw)]
        if len(values) < 4:
            raise ValueError(
                f"{sample_id}: could not parse a bounding box from ground truth {raw!r}"
            )
        x_min, y_min, x_max, y_max = values[:4]
        scale_x, scale_y = width / BOX_COORDINATE_SCALE, height / BOX_COORDINATE_SCALE
        return [
            BoundingBox(
                x_min=min(max(x_min * scale_x, 0.0), width),
                y_min=min(max(y_min * scale_y, 0.0), height),
                # Guard against zero-area targets, which BoundingBox rejects outright.
                x_max=min(max(x_max * scale_x, x_min * scale_x + 1.0), width),
                y_max=min(max(y_max * scale_y, y_min * scale_y + 1.0), height),
                label="",
                image_index=0,
            )
        ]

    def __len__(self) -> int:
        """Number of records in this split and subset."""
        return len(self._load_index())

    def __getitem__(self, index: int) -> Sample:
        """Return record `index`, normalised to the unified schema.

        Instructions are stored WITHOUT their GSD token; `Sample.prompt()` adds it. They
        keep their `[caption]` / `[vqa]` / `[refer]` tag, because that tag is part of the
        format the model was tuned on. The `<image>` placeholder is stripped: the
        backbone wrapper owns placement of that.

        VRSBench images are plain PNGs with no affine transform, so `gsd_m` is None and
        the token degrades to `<gsd:unknown>`. That is the honest answer -- the corpus
        mixes aerial sources at different resolutions, so there is no single GSD to
        assume.
        """
        record = self._load_index()[index]
        raw_task = self.subset

        if self.split == "train":
            human = record["conversations"][0]["value"]
            gpt = record["conversations"][1]["value"]
            instruction = human.replace("<image>", "").strip()
            answer = gpt.strip()
            raw_question = instruction
            image_name = record["image"]
            sample_id = f"{self.SOURCE}-train-{index}"
            question_type = raw_task
        else:
            image_name = record["image_id"]
            answer = str(record["ground_truth"]).strip()
            sample_id = f"{self.SOURCE}-{self.split}-{record.get('question_id', index)}"
            question_type = str(record.get("type", raw_task))
            question = str(record.get("question", "")).strip().rstrip("?.")
            raw_question = question
            if raw_task == "caption":
                instruction = INSTRUCTION_CAPTION
            elif raw_task == "vqa":
                instruction = INSTRUCTION_VQA_TEMPLATE.format(question=question)
            else:
                instruction = INSTRUCTION_REFER_TEMPLATE.format(phrase=question)

        image_path = self._image_dir() / image_name
        width, height = self._image_size(image_path)

        boxes: list[BoundingBox] = []
        answer_text: str | None = answer
        if raw_task == "referring":
            boxes = self._parse_boxes(answer, width, height, sample_id)
            answer_text = None

        sample = Sample(
            sample_id=sample_id,
            task=self._TASK_TYPES[raw_task],
            answer_type=self._ANSWER_TYPES[raw_task],
            images=[
                ImageRef(
                    path=image_path,
                    modality=Modality.OPTICAL_RGB,
                    gsd_m=None,
                    width=width,
                    height=height,
                )
            ],
            instruction=instruction,
            answer_text=answer_text,
            boxes=boxes,
            source=self.SOURCE,
            split=self.split,
            metadata={
                "subset": raw_task,
                "question_type": question_type,
                # The raw, UNtemplated question. `instruction` already has the frozen
                # template applied, so an evaluator passing that to a tool would template
                # it a second time and prompt the model off-distribution. Eval passes
                # this instead and lets the tool apply the template once.
                "question": raw_question,
            },
        )
        return self.transform(sample) if self.transform else sample
