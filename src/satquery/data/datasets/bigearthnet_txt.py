"""BigEarthNet.txt loader: the primary remote-sensing adaptation corpus.

BigEarthNet.txt pairs every co-registered Sentinel-1 / Sentinel-2 patch of BigEarthNet
v2.0 (reBEN) with generated text annotations. The published corpus is one parquet of
**9,553,962 rows** over four task families, joined to imagery by `patch_id` (Sentinel-2)
and `s1_name` (Sentinel-1) -- the same keys the converted LMDB is indexed on, so no
filename surgery is needed anywhere.

    binary        3,625,160    yes/no questions
    mcq           3,259,184    four inline options, answer is a letter
    bounding box  2,205,686    a point prompt, answer is a normalised box
    captioning      463,932    prose describing region, climate and land cover

This is why the fine-tune is not optional polish: it is the only source in the mix that
carries co-registered SAR and optical for the same scene, at 10 m over ten European
countries.

Two details in the published format are easy to get wrong and both are silent failures.

**Boxes are normalised, the contract is absolute pixels.** The corpus writes
``[0.64 0.0, 1.0 0.71]`` in the unit square. `BoundingBox` is defined in raster pixels,
because that is how VRSBench scores grounding. Emitting the normalised numbers unchanged
would put every box in the top-left 1x1 pixel and score zero, while looking like valid
data all the way through.

**Multiple-choice answers are letters.** The corpus writes ``d``; `Sample` requires
`answer_text` to be one of `choices`. The options live inline in the question text, so
they are parsed out and the letter is resolved to its option.

Only 10 m Sentinel imagery lives here, which is also the central risk: the hidden ISRO
evaluation set is sub-metre Cartosat-2S and RISAT SAR. Scale resampling in
`satquery.data.mixer` bridges that gap; this loader deliberately does not resample and
reports the true GSD so the token never lies.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from satquery.data.datasets import SampleDataset, SampleTransform
from satquery.data.schema import AnswerType, Sample
from satquery.io.reben import ReBENStore
from satquery.preprocess.constants import OPTICAL_BAND_ORDER_4
from satquery.serve.contracts import BoundingBox, ImageRef, Modality, TaskType
from satquery.utils.logging import get_logger
from satquery.utils.paths import artifact_dir, dataset_dir

__all__ = [
    "BIGEARTHNET_TXT_GSD_M",
    "BIGEARTHNET_TXT_SPLITS",
    "BIGEARTHNET_TXT_TASKS",
    "BigEarthNetTxtDataset",
    "parse_choices",
    "parse_normalised_box",
]

logger = get_logger(__name__)

#: Sentinel-1 GRD and Sentinel-2 L2A patches in reBEN are distributed at 10 m.
BIGEARTHNET_TXT_GSD_M: float = 10.0

#: Splits published in the corpus. `bench` is a small curated benchmark slice.
BIGEARTHNET_TXT_SPLITS: tuple[str, ...] = ("train", "validation", "test", "bench")

#: The four annotation families, named exactly as the corpus's `type` column spells them.
BIGEARTHNET_TXT_TASKS: tuple[str, ...] = ("binary", "mcq", "bounding box", "captioning")

#: How each family maps onto the unified schema.
_TASK_MAP: dict[str, tuple[TaskType, AnswerType]] = {
    "binary": (TaskType.VQA, AnswerType.BINARY),
    "mcq": (TaskType.VQA, AnswerType.MULTIPLE_CHOICE),
    "bounding box": (TaskType.GROUNDING, AnswerType.BOXES),
    "captioning": (TaskType.CAPTION, AnswerType.OPEN_TEXT),
}

#: Locates each `a)` .. `d)` marker. Options are the spans BETWEEN markers, not a
#: comma-delimited list: an option may itself contain commas ("Arable land, pastures and
#: forest"), and the first marker follows the question mark rather than a comma. Splitting
#: on commas gets both cases wrong, which is how the first version of this parser found
#: one option where there were four.
_CHOICE_MARKER_RE = re.compile(r"(?:^|[\s,])([a-d])\)\s*")

#: `[x1 y1, x2 y2]`, all four in the unit square.
_BOX_RE = re.compile(r"\[\s*([0-9.]+)\s+([0-9.]+)\s*,\s*([0-9.]+)\s+([0-9.]+)\s*\]")


def parse_choices(question: str) -> list[str]:
    """Extract the inline options from a multiple-choice question.

    The corpus embeds options in the prompt (`... a) Foo, b) Bar`) and answers with a bare
    letter, but `Sample` requires `answer_text` to be one of `choices`. Returns an empty
    list when the question carries no options, which the caller must treat as unusable
    rather than inventing them.
    """
    markers = list(_CHOICE_MARKER_RE.finditer(question))
    if not markers:
        return []

    options: list[str] = []
    for position, marker in enumerate(markers):
        start = marker.end()
        end = markers[position + 1].start() if position + 1 < len(markers) else len(question)
        options.append(question[start:end].strip().strip(",").strip().rstrip("."))
    return [option for option in options if option]


def parse_normalised_box(raw: str, width: int, height: int, label: str) -> BoundingBox | None:
    """Convert one `[x1 y1, x2 y2]` unit-square box into absolute pixels.

    `BoundingBox` is defined in raster pixels because that is how grounding is scored.
    Passing the normalised values straight through would place every box inside the
    top-left pixel -- a silent zero on every grounding metric, with data that looks
    well-formed at every intermediate step.

    Returns None when the string does not parse or the box is degenerate, so a malformed
    row is dropped rather than becoming a zero-area target the loss cannot use.
    """
    match = _BOX_RE.search(raw)
    if match is None:
        return None

    x_min, y_min, x_max, y_max = (float(v) for v in match.groups())
    if x_max <= x_min or y_max <= y_min:
        return None

    return BoundingBox(
        x_min=x_min * width,
        y_min=y_min * height,
        x_max=x_max * width,
        y_max=y_max * height,
        label=label,
    )


class BigEarthNetTxtDataset(SampleDataset):
    """Map-style view over the BigEarthNet.txt annotation table.

    Single responsibility: turn one annotation row into one unified `Sample`, resolving
    the Sentinel-2 patch it refers to. It performs no batching and no tokenisation.

    Patch rasters are materialised to a cache on first access, because `ImageRef` is
    path-based and a training sample must travel the same code path as an inference
    request -- that identity is what keeps the GSD token honest on both sides.
    """

    #: Value written to `Sample.source`. Never branched on downstream.
    SOURCE = "bigearthnet_txt"

    def __init__(
        self,
        root: Path | None = None,
        split: str = "train",
        tasks: tuple[str, ...] = BIGEARTHNET_TXT_TASKS,
        *,
        lmdb_path: Path | None = None,
        cache_dir: Path | None = None,
        max_rows: int | None = None,
        transform: SampleTransform | None = None,
    ) -> None:
        """Configure the view without touching disk.

        Args:
            root: Corpus directory holding `annotations/BigEarthNet.txt.parquet`.
            split: One of `BIGEARTHNET_TXT_SPLITS`.
            tasks: Annotation families to keep, a subset of `BIGEARTHNET_TXT_TASKS`.
            lmdb_path: Converted reBEN LMDB. Defaults to `<root>/../bigearthnet_lmdb`.
            cache_dir: Where patch rasters are materialised.
            max_rows: Cap the index. The full train split is 4.7M rows; a cap makes a
                smoke run tractable. Rows are taken by stride so the subset spans the
                whole corpus rather than one tile's worth of annotations.
            transform: Optional `Sample`-to-`Sample` rewrite applied last.

        Raises:
            ValueError: Unknown split, or a task family the corpus does not publish.
        """
        if split not in BIGEARTHNET_TXT_SPLITS:
            raise ValueError(
                f"unknown BigEarthNet.txt split {split!r}; expected one of "
                f"{BIGEARTHNET_TXT_SPLITS!r}"
            )
        unknown = tuple(task for task in tasks if task not in BIGEARTHNET_TXT_TASKS)
        if unknown:
            raise ValueError(
                f"unknown BigEarthNet.txt task(s) {unknown!r}; expected a subset of "
                f"{BIGEARTHNET_TXT_TASKS!r}"
            )
        if not tasks:
            raise ValueError("tasks must name at least one annotation family")

        self.root = Path(root) if root is not None else dataset_dir(self.SOURCE)
        self.split = split
        self.tasks = tuple(tasks)
        self.max_rows = max_rows
        self.transform = transform
        self.lmdb_path = (
            Path(lmdb_path) if lmdb_path is not None else self.root.parent / "bigearthnet_lmdb"
        )
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else artifact_dir("cache", "bigearthnet_fusion", create=False)
        )
        self._store = ReBENStore(self.lmdb_path)
        self._index: Any | None = None

        logger.debug(
            "BigEarthNetTxtDataset configured: root=%s split=%s tasks=%s",
            self.root,
            self.split,
            self.tasks,
        )

    # -- index ------------------------------------------------------------------------

    def annotations_path(self) -> Path:
        """Path of the annotation table. Pure, safe to call without data present."""
        return self.root / "annotations" / "BigEarthNet.txt.parquet"

    def _load_index(self) -> Any:
        """Read the annotation table for this split and task set."""
        if self._index is not None:
            return self._index

        import pandas as pd

        path = self.annotations_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"BigEarthNet.txt annotations not found at {path}. Obtain them with "
                "`huggingface-cli download BIFOLD-BigEarthNetv2-0/BigEarthNet.txt "
                "--repo-type dataset`; nothing is downloaded automatically. The imagery is "
                "separate and lives in the converted reBEN LMDB."
            )

        frame = pd.read_parquet(
            path, columns=["patch_id", "s1_name", "input", "output", "type", "category", "split"]
        )
        frame = frame[(frame["split"] == self.split) & (frame["type"].isin(self.tasks))]

        if self.max_rows is not None and self.max_rows < len(frame):
            # A stride, never a head slice: the table is ordered by patch, so the first N
            # rows are a handful of scenes and a capped run would not be representative.
            stride = max(len(frame) // self.max_rows, 1)
            frame = frame.iloc[::stride].head(self.max_rows)

        self._index = frame.reset_index(drop=True)
        logger.info(
            "BigEarthNetTxtDataset: %d rows in split %r for tasks %s",
            len(self._index),
            self.split,
            self.tasks,
        )
        return self._index

    # -- imagery ----------------------------------------------------------------------

    def _patch_cache(self, patch_id: str) -> Path:
        """Sharded on a hash of the id; reBEN ids end in a grid row and column, so
        slicing the id itself clusters hard."""
        digest = hashlib.sha1(patch_id.encode("utf-8")).hexdigest()
        return self._cache_dir / digest[:2] / digest[2:4] / patch_id

    def _optical_ref(self, patch_id: str) -> ImageRef:
        """Materialise the Sentinel-2 patch if needed and describe it.

        Written raw, matching `bigearthnet_fusion`: the collator applies the frozen
        stretch at load time, so one cache serves both loaders and neither sees pixels the
        other would not.
        """
        import numpy as np

        from satquery.io.raster import write_raster
        from satquery.io.reben import patch_gsd_m

        cache = self._patch_cache(patch_id)
        optical_path = cache / "optical.tif"

        if not optical_path.is_file():
            cache.mkdir(parents=True, exist_ok=True)
            stack, _ = self._store.optical(patch_id, OPTICAL_BAND_ORDER_4)
            write_raster(
                optical_path, stack.astype(np.uint16), band_names=list(OPTICAL_BAND_ORDER_4)
            )

        stack, _ = self._store.optical(patch_id, OPTICAL_BAND_ORDER_4)
        height, width = stack.shape[-2:]
        return ImageRef(
            path=optical_path,
            modality=Modality.MULTISPECTRAL,
            gsd_m=patch_gsd_m(width),
            band_names=list(OPTICAL_BAND_ORDER_4),
            width=int(width),
            height=int(height),
            warnings=[
                "georeferencing is not preserved by the reBEN LMDB conversion: GSD is "
                "computed from the patch extent, but CRS and origin are unknown and are "
                "not fabricated"
            ],
        )

    # -- SampleDataset ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of annotation rows in this split after task filtering."""
        return len(self._load_index())

    def __getitem__(self, index: int) -> Sample:
        """Return annotation row `index` as a unified `Sample`.

        Raises:
            ValueError: The row's family needs data it does not carry -- a
                multiple-choice question with no parseable options, or a box that does not
                parse. Raising beats emitting a degenerate target that trains on nothing.
        """
        row = self._load_index().iloc[index]
        patch_id = str(row["patch_id"])
        family = str(row["type"])
        task, answer_type = _TASK_MAP[family]

        image = self._optical_ref(patch_id)
        question = str(row["input"])
        raw_answer = str(row["output"])

        choices: list[str] | None = None
        answer_text: str | None = raw_answer
        boxes: list[BoundingBox] = []

        if answer_type is AnswerType.MULTIPLE_CHOICE:
            choices = parse_choices(question)
            letter = raw_answer.strip().lower()[:1]
            position = "abcd".find(letter)
            if not choices or position < 0 or position >= len(choices):
                raise ValueError(
                    f"row {index} ({patch_id}): multiple-choice answer {raw_answer!r} does "
                    f"not resolve against {len(choices)} parsed option(s). The options are "
                    "embedded in the question text, so a parse failure here would silently "
                    "train on a wrong target."
                )
            answer_text = choices[position]

        elif answer_type is AnswerType.BOXES:
            box = parse_normalised_box(
                raw_answer, image.width or 120, image.height or 120, str(row["category"])
            )
            if box is None:
                raise ValueError(
                    f"row {index} ({patch_id}): could not parse a box from {raw_answer!r}. "
                    "The corpus writes normalised [x1 y1, x2 y2]; a silent failure here "
                    "would emit an empty target."
                )
            boxes = [box]
            answer_text = None

        sample = Sample(
            sample_id=f"{patch_id}:{index}",
            task=task,
            answer_type=answer_type,
            images=[image],
            instruction=question,
            answer_text=answer_text,
            choices=choices,
            boxes=boxes,
            source=self.SOURCE,
            split=self.split,
            metadata={
                "family": family,
                "category": str(row["category"]),
                "s1_name": str(row["s1_name"]),
            },
        )
        return self.transform(sample) if self.transform is not None else sample
