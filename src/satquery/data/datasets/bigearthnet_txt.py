"""BigEarthNet.txt loader: the primary remote-sensing adaptation corpus.

BigEarthNet.txt pairs every co-registered Sentinel-1 / Sentinel-2 patch of
BigEarthNet v2.0 (reBEN) with generated text annotations, giving roughly 464,044
image pairs and ~9.6M annotations spread over 15 tasks: captioning, binary VQA,
multiple-choice VQA and referring-expression detection. Everything is Sentinel at
10 m ground sampling distance over 10 European countries.

Canonical source: https://txt.bigearth.net

.. warning::
   ``CLAUDE.md`` cites arXiv **2603.29630** for this dataset. That identifier is
   **UNVERIFIED** and structurally suspicious (the ``2603`` YYMM prefix reads as
   March 2026, and the serial is longer than arXiv's five digits). Confirm it
   against the real listing before it appears in any writeup, poster or paper.

This corpus is why the fine-tune is not optional polish: it is the only source in
the mix that carries co-registered SAR and optical for the same scene, so it
supplies both the LoRA adaptation signal and the fusion tool's training pairs.

Only 10 m Sentinel imagery lives here, which is also the central risk: the hidden
ISRO evaluation set is sub-metre Cartosat-2S and RISAT SAR. Scale resampling in
`satquery.data.mixer` is what bridges that gap; this loader deliberately does not
resample and reports the true 10 m GSD so the token never lies.
"""

from __future__ import annotations

from pathlib import Path

from satquery.data.datasets import SampleDataset, SampleTransform
from satquery.data.schema import Sample
from satquery.utils.logging import get_logger
from satquery.utils.paths import dataset_dir

__all__ = ["BIGEARTHNET_TXT_GSD_M", "BIGEARTHNET_TXT_TASKS", "BigEarthNetTxtDataset"]

logger = get_logger(__name__)

#: Sentinel-1 GRD and Sentinel-2 L2A patches in reBEN are distributed at 10 m.
BIGEARTHNET_TXT_GSD_M: float = 10.0

#: Annotation task families published with the corpus, as named upstream. The 15
#: upstream tasks collapse into these four families once mapped onto `TaskType`.
BIGEARTHNET_TXT_TASKS: tuple[str, ...] = (
    "caption",
    "vqa_binary",
    "vqa_multiple_choice",
    "referring_expression",
)


class BigEarthNetTxtDataset(SampleDataset):
    """Map-style view over the BigEarthNet.txt annotation index.

    Single responsibility: turn one BigEarthNet.txt annotation row into one unified
    `Sample`, resolving the Sentinel-1 and Sentinel-2 patch paths that row refers to.
    It performs no batching, no tokenisation and no pixel work.
    """

    #: Value written to `Sample.source`. Never branched on downstream.
    SOURCE = "bigearthnet_txt"

    def __init__(
        self,
        root: Path | None = None,
        split: str = "train",
        tasks: tuple[str, ...] = BIGEARTHNET_TXT_TASKS,
        transform: SampleTransform | None = None,
    ) -> None:
        """Configure the view without touching disk.

        Args:
            root: Corpus directory. Defaults to `dataset_dir("bigearthnet_txt")`.
            split: Upstream split name, one of `train`, `validation`, `test`.
            tasks: Annotation families to keep, a subset of `BIGEARTHNET_TXT_TASKS`.
            transform: Optional `Sample`-to-`Sample` rewrite applied last, used by the
                mixer for scale resampling.

        Raises:
            ValueError: If `tasks` names a family the corpus does not publish.
        """
        unknown = tuple(task for task in tasks if task not in BIGEARTHNET_TXT_TASKS)
        if unknown:
            raise ValueError(
                f"unknown BigEarthNet.txt task(s) {unknown!r}; "
                f"expected a subset of {BIGEARTHNET_TXT_TASKS!r}"
            )
        if not tasks:
            raise ValueError("tasks must name at least one BigEarthNet.txt annotation family")

        self.root = Path(root) if root is not None else dataset_dir(self.SOURCE)
        self.split = split
        self.tasks = tuple(tasks)
        self.transform = transform
        logger.debug(
            "BigEarthNetTxtDataset configured: root=%s split=%s tasks=%s",
            self.root,
            self.split,
            self.tasks,
        )

    # -- index ------------------------------------------------------------------------

    def _load_index(self) -> list[dict[str, object]]:
        """Read the annotation index for `split`, filtered to `tasks`.

        The index is the per-split annotation table published alongside the patches;
        each row names a patch identifier, a task family, the instruction text and the
        target. Rows are the unit of iteration, not patches, because one patch carries
        many annotations.

        Raises:
            NotImplementedError: Always, until the parser exists.
        """
        raise NotImplementedError(
            "BigEarthNet.txt annotation index parser is not implemented. Missing: the "
            f"per-split annotation table expected at {self._index_path()} (obtain the corpus "
            "from https://txt.bigearth.net via scripts/download_datasets.py -- nothing is "
            "downloaded automatically), and the row-to-Sample mapping for the task families "
            f"{self.tasks!r}."
        )

    def _index_path(self) -> Path:
        """Path of the annotation table this split reads. Pure, safe to call without data."""
        return self.root / "annotations" / f"{self.split}.jsonl"

    # -- SampleDataset ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of annotation rows in this split after task filtering.

        Raises:
            NotImplementedError: Always, until `_load_index` exists.
        """
        raise NotImplementedError(
            "BigEarthNetTxtDataset.__len__ is not implemented. Missing: "
            f"{self._index_path()} and BigEarthNetTxtDataset._load_index."
        )

    def __getitem__(self, index: int) -> Sample:
        """Return annotation row `index` as a unified `Sample`.

        The intended record carries two `ImageRef`s -- the Sentinel-2 optical patch and
        the co-registered Sentinel-1 SAR patch -- for fusion rows, and a single optical
        `ImageRef` for the caption, VQA and referring rows, each with
        `gsd_m = BIGEARTHNET_TXT_GSD_M` read from the patch transform rather than
        assumed from this constant.

        Raises:
            NotImplementedError: Always, until `_load_index` exists.
        """
        raise NotImplementedError(
            "BigEarthNetTxtDataset.__getitem__ is not implemented. Missing: "
            f"{self._index_path()}, BigEarthNetTxtDataset._load_index, and the reBEN patch "
            "reader that resolves a patch id to its Sentinel-2 and Sentinel-1 GeoTIFF paths "
            "(satquery.io.raster)."
        )
