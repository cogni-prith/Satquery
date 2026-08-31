"""RSVQA loader: closed-set visual question answering, evaluation only.

RSVQA (Lobry et al., "RSVQA: Visual Question Answering for Remote Sensing Data",
IEEE TGRS 2020) asks questions whose answers come from a small fixed vocabulary --
presence yes/no, comparisons, counts bucketed into ranges, rural/urban. It is used
here as an evaluation set for the mandatory single-image VQA row, not as training
data, so it is deliberately absent from the default training mix.

**Answer normalisation is the whole game on this benchmark.** A model that answers
"yes, there is a building" is correct and scores zero under naive exact match, and
count answers must be bucketed the same way the ground truth was. Normalisation is
owned by `satquery.eval.answer_norm` and lives nowhere else -- this loader emits the
raw upstream answer string verbatim and does not pre-normalise, so that the training
target and the scored prediction pass through exactly one normaliser.

Canonical source: https://rsvqa.sylvainlobry.com
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from satquery.data.datasets import SampleDataset, SampleTransform
from satquery.data.schema import AnswerType, Sample
from satquery.preprocess.constants import INSTRUCTION_VQA_TEMPLATE
from satquery.serve.contracts import ImageRef, Modality, TaskType
from satquery.utils.logging import get_logger
from satquery.utils.paths import dataset_dir

__all__ = ["RSVQA_QUESTION_TYPES", "RSVQA_SPLITS", "RSVQA_SUBSETS", "RSVQADataset"]

#: Published sub-corpora. `lr` is Sentinel-2 at 10 m, `hr` is USGS aerial at 0.15 m --
#: between them they straddle the train/test resolution gap, so both are worth scoring.
RSVQA_SUBSETS: tuple[str, ...] = ("lr", "hr")

#: Upstream split names. `hr` additionally publishes a second, harder test split.
RSVQA_SPLITS: tuple[str, ...] = ("train", "val", "test", "test_2")

#: Upstream question types. Kept for per-type eval slicing via `Sample.metadata`.
RSVQA_QUESTION_TYPES: tuple[str, ...] = ("presence", "comparison", "count", "rural_urban", "area")

logger = get_logger(__name__)


class RSVQADataset(SampleDataset):
    """Map-style view over one RSVQA sub-corpus and split.

    Single responsibility: turn one RSVQA question record into one unified `Sample`
    with the raw, un-normalised answer string. Normalisation, bucketing and scoring
    belong to `satquery.eval.answer_norm` and the VQA metric, never to this loader.
    """

    #: Value written to `Sample.source`. Never branched on downstream.
    SOURCE = "rsvqa"

    def __init__(
        self,
        root: Path | None = None,
        split: str = "test",
        subset: str = "lr",
        transform: SampleTransform | None = None,
    ) -> None:
        """Configure the view without touching disk.

        Args:
            root: Corpus directory. Defaults to `dataset_dir("rsvqa")`.
            split: One of `RSVQA_SPLITS`.
            subset: One of `RSVQA_SUBSETS`.
            transform: Optional `Sample`-to-`Sample` rewrite applied last.

        Raises:
            ValueError: If `split` or `subset` is not an upstream name.
        """
        if split not in RSVQA_SPLITS:
            raise ValueError(f"unknown RSVQA split {split!r}; expected one of {RSVQA_SPLITS!r}")
        if subset not in RSVQA_SUBSETS:
            raise ValueError(f"unknown RSVQA subset {subset!r}; expected one of {RSVQA_SUBSETS!r}")

        self.root = Path(root) if root is not None else dataset_dir(self.SOURCE)
        self.split = split
        self.subset = subset
        self.transform = transform
        self._index: list[dict[str, Any]] | None = None
        logger.debug(
            "RSVQADataset configured: root=%s split=%s subset=%s",
            self.root,
            self.split,
            self.subset,
        )

    # -- index ------------------------------------------------------------------------

    def _questions_path(self) -> Path:
        """Path of the questions file for this sub-corpus and split. Pure."""
        return self.root / self.subset / f"{self.split}_questions.json"

    def _answers_path(self) -> Path:
        """Path of the answers file for this sub-corpus and split. Pure."""
        return self.root / self.subset / f"{self.split}_answers.json"

    #: HuggingFace mirrors holding the 2k validation subsets actually on disk here.
    _SUBSET_DIR: ClassVar[dict[str, str]] = {"lr": "RSVQA-LR-2k", "hr": "RSVQA-HR-2k"}

    def _subset_root(self) -> Path:
        """Directory holding this subset's parquet shards."""
        return self.root / self._SUBSET_DIR[self.subset]

    def _image_cache(self) -> Path:
        """Where embedded image bytes are materialised.

        The parquet embeds each image as raw bytes with no filename, but `ImageRef`
        requires a path -- deliberately, because every downstream tool, overlay and
        trace refers to imagery by path. Rather than weaken that contract, the bytes are
        written once to a cache directory and referenced from there.
        """
        return self._subset_root() / "_images"

    def _load_index(self) -> list[dict[str, Any]]:
        """Read the parquet shards, materialising images on first use."""
        if self._index is not None:
            return self._index

        root = self._subset_root()
        shards = sorted(root.glob("data/*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"RSVQA parquet shards not found under {root / 'data'}. Nothing is "
                "downloaded automatically; see scripts/download_datasets.py. These 2k "
                "subsets come from the dmarsili/RSVQA-*-2k HuggingFace mirrors."
            )

        import pyarrow.parquet as pq

        cache = self._image_cache()
        cache.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        position = 0
        for shard in shards:
            table = pq.read_table(shard).to_pylist()
            for row in table:
                path = cache / f"{position:06d}.png"
                if not path.exists():
                    payload = row["image"]
                    data = payload["bytes"] if isinstance(payload, dict) else payload
                    path.write_bytes(data)
                records.append(
                    {
                        "path": path,
                        "question": str(row["question"]).strip(),
                        "answer": str(row["answer"]).strip(),
                    }
                )
                position += 1

        logger.info(
            "RSVQA %s/%s: %d records from %d shard(s)",
            self.subset,
            self.split,
            len(records),
            len(shards),
        )
        self._index = records
        return records

    def __len__(self) -> int:
        """Number of question-answer records in this subset."""
        return len(self._load_index())

    def __getitem__(self, index: int) -> Sample:
        """Return record `index` as a single-image closed-set VQA `Sample`.

        RSVQA is closed-set and scored by exact match, so `answer_type` is CLOSED_SET,
        not OPEN_TEXT -- that choice selects the metric, and getting it wrong would score
        this benchmark with an LLM judge it was never designed for.

        These mirrors carry no question-type column, so `question_type` is left empty and
        per-type accuracy is unavailable for them. The full RSVQA release does have it.
        """
        record = self._load_index()[index]
        with Image.open(record["path"]) as handle:
            width, height = handle.size

        sample = Sample(
            sample_id=f"{self.SOURCE}-{self.subset}-{self.split}-{index}",
            task=TaskType.VQA,
            answer_type=AnswerType.CLOSED_SET,
            images=[
                ImageRef(
                    path=record["path"],
                    modality=Modality.OPTICAL_RGB,
                    gsd_m=None,
                    width=width,
                    height=height,
                )
            ],
            instruction=INSTRUCTION_VQA_TEMPLATE.format(question=record["question"].rstrip("?.")),
            answer_text=record["answer"],
            source=self.SOURCE,
            split=self.split,
            metadata={"subset": self.subset, "question_type": "", "question": record["question"]},
        )
        return self.transform(sample) if self.transform else sample
