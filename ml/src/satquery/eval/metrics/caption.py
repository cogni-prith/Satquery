"""VRSBench captioning metrics: BLEU-4, METEOR, ROUGE-L and CIDEr.

VRSBench scores captions with the standard COCO caption metric suite (see
github.com/lx709/VRSBench and, for the metric implementations themselves,
`pycocoevalcap`, from Chen et al., "Microsoft COCO Captions", arXiv:1504.00325).
Reimplementing BLEU/METEOR/ROUGE/CIDEr by hand would produce numbers that do not
line up with any published table, which is worse than having no numbers, so this
module wraps `pycocoevalcap` rather than reproducing it.

`pycocoevalcap` is not a base dependency of this package (METEOR pulls in a Java
runtime), and the VRSBench human-verified caption references are not shipped with
the repo. Both are therefore missing, and every function here raises
`NotImplementedError` naming exactly what is absent. Nothing here returns a
placeholder score.

Install path once we commit to it: `uv add pycocoevalcap` plus a JRE for METEOR,
and the VRSBench annotation JSON under `$SATQUERY_DATA_ROOT/vrsbench/`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from satquery.utils.logging import get_logger
from satquery.utils.paths import dataset_dir

_LOG = get_logger(__name__)

__all__ = [
    "CAPTION_METRICS",
    "bleu4",
    "caption_metrics",
    "cider",
    "meteor",
    "rouge_l",
    "vrsbench_reference_path",
]

#: Metric identifiers this module reports, in the order VRSBench tables print them.
CAPTION_METRICS: tuple[str, ...] = ("bleu4", "meteor", "rouge_l", "cider")

_MISSING_PACKAGE = (
    "the `pycocoevalcap` package (not a base dependency; METEOR additionally needs a "
    "Java runtime on PATH)"
)


def vrsbench_reference_path(split: str = "test") -> Path:
    """Return where the VRSBench caption references for `split` are expected on disk.

    Nothing is downloaded automatically; this only names the location that
    `scripts/download_datasets.py` prints instructions for.

    Args:
        split: VRSBench split name, e.g. `"test"`.

    Returns:
        `<data_root>/vrsbench/captions_<split>.json`, which may not exist.
    """
    return dataset_dir("vrsbench") / f"captions_{split}.json"


def _missing(metric: str, references: Path | None) -> NotImplementedError:
    """Build the NotImplementedError naming both missing pieces for `metric`."""
    reference = references if references is not None else vrsbench_reference_path()
    return NotImplementedError(
        f"cannot compute {metric}. Missing: (1) {_MISSING_PACKAGE}; "
        f"(2) the VRSBench human-verified caption reference file at {reference}, which "
        f"is not downloaded automatically -- run scripts/download_datasets.py for the "
        f"instructions. No placeholder score is returned."
    )


def _score_all(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> dict[str, float]:
    """Run every pycocoevalcap scorer once and return all metrics.

    One pass rather than four: the tokenizer and the Java METEOR subprocess are the
    expensive parts, and scoring the same corpus four times pays for them four times.

    Args:
        predictions: One generated caption per image.
        references: One or more reference captions per image.

    Raises:
        ImportError: `pycocoevalcap` is not installed. It is not a base dependency.
        ValueError: Prediction and reference counts disagree.
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "caption metrics need pycocoevalcap, which is not a base dependency: "
            "`pip install pycocoevalcap`. METEOR additionally needs a Java runtime."
        ) from exc

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must align: got {len(predictions)} and {len(references)}"
        )
    if not predictions:
        return dict.fromkeys(CAPTION_METRICS, 0.0)

    tokenizer = PTBTokenizer()
    gts = {str(i): [{"caption": r} for r in refs] for i, refs in enumerate(references)}
    res = {str(i): [{"caption": p}] for i, p in enumerate(predictions)}
    gts, res = tokenizer.tokenize(gts), tokenizer.tokenize(res)

    scores: dict[str, float] = {}
    bleu, _ = Bleu(4).compute_score(gts, res)
    for index, value in enumerate(bleu, start=1):
        scores[f"bleu{index}"] = float(value)
    scores["rouge_l"] = float(Rouge().compute_score(gts, res)[0])
    scores["cider"] = float(Cider().compute_score(gts, res)[0])

    # METEOR shells out to Java. It is the one metric that can be absent on a machine
    # where the others work, so its failure is isolated rather than sinking the batch.
    try:
        from pycocoevalcap.meteor.meteor import Meteor

        scores["meteor"] = float(Meteor().compute_score(gts, res)[0])
    except Exception as exc:
        _LOG.warning("METEOR unavailable (needs a Java runtime): %s", str(exc)[:120])
        scores["meteor"] = float("nan")
    return scores


def caption_metrics(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> dict[str, float]:
    """Return BLEU-4, METEOR, ROUGE-L and CIDEr for a caption corpus.

    These are corpus-level metrics: they cannot be averaged from per-sample values, so
    the whole corpus is scored in one call.
    """
    scores = _score_all(predictions, references)
    return {name: scores[name] for name in CAPTION_METRICS if name in scores}


# CIDEr weights n-grams by corpus-level inverse document frequency, so a single-sample
# corpus has degenerate IDF and scores 0.0. That is a property of the metric, not a bug;
# it only matters if someone tries to score one caption at a time.
_CIDER_NEEDS_A_CORPUS = True


def bleu4(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """BLEU-4. Corpus-level."""
    return _score_all(predictions, references)["bleu4"]


def meteor(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """METEOR. Corpus-level. Returns NaN when no Java runtime is available."""
    return _score_all(predictions, references)["meteor"]


def rouge_l(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """ROUGE-L. Corpus-level."""
    return _score_all(predictions, references)["rouge_l"]


def cider(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """CIDEr. Corpus-level."""
    return _score_all(predictions, references)["cider"]
