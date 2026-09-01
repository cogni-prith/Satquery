"""Closed-set VQA metrics: exact match after normalisation, overall and per question type.

RSVQA and CDVQA are both scored by exact match. Per-type accuracy matters as much as
the overall number: a model can look respectable overall while being at chance on
counting questions, and the per-type breakdown is what tells the team where to spend
the next training run.
"""

from __future__ import annotations

from collections.abc import Sequence

from satquery.eval.answer_norm import normalize_answer, normalize_for_dataset

__all__ = ["average_per_type_accuracy", "exact_match_accuracy", "per_type_accuracy"]


def exact_match_accuracy(
    predictions: Sequence[str],
    targets: Sequence[str],
    dataset: str = "",
) -> float:
    """Fraction of predictions that equal their reference after identical normalisation.

    Args:
        predictions: Model answers.
        targets: Reference answers.
        dataset: Benchmark key passed to `normalize_for_dataset`. Empty means the
            generic normaliser.

    Raises:
        ValueError: Sequences of differing length, which would misalign the comparison.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must align: got {len(predictions)} and {len(targets)}"
        )
    if not predictions:
        return 0.0

    normalise = (lambda text: normalize_for_dataset(text, dataset)) if dataset else normalize_answer
    hits = sum(
        1
        for prediction, reference in zip(predictions, targets, strict=True)
        if normalise(prediction) == normalise(reference)
    )
    return hits / len(predictions)


def per_type_accuracy(
    predictions: Sequence[str],
    targets: Sequence[str],
    types: Sequence[str],
    dataset: str = "",
) -> dict[str, float]:
    """Exact-match accuracy split by question type, e.g. presence, count, comparison.

    Returns:
        Accuracy per type, sorted by type name for a stable report ordering.
    """
    if not (len(predictions) == len(targets) == len(types)):
        raise ValueError(
            f"predictions, targets and types must align: got {len(predictions)}, "
            f"{len(targets)} and {len(types)}"
        )

    buckets: dict[str, list[tuple[str, str]]] = {}
    for prediction, reference, question_type in zip(predictions, targets, types, strict=True):
        buckets.setdefault(question_type, []).append((prediction, reference))

    return {
        question_type: exact_match_accuracy(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs], dataset
        )
        for question_type, pairs in sorted(buckets.items())
    }


def average_per_type_accuracy(per_type: dict[str, float]) -> float:
    """Unweighted mean over question types, as RSVQA reports alongside overall accuracy.

    Unweighted because RSVQA's type distribution is heavily skewed; the plain overall
    number is dominated by whichever type happens to be most numerous.
    """
    if not per_type:
        return 0.0
    return sum(per_type.values()) / len(per_type)
