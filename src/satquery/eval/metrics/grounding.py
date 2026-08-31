"""Grounding metrics: box IoU and acc@tau.

VRSBench scores referring-expression grounding as accuracy at an IoU threshold on
horizontal boxes, conventionally tau = 0.5. Pure geometry, no GPU, no model.
"""

from __future__ import annotations

from collections.abc import Sequence

from satquery.serve.contracts import BoundingBox

__all__ = ["acc_at_tau", "best_match", "iou", "mean_iou"]


def iou(first: BoundingBox, second: BoundingBox) -> float:
    """Intersection over union of two axis-aligned boxes, in `[0, 1]`."""
    inter_w = max(0.0, min(first.x_max, second.x_max) - max(first.x_min, second.x_min))
    inter_h = max(0.0, min(first.y_max, second.y_max) - max(first.y_min, second.y_min))
    intersection = inter_w * inter_h
    union = first.area + second.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def best_match(prediction: BoundingBox, targets: Sequence[BoundingBox]) -> float:
    """Highest IoU between a prediction and any of the reference boxes.

    A referring expression can legitimately have several acceptable referents, so the
    prediction is scored against its best match rather than against a fixed index.
    """
    if not targets:
        return 0.0
    return max(iou(prediction, target) for target in targets)


def acc_at_tau(
    predictions: Sequence[BoundingBox | None],
    targets: Sequence[Sequence[BoundingBox]],
    tau: float = 0.5,
) -> float:
    """Fraction of samples whose predicted box reaches IoU >= `tau` with a reference.

    Args:
        predictions: One box per sample, or None where the model produced nothing.
            A None prediction counts as a miss, never as a skip -- dropping it would
            let a model inflate its score by declining to answer.
        targets: Reference boxes per sample.
        tau: IoU threshold. VRSBench reports 0.5.

    Raises:
        ValueError: The two sequences differ in length, which would silently misalign
            every sample after the first mismatch.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must align: got {len(predictions)} and {len(targets)}"
        )
    if not predictions:
        return 0.0
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")

    hits = sum(
        1
        for prediction, reference in zip(predictions, targets, strict=True)
        if prediction is not None and best_match(prediction, reference) >= tau
    )
    return hits / len(predictions)


def mean_iou(
    predictions: Sequence[BoundingBox | None],
    targets: Sequence[Sequence[BoundingBox]],
) -> float:
    """Mean best-match IoU across samples. A None prediction contributes 0.0."""
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must align: got {len(predictions)} and {len(targets)}"
        )
    if not predictions:
        return 0.0
    total = sum(
        0.0 if prediction is None else best_match(prediction, reference)
        for prediction, reference in zip(predictions, targets, strict=True)
    )
    return total / len(predictions)
