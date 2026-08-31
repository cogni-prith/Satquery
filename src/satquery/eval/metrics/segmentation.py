"""Segmentation metrics for the fusion extraction row.

Intersection over union, aggregated over the whole split rather than averaged per
image. The distinction matters here and is not a stylistic preference: built-up is
absent from 84% of reBEN patches and water from 73%, so a per-image mean is dominated
by patches where the class does not occur, where IoU is either undefined or trivially
perfect. Summing intersections and unions first gives a figure that is stable, is
comparable between runs, and cannot be moved by reshuffling the batch order.

Accuracy is deliberately absent from this module. On these classes an empty prediction
scores about 96% pixel accuracy while extracting nothing, so offering the function at
all would be offering a foot-gun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["class_iou", "mask_iou", "mean_agreement"]


def _read_mask(source: Any) -> np.ndarray:
    """Read a categorical mask from a path or accept an array unchanged."""
    if source is None:
        raise ValueError("mask is None; a missing prediction must be handled by the caller")
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, (str, Path)):
        import rasterio

        with rasterio.open(str(source)) as handle:
            return handle.read(1)
    raise TypeError(f"cannot read a mask from {type(source).__name__}")


def mask_iou(predictions: list[Any], targets: list[Any], class_index: int) -> float:
    """Dataset-level IoU for one class code, over paired categorical masks.

    Args:
        predictions: Predicted masks, as paths or arrays. `None` entries count as a
            failed prediction: an empty mask, which contributes nothing to the
            intersection but the full target area to the union. That is the honest
            treatment -- dropping them would quietly score only the samples that worked.
        targets: Ground-truth masks, same length and order.
        class_index: The pixel value identifying this class.

    Returns:
        Intersection over union in `[0, 1]`, or `nan` when the class appears in neither
        the predictions nor the targets anywhere in the split. `nan` rather than 0.0 or
        1.0 because nothing was measured, and both of those are defensible-looking
        numbers for a quantity that does not exist.

    Raises:
        ValueError: The two lists differ in length.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets differ in length: {len(predictions)} vs {len(targets)}"
        )

    intersection = 0
    union = 0
    for prediction, target in zip(predictions, targets, strict=True):
        truth = _read_mask(target) == class_index
        if prediction is None:
            union += int(truth.sum())
            continue
        predicted = _read_mask(prediction) == class_index
        if predicted.shape != truth.shape:
            raise ValueError(
                f"mask shapes differ: predicted {predicted.shape}, target {truth.shape}"
            )
        intersection += int((predicted & truth).sum())
        union += int((predicted | truth).sum())

    return float(intersection / union) if union else float("nan")


def class_iou(
    predictions: list[Any], targets: list[Any], classes: tuple[str, ...]
) -> dict[str, float]:
    """Return per-class IoU keyed by class name, using the frozen class ordering.

    Class `i` of `classes` is encoded as pixel value `i + 1`, matching
    `data.datasets.bigearthnet_fusion.corine_extraction_mask`; 0 is background.
    """
    return {name: mask_iou(predictions, targets, index + 1) for index, name in enumerate(classes)}


def mean_agreement(confidences: list[float | None]) -> float:
    """Mean of the learned-versus-index agreement reported per sample.

    This is the project's confidence signal, so it is reported as a metric in its own
    right rather than buried in per-sample traces: a run where the two halves stop
    agreeing has lost the property that makes its answers defensible on an unseen
    sensor, and that should be visible in the scores table.

    `None` entries -- samples where no deterministic cross-check was possible, usually
    a missing SWIR band -- are excluded rather than counted as zero. Returns `nan` if
    nothing could be compared.
    """
    observed = [value for value in confidences if value is not None]
    return float(np.mean(observed)) if observed else float("nan")
