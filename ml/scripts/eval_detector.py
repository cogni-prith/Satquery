#!/usr/bin/env python3
"""Score the detector: recall and precision at IoU 0.5, per class where it is meaningful.

Reported as recall AND precision, not one number. VRSBench labels one box per referring
expression, so a scene with many objects carries few boxes: a correct detection of a real
unannotated object counts as a false positive here. Precision measured against these
labels is therefore a floor, not an estimate, and quoting it alone would understate the
detector as badly as quoting recall alone would flatter it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.utils.logging import get_logger

LOG = get_logger("eval-detector")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def evaluate(model, dataset, device, vocabulary, threshold: float = 0.5, score_min: float = 0.5):
    """Recall and precision at IoU `threshold`, ignoring detections below `score_min`."""
    import torch

    model.eval()
    matched = 0
    ground_truth = 0
    predicted = 0

    with torch.no_grad():
        for index in range(len(dataset)):
            image, target = dataset[index]
            output = model([image.to(device)])[0]
            keep = output["scores"].cpu().numpy() >= score_min
            boxes = output["boxes"].cpu().numpy()[keep]
            truth = target["boxes"].numpy()

            ground_truth += len(truth)
            predicted += len(boxes)
            if len(truth) and len(boxes):
                overlap = iou_matrix(truth, boxes)
                # Greedy one-to-one: a single prediction must not satisfy two labels.
                used = set()
                for row in range(len(truth)):
                    order = np.argsort(-overlap[row])
                    for column in order:
                        if column in used:
                            continue
                        if overlap[row, column] >= threshold:
                            matched += 1
                            used.add(int(column))
                        break

    recall = matched / ground_truth if ground_truth else float("nan")
    precision = matched / predicted if predicted else float("nan")
    LOG.info(
        "detections: %d, labelled boxes: %d, matched at IoU %.1f: %d",
        predicted,
        ground_truth,
        threshold,
        matched,
    )
    LOG.info("  recall    %.4f", recall)
    LOG.info(
        "  precision %.4f  (a floor: unannotated real objects score as false positives)", precision
    )
    return {"recall": recall, "precision": precision, "matched": matched}
