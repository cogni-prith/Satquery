"""Measurements over masks and boxes. The layer where answers are decided.

Pure NumPy, no torch, no GPU, no training dependency. "Has the built-up area increased?"
has an exact answer: count pixels at each date, multiply by GSD squared, compare. A model
guessing that is a defect, so this module computes it instead.

Everything here is deterministic and testable against a synthetic mask, which is why it
is implemented properly on first pass rather than stubbed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "agreement_iou",
    "area_delta",
    "change_matrix",
    "class_area_m2",
    "class_fraction",
    "connected_components",
    "object_count",
]


def class_fraction(mask: np.ndarray, class_id: int) -> float:
    """Fraction of the scene covered by `class_id`, in [0, 1]."""
    array = np.asarray(mask)
    if array.size == 0:
        raise ValueError("mask is empty")
    return float((array == class_id).sum() / array.size)


def class_area_m2(mask: np.ndarray, class_id: int, gsd_m: float) -> float:
    """Ground area of `class_id` in square metres.

    Args:
        mask: 2D array of class ids.
        class_id: The class to measure.
        gsd_m: Ground sampling distance in metres. One pixel is `gsd_m ** 2` of ground.

    Raises:
        ValueError: `gsd_m` is not positive. An unknown GSD makes area meaningless, and
            substituting a default would put a fabricated number into an answer.
    """
    if gsd_m <= 0.0:
        raise ValueError(
            f"gsd_m must be positive to convert pixels to area, got {gsd_m}. An unknown "
            "GSD means the area is unknown; do not substitute a default."
        )
    array = np.asarray(mask)
    return float((array == class_id).sum()) * float(gsd_m) ** 2


def object_count(boxes: list[dict], label: str | None = None) -> int:
    """Number of detections, optionally filtered to one label."""
    if label is None:
        return len(boxes)
    return sum(1 for box in boxes if box.get("label") == label)


def connected_components(
    mask: np.ndarray, class_id: int, min_area_px: int = 1
) -> list[dict[str, float]]:
    """Split one class into discrete instances.

    Four-connectivity, flood-filled iteratively rather than recursively so a large blob
    cannot exhaust the Python stack.

    Args:
        mask: 2D array of class ids.
        class_id: Class to segment into components.
        min_area_px: Components smaller than this are dropped as speckle.

    Returns:
        One dict per component with `area_px`, `centroid_row`, `centroid_col`, and the
        bounding box as `row_min`/`row_max`/`col_min`/`col_max`.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {array.shape}")

    target = array == class_id
    seen = np.zeros_like(target, dtype=bool)
    components: list[dict[str, float]] = []

    for start in zip(*np.nonzero(target), strict=True):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        pixels: list[tuple[int, int]] = []

        while stack:
            row, col = stack.pop()
            pixels.append((row, col))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r, c = row + dr, col + dc
                in_bounds = 0 <= r < target.shape[0] and 0 <= c < target.shape[1]
                if in_bounds and target[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    stack.append((r, c))

        if len(pixels) < min_area_px:
            continue

        rows = np.array([p[0] for p in pixels])
        cols = np.array([p[1] for p in pixels])
        components.append(
            {
                "area_px": float(len(pixels)),
                "centroid_row": float(rows.mean()),
                "centroid_col": float(cols.mean()),
                "row_min": float(rows.min()),
                "row_max": float(rows.max()),
                "col_min": float(cols.min()),
                "col_max": float(cols.max()),
            }
        )

    components.sort(key=lambda c: c["area_px"], reverse=True)
    return components


def area_delta(
    mask_t1: np.ndarray, mask_t2: np.ndarray, class_id: int, gsd_m: float
) -> dict[str, float]:
    """Absolute and relative change in one class between two dates.

    Returns:
        `absolute_m2` (signed), `relative` (signed fraction of the T1 area), and the two
        endpoint areas. `relative` is 0.0 when the class was absent at T1 -- growth from
        nothing has no finite ratio, and `inf` would propagate into a threshold comparison
        as a spurious "increased".

    Raises:
        ValueError: The two masks differ in shape, which means they are not co-registered
            and no per-pixel comparison between them is valid.
    """
    first, second = np.asarray(mask_t1), np.asarray(mask_t2)
    if first.shape != second.shape:
        raise ValueError(
            f"masks must be co-registered onto one grid, got {first.shape} and {second.shape}"
        )

    area_t1 = class_area_m2(first, class_id, gsd_m)
    area_t2 = class_area_m2(second, class_id, gsd_m)
    absolute = area_t2 - area_t1
    relative = (absolute / area_t1) if area_t1 > 0.0 else 0.0

    return {
        "area_t1_m2": area_t1,
        "area_t2_m2": area_t2,
        "absolute_m2": absolute,
        "relative": relative,
    }


def change_matrix(
    mask_t1: np.ndarray, mask_t2: np.ndarray, gsd_m: float | None = None
) -> dict[tuple[int, int], float]:
    """From-class to-class transitions between two dates.

    This is what SECOND provides and what the symbolic layer needs: not "something
    changed" but "this many square metres went from vegetation to built-up".

    Returns:
        `(from_class, to_class)` to pixel count, or to square metres when `gsd_m` is
        given. Unchanged pixels are included as `(c, c)`; the caller decides whether to
        report them.
    """
    first, second = np.asarray(mask_t1), np.asarray(mask_t2)
    if first.shape != second.shape:
        raise ValueError(
            f"masks must be co-registered onto one grid, got {first.shape} and {second.shape}"
        )

    scale = float(gsd_m) ** 2 if gsd_m else 1.0
    pairs, counts = np.unique(
        np.stack([first.ravel(), second.ravel()], axis=1), axis=0, return_counts=True
    )
    return {
        (int(a), int(b)): float(count) * scale for (a, b), count in zip(pairs, counts, strict=True)
    }


def agreement_iou(learned_mask: np.ndarray, index_mask: np.ndarray) -> float:
    """Intersection over union between a learned mask and a deterministic index mask.

    The project's confidence signal. Two independent estimates of the same quantity: one
    learned, one closed-form arithmetic with no training distribution to fall outside of.
    Where they agree the answer is defensible on a sensor the model has never seen; where
    they disagree, that disagreement is the honest thing to report.

    Returns:
        IoU in [0, 1], or `nan` when neither mask has a positive pixel -- nothing was
        measured, and 0.0 or 1.0 would both be fabricated.
    """
    first = np.asarray(learned_mask).astype(bool)
    second = np.asarray(index_mask).astype(bool)
    if first.shape != second.shape:
        raise ValueError(f"masks differ in shape: {first.shape} and {second.shape}")

    union = int((first | second).sum())
    if union == 0:
        return float("nan")
    return float((first & second).sum() / union)
