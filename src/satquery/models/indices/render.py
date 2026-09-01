"""Render measured masks as images a person can check the answer against.

A number in a table is a claim; a map is evidence. These renderers exist so the area the
system reports can be compared against the pixels it counted, which is the difference
between a demo that asserts and one that shows its work.

Everything here is presentation. No mask is computed in this module -- it only colours
masks that `deterministic.py` already measured, so a picture can never disagree with the
number beside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["render_change_map", "render_mask_overlay", "write_png"]

#: Semantic colours, matched to the frontend's tokens so a class is one colour everywhere.
CLASS_RGB: dict[str, tuple[int, int, int]] = {
    "water": (56, 189, 248),
    "built_up": (251, 191, 36),
    "vegetation": (74, 222, 128),
}
GAINED_RGB = (52, 211, 153)
LOST_RGB = (251, 113, 133)


def write_png(path: Path, rgb: np.ndarray) -> Path:
    """Write an `(H, W, 3)` uint8 array as a PNG."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).save(path)
    return path


def _dim(base: np.ndarray, factor: float = 0.45) -> np.ndarray:
    """Darken the backdrop so an overlay reads without hiding the scene underneath."""
    return (base.astype(np.float32) * factor).astype(np.uint8)


def render_mask_overlay(base_rgb: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Paint class masks over the scene, one flat colour each.

    Flat colour rather than alpha blending: a half-transparent mask over varied terrain
    produces a different apparent colour in every part of the image, and the reader ends
    up judging coverage by brightness. Solid fill on a dimmed backdrop keeps the boundary
    of what was counted unambiguous.
    """
    out = _dim(base_rgb).copy()
    for name, mask in masks.items():
        colour = CLASS_RGB.get(name)
        if colour is None:
            continue
        out[np.asarray(mask, dtype=bool)] = colour
    return out


def render_change_map(base_rgb: np.ndarray, mask_t1: np.ndarray, mask_t2: np.ndarray) -> np.ndarray:
    """Paint what was gained and what was lost between two dates.

    Three states, because "changed" is not one thing: gained (absent at T1, present at
    T2), lost (the reverse), and unchanged-but-present. Collapsing gain and loss into a
    single "changed" colour is how a drained reservoir and a flood come to look identical.
    """
    first = np.asarray(mask_t1, dtype=bool)
    second = np.asarray(mask_t2, dtype=bool)

    out = _dim(base_rgb).copy()
    out[first & second] = (90, 110, 140)  # present at both dates, held steady
    out[~first & second] = GAINED_RGB
    out[first & ~second] = LOST_RGB
    return out
