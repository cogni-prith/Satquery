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
    # Not a land-cover class: the region a radiometric comparison flagged as different.
    "changed": (255, 62, 78),
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


def _outline(mask: np.ndarray) -> np.ndarray:
    """The one-pixel boundary of a mask, by 4-neighbour erosion. Pure NumPy, no scipy."""
    solid = np.asarray(mask, dtype=bool)
    eroded = solid.copy()
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        eroded &= np.roll(solid, shift, axis=axis)
    # Rolling wraps at the border, so a mask touching the edge would lose its rim there.
    eroded[0, :] = eroded[-1, :] = eroded[:, 0] = eroded[:, -1] = False
    return solid & ~eroded


def render_change_alpha(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
    *,
    gained_rgb: tuple[int, int, int] = (255, 62, 78),
    lost_rgb: tuple[int, int, int] = (255, 176, 32),
    outline_rgb: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Changed pixels on a transparent field, as `(H, W, 4)` RGBA.

    Made for laying over the live imagery rather than sitting in a tile, so everything
    that did not change is genuinely transparent -- a dimmed backdrop baked into the
    overlay would double-darken the scene underneath it.

    Three marks, because a coloured blob alone cannot say what it was measured against.
    Gain and loss get separate colours, and the T1 extent is traced as a one-pixel
    outline. That outline is what makes the overlay self-explanatory: on a reservoir that
    only filled, the changed region *is* most of the final water body, so the highlight
    looks like the lake and a viewer reasonably concludes it is just drawing the water.
    With the old shoreline drawn on top, the same picture reads as "everything outside
    this line is new", which is the claim actually being made.

    The edge is deliberately hard, with no feathering. A soft edge would suggest the
    measurement has uncertainty at the boundary that the pixel count does not model.
    """
    first = np.asarray(mask_t1, dtype=bool)
    second = np.asarray(mask_t2, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(f"masks must share a grid, got {first.shape} and {second.shape}")

    out = np.zeros((*first.shape, 4), dtype=np.uint8)

    gained = ~first & second
    lost = first & ~second
    for region, colour in ((gained, gained_rgb), (lost, lost_rgb)):
        out[region, 0], out[region, 1], out[region, 2] = colour
        out[region, 3] = 255

    rim = _outline(first)
    out[rim, 0], out[rim, 1], out[rim, 2] = outline_rgb
    out[rim, 3] = 255
    return out


def render_mask_alpha(
    mask: np.ndarray, *, rgb: tuple[int, int, int] = (56, 189, 248)
) -> np.ndarray:
    """One class mask on a transparent field, for overlaying on the live imagery."""
    solid = np.asarray(mask, dtype=bool)
    out = np.zeros((*solid.shape, 4), dtype=np.uint8)
    out[solid, 0] = rgb[0]
    out[solid, 1] = rgb[1]
    out[solid, 2] = rgb[2]
    out[solid, 3] = 255
    return out
