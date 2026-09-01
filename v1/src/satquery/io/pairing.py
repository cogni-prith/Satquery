"""Geometric and temporal compatibility checks between two rasters.

The router's deterministic gate needs to know whether two inputs are the same place
at two times, the same place through two sensors, or two unrelated scenes. These are
the primitives it asks. `io/validate.py` turns the answers into an `InputConfig`.

The ISRO evaluation imagery is stated to be pre-georeferenced and co-registered, so
these checks should pass trivially on it. They exist because "should" is not "does",
and a silently mismatched pair produces a confident, wrong change map.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from satquery.preprocess.constants import PAIR_MIN_EXTENT_OVERLAP
from satquery.serve.contracts import ImageRef

__all__ = [
    "PairCheck",
    "bounds_of",
    "check_pair",
    "crs_match",
    "extent_overlap",
    "shape_match",
    "timestamp_delta_days",
]

Bounds = tuple[float, float, float, float]


def bounds_of(ref: ImageRef) -> Bounds | None:
    """Return the axis-aligned footprint `(min_x, min_y, max_x, max_y)` in CRS units.

    All four corners are projected through the affine transform rather than assuming
    a north-up raster, so a rotated transform still yields the correct envelope.
    """
    if ref.transform is None or ref.width is None or ref.height is None:
        return None

    a, b, c, d, e, f = ref.transform
    width, height = float(ref.width), float(ref.height)
    corners = ((0.0, 0.0), (width, 0.0), (0.0, height), (width, height))
    xs = [a * col + b * row + c for col, row in corners]
    ys = [d * col + e * row + f for col, row in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def crs_match(first: ImageRef, second: ImageRef) -> bool | None:
    """Whether the two rasters declare the same CRS. None when either is missing one."""
    if first.crs is None or second.crs is None:
        return None
    return first.crs.strip().upper() == second.crs.strip().upper()


def extent_overlap(first: ImageRef, second: ImageRef) -> float | None:
    """Intersection over union of the two footprints, or None if either is unknown.

    Only meaningful when both rasters share a CRS; the caller checks that first.
    """
    first_bounds = bounds_of(first)
    second_bounds = bounds_of(second)
    if first_bounds is None or second_bounds is None:
        return None

    ax0, ay0, ax1, ay1 = first_bounds
    bx0, by0, bx1, by1 = second_bounds

    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return None
    return intersection / union


def shape_match(first: ImageRef, second: ImageRef) -> bool | None:
    """Whether both rasters have identical pixel dimensions. None when either is unknown."""
    if first.shape is None or second.shape is None:
        return None
    return first.shape == second.shape


def timestamp_delta_days(first: ImageRef, second: ImageRef) -> float | None:
    """Absolute acquisition gap in days, or None when either timestamp is missing."""
    if first.timestamp is None or second.timestamp is None:
        return None
    return abs((second.timestamp - first.timestamp).total_seconds()) / 86400.0


class PairCheck(BaseModel):
    """Everything the gate needs to know about a two-raster input."""

    model_config = ConfigDict(extra="forbid")

    crs_match: bool | None = None
    extent_overlap: float | None = None
    shape_match: bool | None = None
    timestamp_delta_days: float | None = None
    same_modality: bool = False
    warnings: list[str] = Field(default_factory=list)

    @property
    def co_registered(self) -> bool:
        """True when the pair is safe to treat as pixel-aligned.

        Requires matching CRS, matching pixel dimensions, and a footprint overlap at
        or above the frozen threshold. Unknown values do not count as satisfied.
        """
        return (
            self.crs_match is True
            and self.shape_match is True
            and self.extent_overlap is not None
            and self.extent_overlap >= PAIR_MIN_EXTENT_OVERLAP
        )


def check_pair(first: ImageRef, second: ImageRef) -> PairCheck:
    """Run every compatibility check on an ordered pair and collect the warnings."""
    warnings: list[str] = []

    same_crs = crs_match(first, second)
    if same_crs is None:
        warnings.append(
            "at least one input has no CRS; geometric comparison is unreliable and the "
            "pair is assumed co-registered on the caller's word"
        )
    elif not same_crs:
        warnings.append(
            f"CRS mismatch: {first.crs} vs {second.crs}. Reproject to a common CRS before "
            "any change or fusion tool is trusted."
        )

    overlap = extent_overlap(first, second) if same_crs else None
    if same_crs and overlap is None:
        warnings.append("footprints could not be compared: a transform or raster size is missing")
    elif overlap is not None and overlap < PAIR_MIN_EXTENT_OVERLAP:
        warnings.append(
            f"footprints overlap by only {overlap:.1%}, below the {PAIR_MIN_EXTENT_OVERLAP:.0%} "
            "threshold; these may not be the same scene"
        )

    same_shape = shape_match(first, second)
    if same_shape is False:
        warnings.append(
            f"pixel dimensions differ: {first.shape} vs {second.shape}; the pair must be "
            "resampled onto a common grid before per-pixel comparison"
        )

    delta = timestamp_delta_days(first, second)
    if delta is None:
        warnings.append(
            "at least one input has no acquisition timestamp; bi-temporal ordering cannot "
            "be verified from metadata"
        )

    return PairCheck(
        crs_match=same_crs,
        extent_overlap=overlap,
        shape_match=same_shape,
        timestamp_delta_days=delta,
        same_modality=first.modality is second.modality,
        warnings=warnings,
    )
