"""Turning measured deltas into verdicts.

Both floors live in `preprocess/constants.py`, never here and never inline in a caller.
A change threshold that differs between training and inference is exactly the silent
drift the frozen constants exist to prevent, and "increased" is a claim a judge can check.
"""

from __future__ import annotations

from satquery.preprocess.constants import (
    CHANGE_ABSOLUTE_FLOOR_M2,
    CHANGE_RELATIVE_FLOOR,
    CONFIDENCE_BANDS,
)

__all__ = ["classify_trend", "confidence_band"]


def classify_trend_relative(delta_rel: float) -> str:
    """Trend from the relative change alone, for imagery with no known GSD.

    Only the relative floor applies: the absolute floor is expressed in square metres and
    there is no scale to compare against. That makes this the weaker test -- a large
    fractional change over a handful of pixels passes it, where the two-floor rule would
    have called it noise -- so the caller must warn that the area floor was not applied.

    It is still far better than the alternative. Dropping the comparison entirely, which
    an earlier version did, reported "no measurable change" on a scene whose water had
    quadrupled: an unconvertible unit was turned into a false statement about the world.
    """
    if abs(delta_rel) < CHANGE_RELATIVE_FLOOR:
        return "unchanged"
    return "increased" if delta_rel > 0 else "decreased"


def classify_trend(delta_abs: float, delta_rel: float) -> str:
    """Return 'increased', 'decreased' or 'unchanged'.

    BOTH floors must be exceeded. A large relative change on a tiny area is noise, and a
    large absolute change on a huge area may be within measurement error -- requiring
    both is what stops the system reporting a trend it cannot defend.
    """
    if abs(delta_abs) < CHANGE_ABSOLUTE_FLOOR_M2 or abs(delta_rel) < CHANGE_RELATIVE_FLOOR:
        return "unchanged"
    return "increased" if delta_abs > 0.0 else "decreased"


def confidence_band(iou: float) -> str:
    """Map a learned-versus-index agreement IoU onto 'high', 'medium' or 'low'.

    `nan` -- nothing measured -- reports 'low' rather than being dropped: an unmeasurable
    agreement is exactly the case a reader most needs flagged.
    """
    if iou != iou:  # NaN
        return "low"
    for name, floor in CONFIDENCE_BANDS:
        if iou >= floor:
            return name
    return "low"
