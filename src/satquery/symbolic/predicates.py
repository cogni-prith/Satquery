"""Spatial relations over component sets, and referring-phrase resolution.

Pure geometry over the dicts `measures.connected_components` returns. No model, no
training, no GPU -- which is why "the largest building" and "the northernmost lake" are
computed rather than guessed.

Image coordinates throughout: row 0 is the top, so north is a SMALLER row index. Getting
that backwards silently inverts every compass answer, which is why it is stated here and
asserted in the tests rather than left to the reader.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "COMPASS_SECTORS",
    "in_sector",
    "largest",
    "nearest_to",
    "northernmost",
    "resolve_reference",
    "smallest",
]

#: Compass sector to its centre bearing in degrees, clockwise from north.
COMPASS_SECTORS: dict[str, float] = {
    "north": 0.0,
    "north-east": 45.0,
    "east": 90.0,
    "south-east": 135.0,
    "south": 180.0,
    "south-west": 225.0,
    "west": 270.0,
    "north-west": 315.0,
}


def largest(components: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The component with the greatest area, or None when there are none."""
    return max(components, key=lambda c: c["area_px"], default=None)


def smallest(components: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The component with the least area, or None when there are none."""
    return min(components, key=lambda c: c["area_px"], default=None)


def northernmost(components: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The component nearest the top of the image.

    Row 0 is the top, so this is the MINIMUM centroid row. Using the maximum would return
    the southernmost and nothing downstream would notice.
    """
    return min(components, key=lambda c: c["centroid_row"], default=None)


def nearest_to(components: list[dict[str, Any]], row: float, col: float) -> dict[str, Any] | None:
    """The component whose centroid is closest to `(row, col)`, Euclidean in pixels."""
    return min(
        components,
        key=lambda c: math.hypot(c["centroid_row"] - row, c["centroid_col"] - col),
        default=None,
    )


def bearing_from_centre(component: dict[str, Any], height: int, width: int) -> float:
    """Compass bearing of a component from the image centre, degrees clockwise from north.

    The row axis is negated because it grows downwards while north is up.
    """
    d_row = component["centroid_row"] - height / 2.0
    d_col = component["centroid_col"] - width / 2.0
    return math.degrees(math.atan2(d_col, -d_row)) % 360.0


def in_sector(
    components: list[dict[str, Any]], sector: str, height: int, width: int
) -> list[dict[str, Any]]:
    """Components lying within a 45-degree compass sector of the image centre.

    Raises:
        ValueError: Unknown sector name. Better than silently returning nothing, which
            reads as "no such region" rather than "you asked for a sector I do not have".
    """
    if sector not in COMPASS_SECTORS:
        raise ValueError(f"unknown sector {sector!r}; expected one of {sorted(COMPASS_SECTORS)}")

    centre = COMPASS_SECTORS[sector]
    chosen = []
    for component in components:
        bearing = bearing_from_centre(component, height, width)
        offset = abs((bearing - centre + 180.0) % 360.0 - 180.0)
        if offset <= 22.5:
            chosen.append(component)
    return chosen


def resolve_reference(
    components: list[dict[str, Any]],
    phrase: str,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    """Map a referring phrase onto components, ranked rather than reduced to one guess.

    Returns a ranked list with scores, not a single answer, and that is deliberate: a
    referring phrase is often genuinely ambiguous ("the building near the river" when
    there are three), and collapsing that to one box discards the ambiguity instead of
    reporting it. The caller -- and the trace -- can then show the runners-up.

    Scoring is additive over the cues present in the phrase, so "the largest building in
    the north" ranks a component that satisfies both above one that satisfies either.

    Returns:
        Components sorted by descending `score`, each a copy with `score` and `matched`
        (the cues that fired) added. Empty when `components` is empty.
    """
    if not components:
        return []

    text = phrase.lower()
    ranked: list[dict[str, Any]] = []

    areas = [c["area_px"] for c in components]
    max_area = max(areas)
    min_area = min(areas)

    for component in components:
        score = 0.0
        matched: list[str] = []

        if any(word in text for word in ("largest", "biggest", "large")):
            score += component["area_px"] / max_area
            matched.append("largest")
        if any(word in text for word in ("smallest", "tiniest", "small")):
            score += min_area / max(component["area_px"], 1.0)
            matched.append("smallest")

        for sector in COMPASS_SECTORS:
            if sector in text or sector.replace("-", " ") in text:
                bearing = bearing_from_centre(component, height, width)
                offset = abs((bearing - COMPASS_SECTORS[sector] + 180.0) % 360.0 - 180.0)
                # Full credit at the sector centre, none at 90 degrees away.
                score += max(0.0, 1.0 - offset / 90.0)
                matched.append(sector)
                break

        if "centre" in text or "center" in text or "middle" in text:
            distance = math.hypot(
                component["centroid_row"] - height / 2.0,
                component["centroid_col"] - width / 2.0,
            )
            score += max(0.0, 1.0 - distance / (math.hypot(height, width) / 2.0))
            matched.append("centre")

        ranked.append({**component, "score": round(score, 4), "matched": matched})

    ranked.sort(key=lambda c: c["score"], reverse=True)
    return ranked
