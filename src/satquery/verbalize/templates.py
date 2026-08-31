"""Deterministic wording of an `AnswerRecord`. One template family per task.

This module is the far side of the hallucination firewall. Its signature takes an
`AnswerRecord` and nothing else -- no image, no path, no feature map, no tool handle --
and that absence is the whole design, not an oversight. `tests/test_verbalize.py` asserts
the signature stays that way, because the day someone adds an `image` parameter "just for
context" is the day the text can contain claims no measurement backs.

Templates come first and the LLM rewrite is optional. A template cannot hallucinate; it
can only be clumsy. Clumsy is a CIDEr problem, hallucination is a correctness problem.
"""

from __future__ import annotations

from satquery.symbolic.record import AnswerRecord

__all__ = ["verbalize"]


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _area(value_m2: float) -> str:
    """Square metres, or hectares/square kilometres once those read better."""
    if abs(value_m2) >= 1_000_000.0:
        return f"{value_m2 / 1_000_000.0:.2f} km²"
    if abs(value_m2) >= 10_000.0:
        return f"{value_m2 / 10_000.0:.2f} ha"
    return f"{value_m2:.0f} m²"


def _change_trend(record: AnswerRecord) -> str:
    parts: list[str] = []
    for class_name, delta in record.area_deltas.items():
        trend = delta.trend
        readable = class_name.replace("_", " ")
        if trend == "unchanged":
            parts.append(
                f"{readable} is unchanged (change of {_area(delta.absolute_m2)} is "
                "below the reporting threshold)"
            )
        else:
            parts.append(
                f"{readable} {trend} by {_area(abs(delta.absolute_m2))} "
                f"({_percent(abs(delta.relative))} of its earlier extent)"
            )
    return "; ".join(parts) + "." if parts else "No class showed a measurable change."


def _proportions(record: AnswerRecord) -> str:
    if not record.class_proportions:
        return "No land cover proportions were measured."
    ordered = sorted(record.class_proportions.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{name.replace('_', ' ')} {_percent(value)}" for name, value in ordered) + "."


def _counts(record: AnswerRecord) -> str:
    if not record.object_counts:
        return "No objects were counted."
    return (
        ", ".join(
            f"{count} {name.replace('_', ' ')}{'' if count == 1 else 's'}"
            for name, count in record.object_counts.items()
        )
        + "."
    )


def _regions(record: AnswerRecord) -> str:
    if not record.referenced_regions:
        return "No region matched the phrase."
    best = record.referenced_regions[0]
    text = (
        f"The best match is at row {best['centroid_row']:.0f}, column "
        f"{best['centroid_col']:.0f}, covering {best['area_px']:.0f} pixels"
    )
    if len(record.referenced_regions) > 1:
        # Ambiguity is reported, never resolved silently. The runners-up existing is
        # itself information the reader needs.
        text += f", with {len(record.referenced_regions) - 1} other candidate(s) considered"
    return text + "."


_FAMILIES = {
    "change_trend": _change_trend,
    "land_cover": _proportions,
    "count": _counts,
    "reference": _regions,
}


def verbalize(record: AnswerRecord) -> str:
    """Word an `AnswerRecord` as English. Deterministic, and record-only by construction.

    Every sentence is built from a field of the record, so every claim traces to a `Fact`
    with provenance. Confidence is appended when the record carries agreement scores,
    phrased as a band rather than a number, because the number only means anything as a
    comparison between two independent signals.

    Raises:
        ValueError: The record's facts are not all provenanced. Refusing here is the
            firewall doing its job -- a fact from nowhere must never reach prose.
    """
    record.validate_provenance()

    body = _FAMILIES.get(record.intent)
    text = body(record) if body else _proportions(record)

    if record.agreement_scores:
        from satquery.symbolic.thresholds import confidence_band

        bands = {name: confidence_band(score) for name, score in record.agreement_scores.items()}
        worst = (
            "low"
            if "low" in bands.values()
            else ("medium" if "medium" in bands.values() else "high")
        )
        text += (
            f" Confidence is {worst}, measured as agreement between two independent "
            "estimates of the same quantity."
        )

    if record.warnings:
        text += f" ({len(record.warnings)} warning(s) recorded.)"

    return text
