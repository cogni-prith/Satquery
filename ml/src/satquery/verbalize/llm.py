"""Optional LLM rewrite of already-verbalized text. Same firewall, same inputs.

Exists because CIDEr punishes template phrasing, not because the system needs a language
model to know anything. It receives the `AnswerRecord` and the template output, and may
reword. It may not add, infer, or extrapolate a fact.

Two guarantees, and only one of them is a request:

The model call has no image parameter and no tool access, exactly like `templates.py`.
That is structural — an absent parameter cannot be prompted away.

The output is then checked: every number it contains must already appear in the record or
in the draft. A rewrite that invents "approximately 3 square kilometres" from a record
holding 2.7 is a hallucination wearing a hedge, and the check is what catches it. This
guard is implemented and tested now, before the model that needs it exists, because the
guard is the part that makes the rewrite safe to turn on later.
"""

from __future__ import annotations

import re

from satquery.symbolic.record import AnswerRecord

__all__ = ["NumberGuardError", "check_numbers", "extract_numbers", "rewrite"]

#: Any decimal number, with optional sign, thousands separators and decimal part.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

#: Numbers a rewrite may introduce without them appearing in the record.
#:
#: Small integers are ordinary English ("one of the three regions", "a second candidate")
#: rather than measurements, and rejecting them would make every fluent rewrite fail. The
#: ceiling is deliberately low: a measurement that happens to be 7 is rare, and the cost
#: of missing one is far below the cost of a guard nobody can leave enabled.
_FREE_INTEGERS: frozenset[float] = frozenset(float(n) for n in range(0, 11))

#: Fractional tolerance when matching a rewritten number against a known one, so that
#: rounding "1234.56" to "1234.6" or "1235" is allowed but changing it is not.
_TOLERANCE = 0.01


class NumberGuardError(ValueError):
    """A rewrite introduced a number no measurement supports."""


def extract_numbers(text: str) -> list[float]:
    """Every number in `text`, as floats. Thousands separators are stripped."""
    values: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        try:
            values.append(float(match.group().replace(",", "")))
        except ValueError:  # pragma: no cover - the regex cannot produce this
            continue
    return values


def _known_numbers(record: AnswerRecord, draft: str) -> list[float]:
    """Every number the rewrite is allowed to use.

    Drawn from the record's own values AND from the draft, because the draft is itself
    derived from the record and may have converted units -- 2,700,000 m² rendered as
    "2.70 km²". Rejecting the converted form would fail every honest rewrite.
    """
    known: list[float] = list(extract_numbers(draft))

    for fact in record.facts:
        if isinstance(fact.value, (int, float)) and not isinstance(fact.value, bool):
            known.append(float(fact.value))
    for value in record.class_proportions.values():
        known.extend((float(value), float(value) * 100.0))
    known.extend(float(count) for count in record.object_counts.values())
    for delta in record.area_deltas.values():
        known.extend(
            (
                abs(delta.absolute_m2),
                abs(delta.relative),
                abs(delta.relative) * 100.0,
                delta.area_t1_m2,
                delta.area_t2_m2,
            )
        )
    known.extend(abs(area) for area in record.change_transitions.values())
    for score in record.agreement_scores.values():
        known.extend((float(score), float(score) * 100.0))
    for region in record.referenced_regions:
        known.extend(float(v) for v in region.values() if isinstance(v, (int, float)))

    return known


def check_numbers(record: AnswerRecord, draft: str, rewritten: str) -> None:
    """Raise unless every number in `rewritten` is supported by the record or the draft.

    Matching is proportional rather than exact so a rewrite may round -- 1234.56 to
    1234.6 or to 1235 -- but may not change a value. Small integers pass freely because
    they are ordinary English rather than measurements.

    Raises:
        NumberGuardError: naming the unsupported values, so the failure says what the
            model made up rather than only that it did.
    """
    known = _known_numbers(record, draft)
    unsupported: list[float] = []

    for value in extract_numbers(rewritten):
        if value in _FREE_INTEGERS:
            continue
        scale = max(abs(value), 1.0)
        if any(
            abs(value - candidate) <= _TOLERANCE * max(scale, abs(candidate)) for candidate in known
        ):
            continue
        unsupported.append(value)

    if unsupported:
        raise NumberGuardError(
            f"rewrite introduced {unsupported!r}, which no fact in the AnswerRecord "
            "supports. Every number in the answer must trace to a measurement; a "
            "rewrite may reword but never add."
        )


def rewrite(record: AnswerRecord, draft: str) -> str:
    """Reword `draft` more fluently without adding any claim.

    The guard in `check_numbers` is implemented and tested; only the model call is
    missing. On implementation the flow is: call the model with the draft alone, run
    `check_numbers`, and fall back to the draft on a guard failure -- loudly, in the
    trace, never silently.

    Raises:
        NotImplementedError: Always, until the rewrite model exists. Returning `draft`
            unchanged would be worse than failing, because callers would believe a
            rewrite happened and the fallback would be invisible.
    """
    raise NotImplementedError(
        "verbalize.llm.rewrite is not implemented. Missing: the small instruct model that "
        f"rewords the draft ({len(draft.split())} words from {len(record.facts)} fact(s)). "
        "The output guard it needs -- verbalize.llm.check_numbers -- IS implemented and "
        "tested, so wiring a model in is the only remaining step."
    )
