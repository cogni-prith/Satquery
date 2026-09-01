"""The hallucination firewall, asserted rather than documented."""

from __future__ import annotations

import inspect

import pytest

from satquery.symbolic.record import AnswerRecord, AreaDelta, Fact
from satquery.verbalize import templates


def test_verbalizer_cannot_see_an_image() -> None:
    """The firewall is the signature, not a convention.

    A verbalizer that could reach pixels could describe them, and the text would then
    contain claims no measurement backs. This test fails the day someone adds an `image`
    parameter "just for context".
    """
    params = set(inspect.signature(templates.verbalize).parameters)
    assert params == {"record"}
    forbidden = {"image", "images", "path", "array", "features", "tool", "tools", "model"}
    assert not (params & forbidden)


def test_a_fact_without_provenance_never_reaches_prose() -> None:
    record = AnswerRecord(intent="land_cover", facts=[Fact(key="x", value=1, provenance="t")])
    record.facts[0].provenance = "   "
    with pytest.raises(ValueError, match="no provenance"):
        templates.verbalize(record)


def test_change_trend_reports_the_threshold_when_unchanged() -> None:
    """'Unchanged' must say why, or a reader assumes nothing was measured."""
    record = AnswerRecord(
        intent="change_trend",
        facts=[Fact(key="d", value=500.0, unit="m2", provenance="symbolic.measures.area_delta")],
        area_deltas={
            "built_up": AreaDelta(
                area_t1_m2=120000.0,
                area_t2_m2=120500.0,
                absolute_m2=500.0,
                relative=0.004,
                trend="unchanged",
            )
        },
    )
    text = templates.verbalize(record)
    assert "unchanged" in text
    assert "threshold" in text


def test_confidence_is_worded_as_agreement_not_a_score() -> None:
    record = AnswerRecord(
        intent="land_cover",
        facts=[Fact(key="w", value=0.2, provenance="seg.landcover")],
        class_proportions={"water": 0.2},
        agreement_scores={"water": 0.9},
    )
    text = templates.verbalize(record)
    assert "Confidence is high" in text
    assert "agreement" in text


def test_ambiguity_is_reported_not_resolved() -> None:
    """Collapsing three candidates to one box discards the ambiguity instead of showing it."""
    record = AnswerRecord(
        intent="reference",
        facts=[Fact(key="r", value=3, provenance="symbolic.predicates.resolve_reference")],
        referenced_regions=[
            {"centroid_row": 10.0, "centroid_col": 20.0, "area_px": 100.0, "score": 0.9},
            {"centroid_row": 40.0, "centroid_col": 50.0, "area_px": 80.0, "score": 0.5},
        ],
    )
    assert "1 other candidate" in templates.verbalize(record)


# -- the rewrite guard ---------------------------------------------------------------------


def _record_with_area() -> AnswerRecord:
    return AnswerRecord(
        intent="change_trend",
        facts=[Fact(key="d", value=2_700_000.0, unit="m2", provenance="symbolic.measures")],
        area_deltas={
            "built_up": AreaDelta(
                area_t1_m2=10_000_000.0,
                area_t2_m2=12_700_000.0,
                absolute_m2=2_700_000.0,
                relative=0.27,
                trend="increased",
            )
        },
    )


def test_guard_allows_a_faithful_rewording() -> None:
    from satquery.verbalize.llm import check_numbers

    record = _record_with_area()
    draft = templates.verbalize(record)
    check_numbers(record, draft, "Built-up land grew by 2.70 km², or 27.0% of its former extent.")


def test_guard_allows_rounding_but_not_changing() -> None:
    from satquery.verbalize.llm import NumberGuardError, check_numbers

    record = _record_with_area()
    draft = templates.verbalize(record)

    check_numbers(record, draft, "Growth of 2.7 km².")  # rounded, fine
    with pytest.raises(NumberGuardError):
        check_numbers(record, draft, "Growth of 3.9 km².")  # changed, not fine


def test_guard_catches_an_invented_number() -> None:
    """A rewrite that invents a figure is a hallucination wearing a hedge."""
    from satquery.verbalize.llm import NumberGuardError, check_numbers

    record = _record_with_area()
    draft = templates.verbalize(record)
    with pytest.raises(NumberGuardError, match="no fact"):
        check_numbers(record, draft, "Built-up land grew across roughly 45 hectares.")


def test_guard_lets_small_integers_through() -> None:
    """'one of the three regions' is English, not a measurement. Rejecting it would make
    every fluent rewrite fail and the guard would be turned off."""
    from satquery.verbalize.llm import check_numbers

    record = _record_with_area()
    check_numbers(record, templates.verbalize(record), "There are 2 findings of note.")


def test_rewrite_refuses_rather_than_silently_returning_the_draft() -> None:
    from satquery.verbalize.llm import rewrite

    record = _record_with_area()
    with pytest.raises(NotImplementedError, match="check_numbers"):
        rewrite(record, templates.verbalize(record))
