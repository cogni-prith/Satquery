"""The `AnswerRecord`: the only thing the verbalizer is ever allowed to see.

This module is the hallucination firewall in code form. Perception produces facts; the
verbalizer words them. Nothing crosses that line except an `AnswerRecord`, and every
`Fact` inside it names the tool or computation that produced it.

The rule that makes "evidence-grounded" true rather than aspirational: if a claim in the
final text has no matching provenance entry, that is a bug. `validate_provenance()` is
where that stops being a convention and becomes an assertion.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["AnswerRecord", "AreaDelta", "Fact", "build_record"]


class Fact(BaseModel):
    """One measured quantity, and where it came from.

    `provenance` is not documentation. It is the audit trail the trace is checking, and
    an empty one means a number appeared from nowhere -- which is exactly the failure a
    generative pipeline makes invisible.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, description="Stable identifier, e.g. 'built_up_area_t1'.")
    value: Any = Field(description="The measured value. Numbers stay numbers; never pre-formatted.")
    unit: str | None = Field(default=None, description="SI unit, e.g. 'm2', 'percent', 'count'.")
    provenance: str = Field(
        min_length=1,
        description=(
            "The tool or computation that produced this value, e.g. "
            "'symbolic.measures.class_area_m2' or 'seg.landcover@0.2.0'. Never empty."
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Agreement between independent signals, never a softmax maximum. A model "
            "score here would make the trace lie about what it knows."
        ),
    )

    @field_validator("provenance")
    @classmethod
    def _reject_blank_provenance(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provenance cannot be blank; name the tool or computation")
        return value


class AreaDelta(BaseModel):
    """Change in one class between two dates, with its verdict.

    A model rather than a loose dict because `trend` is the answer. Read from a dict it
    would be `.get("trend", "unchanged")`, which turns a misspelled key into a confident
    "no change" -- the measurement said otherwise and nothing would show it.
    """

    model_config = ConfigDict(extra="forbid")

    area_t1_m2: float
    area_t2_m2: float
    absolute_m2: float
    relative: float
    trend: str = Field(pattern="^(increased|decreased|unchanged)$")


class AnswerRecord(BaseModel):
    """Everything the verbalizer receives, and the boundary of what it may say.

    Deliberately carries no image, no path to an image, and no feature map. A verbalizer
    that could reach pixels could describe them, and then the text would contain claims
    no measurement backs.
    """

    model_config = ConfigDict(extra="forbid")

    intent: str = Field(min_length=1, description="Parsed query intent, e.g. 'change_trend'.")
    facts: list[Fact] = Field(default_factory=list)

    class_proportions: dict[str, float] = Field(
        default_factory=dict, description="Class name to fraction of scene in [0, 1]."
    )
    object_counts: dict[str, int] = Field(default_factory=dict)
    area_deltas: dict[str, AreaDelta] = Field(default_factory=dict)
    change_transitions: dict[str, float] = Field(
        default_factory=dict, description="'from_class->to_class' to area in m2."
    )
    referenced_regions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked candidates for a referring phrase, with scores. Never one guess.",
    )
    agreement_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Class name to learned-versus-index IoU. This is the confidence signal.",
    )

    warnings: list[str] = Field(default_factory=list)

    def validate_provenance(self) -> None:
        """Raise unless every fact names its source.

        Called before the record reaches the verbalizer. Failing loudly here is the point:
        a fact with no provenance is a sentence the system cannot defend, and it is
        cheaper to crash than to ship it.
        """
        blank = [fact.key for fact in self.facts if not fact.provenance.strip()]
        if blank:
            raise ValueError(
                f"AnswerRecord facts with no provenance: {blank}. Every value must name "
                "the tool or computation that produced it."
            )

    def fact(self, key: str) -> Fact | None:
        """Return the fact with `key`, or None."""
        return next((f for f in self.facts if f.key == key), None)


def build_record(
    intent: str,
    perception_outputs: dict[str, Any],
    gsd_m: float | None,
) -> AnswerRecord:
    """Assemble an `AnswerRecord` from perception outputs, provenance filled throughout.

    Args:
        intent: Parsed query intent.
        perception_outputs: Whatever the perception bank produced for this request --
            masks, boxes, index maps -- keyed by the tool that produced each.
        gsd_m: Ground sampling distance, needed to turn pixel counts into areas. `None`
            means areas cannot be computed and the facts that need them are omitted
            rather than guessed.

    Raises:
        NotImplementedError: Always, until the perception bank exists. Assembling a
            record from tools that do not yet run would mean inventing the facts.
    """
    raise NotImplementedError(
        "build_record is not implemented. Missing: the perception bank whose outputs it "
        f"assembles (got keys {sorted(perception_outputs)!r} for intent {intent!r}, "
        f"gsd_m={gsd_m}). Every downstream Fact needs a real tool to name as provenance, "
        "so this cannot be written before the tools it reads from."
    )
