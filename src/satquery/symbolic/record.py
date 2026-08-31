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

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from satquery.preprocess.constants import CHANGE_ABSOLUTE_FLOOR_M2

__all__ = ["AnswerRecord", "AreaDelta", "Fact", "build_record"]


#: Every key `build_record` understands. Anything else is refused rather than ignored.
_KNOWN_PERCEPTION_KEYS: frozenset[str] = frozenset(
    {"masks", "masks_t1", "masks_t2", "boxes", "components", "agreement", "regions", "warnings"}
)


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

    This is the seam. Every specialist drops its output into `perception_outputs` under
    the name of the tool that produced it, and this function turns those arrays into
    measured facts. Nothing here decides anything about an image; it counts pixels that
    a tool already labelled, and names that tool on every number it emits.

    Recognised keys, all optional:

    - `masks`: `{class_name: (H, W) bool array}` for a single date.
    - `masks_t1` / `masks_t2`: the same, for two dates. Both must be present for deltas.
    - `boxes`: list of detections, each with a `label`.
    - `components`: `{class_name: [component dicts]}` from `measures.connected_components`.
    - `agreement`: `{class_name: iou}`, learned-vs-index, already computed by the caller.
    - `regions`: ranked referring-expression candidates from `predicates.resolve_reference`.
    - `warnings`: strings to carry through to the answer.

    Args:
        intent: Parsed query intent.
        perception_outputs: Whatever the perception bank produced for this request --
            masks, boxes, index maps -- keyed by the tool that produced each.
        gsd_m: Ground sampling distance, needed to turn pixel counts into areas. `None`
            means areas cannot be computed, so area facts are omitted and a warning says
            so. Substituting a default would put a fabricated number into an answer.

    Returns:
        A record whose every fact names the tool that produced it.

    Raises:
        ValueError: An unknown key was passed. Silently ignoring it would drop a
            specialist's entire output while the answer still looked complete.
    """
    from satquery.symbolic import measures
    from satquery.symbolic.thresholds import classify_trend

    unknown = set(perception_outputs) - _KNOWN_PERCEPTION_KEYS
    if unknown:
        raise ValueError(
            f"unknown perception output key(s) {sorted(unknown)}; expected a subset of "
            f"{sorted(_KNOWN_PERCEPTION_KEYS)}. An unrecognised key means a specialist's "
            "output would be dropped while the answer still looked complete."
        )

    facts: list[Fact] = []
    warnings: list[str] = list(perception_outputs.get("warnings", []))
    class_proportions: dict[str, float] = {}
    object_counts: dict[str, int] = {}
    area_deltas: dict[str, AreaDelta] = {}
    change_transitions: dict[str, float] = {}
    agreement_scores: dict[str, float] = {
        name: float(value) for name, value in perception_outputs.get("agreement", {}).items()
    }

    have_gsd = gsd_m is not None and gsd_m > 0.0
    if not have_gsd and ("masks" in perception_outputs or "masks_t1" in perception_outputs):
        warnings.append(
            "no ground sampling distance could be read from the image, so coverage is "
            "reported as a fraction of the scene and no ground area is given"
        )

    # -- single date -------------------------------------------------------------
    for name, mask in perception_outputs.get("masks", {}).items():
        array = np.asarray(mask)
        fraction = measures.class_fraction(array, True)
        class_proportions[name] = fraction
        facts.append(
            Fact(
                key=f"{name}_fraction",
                value=fraction,
                unit="fraction",
                provenance="symbolic.measures.class_fraction",
                confidence=agreement_scores.get(name),
            )
        )
        if have_gsd:
            facts.append(
                Fact(
                    key=f"{name}_area",
                    value=measures.class_area_m2(array, True, float(gsd_m)),
                    unit="m2",
                    provenance="symbolic.measures.class_area_m2",
                    confidence=agreement_scores.get(name),
                )
            )

    # -- two dates ---------------------------------------------------------------
    masks_t1 = perception_outputs.get("masks_t1", {})
    masks_t2 = perception_outputs.get("masks_t2", {})
    if masks_t1 and masks_t2:
        for name in sorted(set(masks_t1) & set(masks_t2)):
            first = np.asarray(masks_t1[name])
            second = np.asarray(masks_t2[name])
            if not have_gsd:
                # Without a GSD the trend is still decidable from pixel counts, but the
                # absolute floor is expressed in square metres and cannot be applied.
                # Reporting a trend the threshold never sanctioned would be a guess.
                warnings.append(
                    f"{name}: no GSD, so the change floor of "
                    f"{CHANGE_ABSOLUTE_FLOOR_M2:.0f} m2 cannot be applied and no trend "
                    "is reported"
                )
                continue
            delta = measures.area_delta(first, second, True, float(gsd_m))
            trend = classify_trend(delta["absolute_m2"], delta["relative"])
            area_deltas[name] = AreaDelta(
                area_t1_m2=delta["area_t1_m2"],
                area_t2_m2=delta["area_t2_m2"],
                absolute_m2=delta["absolute_m2"],
                relative=delta["relative"],
                trend=trend,
            )
            facts.append(
                Fact(
                    key=f"{name}_delta",
                    value=delta["absolute_m2"],
                    unit="m2",
                    provenance="symbolic.measures.area_delta",
                    confidence=agreement_scores.get(name),
                )
            )
            gained = int(np.count_nonzero(~first.astype(bool) & second.astype(bool)))
            lost = int(np.count_nonzero(first.astype(bool) & ~second.astype(bool)))
            scale = float(gsd_m) ** 2
            change_transitions[f"to_{name}"] = gained * scale
            change_transitions[f"from_{name}"] = lost * scale

    # -- detections and components -----------------------------------------------
    boxes = perception_outputs.get("boxes", [])
    for label in sorted({box.get("label", "object") for box in boxes}):
        count = measures.object_count(boxes, label)
        object_counts[label] = count
        facts.append(
            Fact(
                key=f"{label}_count",
                value=count,
                unit="count",
                provenance="symbolic.measures.object_count",
            )
        )

    for name, components in perception_outputs.get("components", {}).items():
        object_counts.setdefault(name, len(components))
        facts.append(
            Fact(
                key=f"{name}_component_count",
                value=len(components),
                unit="count",
                provenance="symbolic.measures.connected_components",
            )
        )

    record = AnswerRecord(
        intent=intent,
        facts=facts,
        class_proportions=class_proportions,
        object_counts=object_counts,
        area_deltas=area_deltas,
        change_transitions=change_transitions,
        referenced_regions=list(perception_outputs.get("regions", [])),
        agreement_scores=agreement_scores,
        warnings=list(dict.fromkeys(warnings)),
    )
    record.validate_provenance()
    return record
