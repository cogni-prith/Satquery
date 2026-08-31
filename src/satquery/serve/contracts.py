"""The wire contract between this package and the backend service.

This file and `models/registry.py` are the entire public API. The backend imports
them, reads the registry to decide what to call, and calls tools that all share one
shape::

    def run(request: ToolRequest) -> ToolResult

`ToolResult` is the only thing that crosses the boundary. If a piece of information
is not on `ToolResult`, the backend cannot see it and neither can a judge.

Changing anything in this file is a cross-team event: update `docs/INTERFACE.md` in
the same commit and say so in the commit message.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from satquery.preprocess.gsd import format_gsd_token

__all__ = [
    "CONTRACT_VERSION",
    "BoundingBox",
    "Evidence",
    "ImageRef",
    "InputConfig",
    "Modality",
    "StepStatus",
    "TaskType",
    "Tool",
    "ToolRequest",
    "ToolResult",
    "ToolSpec",
    "Trace",
    "TraceStep",
]

# Bumped on any breaking change to the models below. The backend pins it.
CONTRACT_VERSION = "1.0.0"


if sys.version_info >= (3, 11):  # pragma: no cover - depends on the interpreter
    from enum import StrEnum
else:  # pragma: no cover

    class StrEnum(str, Enum):
        """`enum.StrEnum` for Python 3.10.

        The project targets 3.11, but the working environment is the `tf-torch` conda
        env, which is 3.10 and carries a CUDA build of PyTorch that is worth far more
        than the version convention.

        Inheriting `(str, Enum)` already gives the comparison and JSON behaviour the
        wire contract needs. The explicit `__str__` restores the one thing plain
        `(str, Enum)` gets wrong: `f"{Modality.SAR}"` must render as `sar`, not
        `Modality.SAR`. On 3.11+ the real `StrEnum` is imported instead and nothing
        else changes.
        """

        def __str__(self) -> str:
            return str(self.value)


class Modality(StrEnum):
    """Sensor family of a single raster input."""

    OPTICAL_RGB = "optical_rgb"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    PANCHROMATIC = "panchromatic"


class InputConfig(StrEnum):
    """Shape of the input set, decided by `io/validate.py` before any tool runs."""

    SINGLE = "single"
    CROSS_MODAL_PAIR = "cross_modal_pair"
    BI_TEMPORAL_PAIR = "bi_temporal_pair"


class TaskType(StrEnum):
    """What the user is asking for. Stage one of the router narrows to a subset of these."""

    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_DESCRIPTION = "change_description"
    CHANGE_VQA = "change_vqa"
    CHANGE_MASK = "change_mask"
    FUSION_EXTRACTION = "fusion_extraction"


class StepStatus(StrEnum):
    """Outcome of one step in an execution trace."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class BoundingBox(BaseModel):
    """An axis-aligned box in the pixel grid of one input image.

    Coordinates are absolute pixels in `xyxy` order against the raster referenced by
    `image_index`, matching how VRSBench scores grounding (`acc@tau` on horizontal
    boxes). They are not normalised and not geographic; the backend converts to map
    coordinates using that image's affine transform when it needs to.
    """

    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(description="Left edge, pixels.")
    y_min: float = Field(description="Top edge, pixels.")
    x_max: float = Field(description="Right edge, pixels, strictly greater than x_min.")
    y_max: float = Field(description="Bottom edge, pixels, strictly greater than y_min.")
    label: str = Field(description="Class or referring phrase this box answers.")
    score: float | None = Field(default=None, ge=0.0, le=1.0, description="Detector confidence.")
    image_index: int = Field(
        default=0,
        ge=0,
        description="Index into ToolRequest.images that this box is drawn on.",
    )

    @model_validator(mode="after")
    def _check_ordering(self) -> BoundingBox:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(
                f"degenerate box: expected x_min < x_max and y_min < y_max, got "
                f"({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})"
            )
        return self

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Return the box as a plain `(x_min, y_min, x_max, y_max)` tuple."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class ImageRef(BaseModel):
    """One georeferenced raster input, as parsed from disk by `io/raster.py`.

    Everything the deterministic router gate needs is on this model. `gsd_m` is
    computed from `transform`, never from the filename and never assumed from the
    sensor name; when it cannot be computed it is `None` and `gsd_token` degrades to
    `<gsd:unknown>` rather than guessing.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path = Field(description="Absolute path to the raster on shared storage.")
    modality: Modality
    gsd_m: float | None = Field(
        default=None, gt=0.0, description="Ground sampling distance in metres, from the transform."
    )
    crs: str | None = Field(
        default=None, description='CRS as an authority string, e.g. "EPSG:32643".'
    )
    transform: tuple[float, float, float, float, float, float] | None = Field(
        default=None,
        description=(
            "Affine transform (a, b, c, d, e, f) where "
            "x = a*col + b*row + c and y = d*col + e*row + f."
        ),
    )
    timestamp: datetime | None = Field(default=None, description="Acquisition time, UTC.")
    band_names: list[str] = Field(
        default_factory=list,
        description="Canonical band names in stored order, already mapped by io/modality.py.",
    )
    width: int | None = Field(default=None, gt=0, description="Raster width in pixels.")
    height: int | None = Field(default=None, gt=0, description="Raster height in pixels.")
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal problems found while ingesting this raster."
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_computed(cls, data: Any) -> Any:
        """Ignore `gsd_token` on input so a serialised ImageRef round-trips.

        `gsd_token` is a computed field: it is emitted by `model_dump` but derived from
        `gsd_m`, so it must not be settable. Without this, JSON this model produced
        cannot be fed back into it -- which the backend hits the moment it replays a
        logged request. Dropping it here keeps `extra="forbid"` catching genuine typos.
        """
        if isinstance(data, dict) and "gsd_token" in data:
            data = {key: value for key, value in data.items() if key != "gsd_token"}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gsd_token(self) -> str:
        """The frozen GSD token that prefixes every instruction built from this image."""
        return format_gsd_token(self.gsd_m)

    @property
    def band_count(self) -> int:
        """Number of bands named on this reference."""
        return len(self.band_names)

    @property
    def shape(self) -> tuple[int, int] | None:
        """`(height, width)` when both are known, else `None`."""
        if self.height is None or self.width is None:
            return None
        return (self.height, self.width)


class Evidence(BaseModel):
    """Everything visual a tool produced, so a judge can check the answer against a map."""

    model_config = ConfigDict(extra="forbid")

    boxes: list[BoundingBox] = Field(default_factory=list)
    mask_path: Path | None = Field(
        default=None, description="Single-channel mask raster written under the artifact root."
    )
    overlay_path: Path | None = Field(default=None, description="Rendered RGB overlay for display.")
    index_maps: dict[str, Path] = Field(
        default_factory=dict,
        description='Deterministic index rasters by name, e.g. {"ndwi": ..., "ndbi": ...}.',
    )

    @property
    def is_empty(self) -> bool:
        """True when the tool returned text only."""
        return (
            not self.boxes
            and self.mask_path is None
            and self.overlay_path is None
            and not self.index_maps
        )


class ToolRequest(BaseModel):
    """One call from the backend into one tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="The user's natural-language question, verbatim.")
    images: list[ImageRef] = Field(
        min_length=1,
        max_length=2,
        description="One image, or an ordered pair. For bi-temporal pairs, earlier image first.",
    )
    task: TaskType | None = Field(
        default=None, description="Set once the router has classified; None before that."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters, validated against ToolSpec.param_schema.",
    )
    request_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Correlates request, result and trace. Generated by the backend.",
    )


class ToolResult(BaseModel):
    """The only object that crosses the boundary back to the backend.

    Keep it complete. A field left unpopulated is a field the backend, the frontend
    and the judges cannot see.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    tool_name: str
    tool_version: str
    answer: str | None = Field(default=None, description="Natural-language answer for the user.")
    evidence: Evidence = Field(default_factory=Evidence)
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated confidence. For tools with a deterministic index counterpart this is "
            "the agreement between the learned output and the index output."
        ),
    )
    params_used: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters actually applied, after defaults. Not the requested params.",
    )
    latency_ms: float = Field(default=0.0, ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = Field(
        default=None, description="Set when the tool failed. `answer` is then unreliable."
    )

    @property
    def ok(self) -> bool:
        """True when the tool completed without an error."""
        return self.error is None

    @model_validator(mode="after")
    def _answer_or_error(self) -> ToolResult:
        if self.error is None and self.answer is None and self.evidence.is_empty:
            raise ValueError(
                "a successful ToolResult must carry an answer or evidence; "
                "set `error` if the tool could not produce either"
            )
        return self


class TraceStep(BaseModel):
    """One tool invocation inside an auditable execution summary."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0, description="Zero-based position in the trace.")
    tool_name: str
    tool_version: str
    params: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0, ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: StepStatus = StepStatus.OK
    message: str | None = Field(
        default=None, description="Why a step warned, errored or was skipped."
    )


class Trace(BaseModel):
    """The auditable execution summary serialised on every call.

    The problem statement does not require or evaluate internal reasoning, only the
    observable trace, so this must always be well formed -- including when a tool
    fails. Build it with `add_step` / `record` rather than by hand.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    input_config: InputConfig
    task: TaskType | None = None
    steps: list[TraceStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _drop_computed(cls, data: Any) -> Any:
        """Ignore `total_latency_ms` on input so a serialised Trace round-trips.

        It is derived from the steps, so it must not be settable, but it IS emitted by
        `to_json`. A persisted trace has to be re-readable for an audit to mean anything.
        """
        if isinstance(data, dict) and "total_latency_ms" in data:
            data = {key: value for key, value in data.items() if key != "total_latency_ms"}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_latency_ms(self) -> float:
        """Sum of every step's latency."""
        return round(sum(step.latency_ms for step in self.steps), 3)

    def add_step(
        self,
        *,
        tool_name: str,
        tool_version: str,
        params: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
        confidence: float | None = None,
        status: StepStatus = StepStatus.OK,
        message: str | None = None,
    ) -> TraceStep:
        """Append a step, assigning its index. Returns the step just added."""
        step = TraceStep(
            step=len(self.steps),
            tool_name=tool_name,
            tool_version=tool_version,
            params=params or {},
            latency_ms=latency_ms,
            confidence=confidence,
            status=status,
            message=message,
        )
        self.steps.append(step)
        return step

    def record(self, result: ToolResult) -> TraceStep:
        """Append the step implied by a `ToolResult`, so trace and result cannot disagree."""
        if result.error is not None:
            status = StepStatus.ERROR
        elif result.warnings:
            status = StepStatus.WARNING
        else:
            status = StepStatus.OK
        return self.add_step(
            tool_name=result.tool_name,
            tool_version=result.tool_version,
            params=result.params_used,
            latency_ms=result.latency_ms,
            confidence=result.confidence,
            status=status,
            message=result.error or ("; ".join(result.warnings) or None),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the trace to JSON. This is what gets persisted for every call."""
        return self.model_dump_json(indent=indent)


class ToolSpec(BaseModel):
    """Declarative description of one tool, rich enough for the router gate to be a lookup.

    Stage one of the router is deterministic: parsed raster metadata narrows the
    candidate set. That only works if every constraint a tool has is declared here
    rather than discovered by calling it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description='Stable identifier, e.g. "vlm.vqa".')
    version: str = Field(description="Semantic version of the tool implementation.")
    task: TaskType
    accepted_modalities: list[Modality] = Field(min_length=1)
    accepted_input_configs: list[InputConfig] = Field(min_length=1)
    min_gsd_m: float | None = Field(
        default=None, gt=0.0, description="Finest GSD in metres this tool is valid for."
    )
    max_gsd_m: float | None = Field(
        default=None, gt=0.0, description="Coarsest GSD in metres this tool is valid for."
    )
    param_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for the permitted `ToolRequest.params`."
    )
    returns: list[str] = Field(
        default_factory=list, description="ToolResult fields this tool populates."
    )
    description: str = Field(
        default="", description="One line for the router's stage-two constrained classifier."
    )
    requires_gpu: bool = Field(default=True)
    implemented: bool = Field(
        default=False,
        description=(
            "False while the tool is an honest stub. The backend can list a spec and still "
            "know that calling it raises NotImplementedError."
        ),
    )

    @field_validator("accepted_modalities")
    @classmethod
    def _sort_modalities(cls, value: list[Modality]) -> list[Modality]:
        return sorted(set(value), key=lambda m: m.value)

    @field_validator("accepted_input_configs")
    @classmethod
    def _sort_configs(cls, value: list[InputConfig]) -> list[InputConfig]:
        return sorted(set(value), key=lambda c: c.value)

    @model_validator(mode="after")
    def _check_gsd_range(self) -> ToolSpec:
        if (
            self.min_gsd_m is not None
            and self.max_gsd_m is not None
            and self.min_gsd_m > self.max_gsd_m
        ):
            raise ValueError(
                f"min_gsd_m ({self.min_gsd_m}) must not exceed max_gsd_m ({self.max_gsd_m})"
            )
        return self

    @property
    def key(self) -> str:
        """`name@version`, the registry's unique key."""
        return f"{self.name}@{self.version}"

    def accepts_gsd(self, gsd_m: float | None) -> bool:
        """Whether this tool accepts imagery at `gsd_m`.

        An unknown GSD never disqualifies a tool. The hidden evaluation set may carry
        transforms we cannot interpret, and refusing to run at all is a worse failure
        than running with a `<gsd:unknown>` token and saying so in the warnings.
        """
        if gsd_m is None:
            return True
        below_range = self.min_gsd_m is not None and gsd_m < self.min_gsd_m
        above_range = self.max_gsd_m is not None and gsd_m > self.max_gsd_m
        return not (below_range or above_range)

    def accepts(
        self,
        *,
        input_config: InputConfig,
        modalities: list[Modality],
        gsd_m: float | None = None,
        task: TaskType | None = None,
    ) -> bool:
        """Whether this tool is a legal candidate for the given parsed input."""
        if task is not None and self.task is not task:
            return False
        if input_config not in self.accepted_input_configs:
            return False
        if not all(modality in self.accepted_modalities for modality in modalities):
            return False
        return self.accepts_gsd(gsd_m)


@runtime_checkable
class Tool(Protocol):
    """The shape every registered tool implements."""

    spec: ToolSpec

    def run(self, request: ToolRequest) -> ToolResult:
        """Execute the tool. Must return a `ToolResult` even on failure, with `error` set."""
        ...


def _json_default(value: Any) -> Any:
    """Fallback encoder so `Path` and `datetime` survive `json.dumps` in helper scripts."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def dumps(model: BaseModel, *, indent: int | None = 2) -> str:
    """Serialise any contract model to JSON. Thin wrapper kept for symmetry with `Trace.to_json`."""
    return json.dumps(model.model_dump(mode="json"), indent=indent, default=_json_default)
