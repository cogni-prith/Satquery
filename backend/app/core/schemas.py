"""HTTP request and response shapes.

Deliberately thin. `ToolResult` and `Trace` are NOT redefined here -- they cross the wire
as the ML layer serialises them. A backend-side view model would be a second definition of
the contract, free to drift from the one the tools actually emit, and the trace is a scored
judging artifact that has to be the real thing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UploadedImage(BaseModel):
    """What an upload turned into, including anything ingest complained about."""

    image_id: str
    filename: str
    modality: str
    gsd_m: float | None
    gsd_token: str = Field(description="The frozen GSD token this image will carry.")
    width: int | None
    height: int | None
    band_names: list[str]
    crs: str | None
    warnings: list[str]


class QuerySubmission(BaseModel):
    query: str = Field(min_length=1)
    image_ids: list[str] = Field(min_length=1, max_length=2)
    task: str | None = Field(default=None, description="Force a task; None lets the gate decide.")


class JobAccepted(BaseModel):
    job_id: str


class JobView(BaseModel):
    """A job's current state. `result` and `trace` are the ML layer's own models."""

    job_id: str
    status: str
    result: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None
    error: str | None = None


class HealthView(BaseModel):
    models_loaded: bool
    loading: bool
    vram_used_mb: float | None
    contract_version: str
    detail: str | None = None
