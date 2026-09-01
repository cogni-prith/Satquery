"""Capability and health endpoints.

`/api/tools` returns the registry verbatim so the frontend renders what the system can
actually do from the same source the router reads. A hand-maintained capability list in the
UI would be a second source of truth, free to drift.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.schemas import HealthView
from app.state import STATE

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/tools")
async def tools() -> dict:
    from satquery.models.registry import REGISTRY
    from satquery.serve.contracts import CONTRACT_VERSION

    specs = REGISTRY.list_specs()
    return {
        "contract_version": CONTRACT_VERSION,
        "tool_count": len(specs),
        "tools": [spec.model_dump(mode="json") for spec in specs],
    }


@router.get("/health", response_model=HealthView)
async def health() -> HealthView:
    from satquery.serve.contracts import CONTRACT_VERSION

    return HealthView(
        models_loaded=STATE.runtime.loaded,
        loading=STATE.runtime.loading,
        vram_used_mb=STATE.runtime.vram_used_mb(),
        contract_version=CONTRACT_VERSION,
        detail=STATE.runtime.error,
    )
