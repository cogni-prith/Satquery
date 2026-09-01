"""Process-wide singletons, wired once.

Deliberately a module rather than FastAPI dependency injection: the GPU runtime must be one
instance for the process lifetime, and a DI graph that could construct a second one is a
graph that can exhaust an 8 GB card.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import SETTINGS
from app.core.blobs import BlobStore, LocalBlobStore
from app.core.jobs import JobStore, MemoryJobStore
from app.worker.dispatch import Dispatcher
from app.worker.runtime import GpuRuntime


@dataclass
class AppState:
    blobs: BlobStore
    jobs: JobStore
    runtime: GpuRuntime
    dispatcher: Dispatcher


def _build() -> AppState:
    blobs = LocalBlobStore(SETTINGS.upload_root)
    jobs = MemoryJobStore(ttl_seconds=SETTINGS.job_ttl_seconds)
    runtime = GpuRuntime()
    return AppState(blobs=blobs, jobs=jobs, runtime=runtime, dispatcher=Dispatcher(runtime, jobs))


STATE = _build()
