"""Job lifecycle.

A seam. The demo keeps jobs in a dict; hosting later swaps in Redis. Callers see a
Protocol, not a dict.

Jobs are the reason this API is asynchronous at all: a VQA call is 3.5 seconds, which is
long enough that holding an HTTP connection open across a laptop's wifi during a live demo
is a hung spinner waiting to happen.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    """One submitted query and, eventually, its result.

    `result` holds a serialised `ToolResult` and `trace` a serialised `Trace`, both exactly
    as the ML layer produced them. A tool that errored still produces a well-formed
    `ToolResult` with `error` set, so `DONE` with an error is a normal outcome and is not
    the same as `FAILED` -- which means the service itself broke.
    """

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    result: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class JobStore(Protocol):
    def create(self) -> Job: ...
    def get(self, job_id: str) -> Job: ...
    def update(self, job: Job) -> None: ...


class MemoryJobStore:
    """Jobs in a dict, evicted after a TTL so a long demo does not grow without bound."""

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._jobs: dict[str, Job] = {}
        self._ttl = ttl_seconds

    def create(self) -> Job:
        self._evict_expired()
        job = Job(job_id=uuid.uuid4().hex)
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"no job with id {job_id!r}")
        return job

    def update(self, job: Job) -> None:
        self._jobs[job.job_id] = job

    def _evict_expired(self) -> None:
        cutoff = time.time() - self._ttl
        for job_id, job in list(self._jobs.items()):
            if job.finished_at is not None and job.finished_at < cutoff:
                del self._jobs[job_id]
