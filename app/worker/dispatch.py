"""The bridge between the async API and the single-threaded GPU worker.

A seam. The demo runs one worker thread fed by an `asyncio.Queue`; hosting later swaps this
for Celery or a separate worker service without callers changing.

Inference is synchronous, blocking, and holds the GPU. Running it on the event loop would
freeze every other request for 3.5 seconds, so it runs in a thread and the loop awaits the
result.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from app.core.jobs import Job, JobStatus, JobStore
from app.router.gate import gate
from app.router.select import select
from app.worker.runtime import GpuRuntime, build_request
from satquery.serve.contracts import ImageRef, StepStatus, TaskType, Trace, TraceStep
from satquery.utils.logging import get_logger

_LOG = get_logger("dispatch")

#: Version stamped on the router's own trace steps. Bumped when gate or selection changes,
#: so a stored trace says which routing logic produced it.
ROUTER_VERSION = "0.1.0"


@dataclass
class QueryTask:
    job_id: str
    query: str
    images: list[ImageRef]
    task: TaskType | None


class Dispatcher:
    """Accepts query tasks and runs them one at a time on the GPU runtime."""

    def __init__(self, runtime: GpuRuntime, jobs: JobStore) -> None:
        self.runtime = runtime
        self.jobs = jobs
        self._queue: asyncio.Queue[QueryTask] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._consume())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def submit(self, task: QueryTask) -> None:
        await self._queue.put(task)

    async def _consume(self) -> None:
        while True:
            task = await self._queue.get()
            try:
                await self._run(task)
            except Exception:  # noqa: BLE001 - one bad job must not kill the worker loop
                _LOG.exception("job %s crashed the dispatcher", task.job_id)
                job = self.jobs.get(task.job_id)
                job.status = JobStatus.FAILED
                job.error = "the worker failed unexpectedly; see server logs"
                job.finished_at = time.time()
                self.jobs.update(job)
            finally:
                self._queue.task_done()

    async def _run(self, task: QueryTask) -> None:
        job = self.jobs.get(task.job_id)
        job.status = JobStatus.RUNNING
        self.jobs.update(job)

        steps: list[TraceStep] = []

        # -- stage one: the deterministic gate
        started = time.perf_counter()
        config, candidates = gate(task.images, task.task)
        steps.append(TraceStep(
            step=len(steps) + 1,
            tool_name="router.gate",
            tool_version=ROUTER_VERSION,
            params={
                "input_config": config.value,
                "candidates": [spec.name for spec in candidates],
                "modalities": [image.modality.value for image in task.images],
            },
            latency_ms=(time.perf_counter() - started) * 1000.0,
            status=StepStatus.OK if candidates else StepStatus.ERROR,
            message=None if candidates else "no implemented tool serves this input",
        ))

        if not candidates:
            self._finish(job, None, Trace(
                request_id=job.job_id, input_config=config, task=task.task, steps=steps,
            ), error="no implemented tool can serve this combination of images")
            return

        # -- stage two: selection
        started = time.perf_counter()
        chosen, reason = select(candidates, task.query)
        steps.append(TraceStep(
            step=len(steps) + 1,
            tool_name="router.select",
            tool_version=ROUTER_VERSION,
            params={"reason": reason, "chosen": chosen.name, "chosen_version": chosen.version},
            latency_ms=(time.perf_counter() - started) * 1000.0,
            status=StepStatus.OK,
            message=reason,
        ))

        # -- inference, off the event loop
        request = build_request(task.query, task.images, chosen.task)
        result = await asyncio.to_thread(self.runtime.run, chosen.name, request)

        steps.append(TraceStep(
            step=len(steps) + 1,
            tool_name=result.tool_name,
            tool_version=result.tool_version,
            params=result.params_used,
            latency_ms=result.latency_ms,
            confidence=result.confidence,
            status=StepStatus.OK if result.ok else StepStatus.ERROR,
            message=result.error,
        ))

        trace = Trace(
            request_id=result.request_id,
            input_config=config,
            task=chosen.task,
            steps=steps,
        )
        self._finish(job, result, trace, error=result.error)

    def _finish(self, job: Job, result: Any, trace: Trace, error: str | None) -> None:
        """Record the outcome.

        A tool error is a COMPLETED job carrying an error, not a failed one: the tool ran,
        the trace is well formed, and the frontend should render both. `FAILED` is reserved
        for the service itself breaking.
        """
        job.status = JobStatus.DONE
        job.result = result.model_dump(mode="json") if result is not None else None
        job.trace = trace.model_dump(mode="json")
        job.error = error
        job.finished_at = time.time()
        self.jobs.update(job)
