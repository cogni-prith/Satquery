"""Submit a query, poll a job."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.jobs import JobStatus
from app.core.schemas import JobAccepted, JobView, QuerySubmission
from app.state import STATE
from app.worker.dispatch import QueryTask

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=JobAccepted, status_code=202)
async def submit(submission: QuerySubmission) -> JobAccepted:
    """Queue a query. Returns immediately with a job id.

    Rejects while models are still loading rather than queueing work that would sit behind
    a 45-second load: an explicit 503 lets the UI say "still starting" instead of showing a
    spinner that looks identical to a hang.
    """
    from satquery.serve.contracts import TaskType

    from app.worker.runtime import build_image_refs

    if not STATE.runtime.loaded:
        detail = STATE.runtime.error or "models are still loading"
        raise HTTPException(status_code=503, detail=detail)

    try:
        paths = [STATE.blobs.path_for(image_id) for image_id in submission.image_ids]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        images = build_image_refs(paths)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not parse an image: {exc}") from exc

    task = None
    if submission.task:
        try:
            task = TaskType(submission.task)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown task {submission.task!r}") from exc

    job = STATE.jobs.create()
    await STATE.dispatcher.submit(
        QueryTask(job_id=job.job_id, query=submission.query, images=images, task=task)
    )
    return JobAccepted(job_id=job.job_id)


@router.get("/jobs/{job_id}", response_model=JobView)
async def poll(job_id: str) -> JobView:
    """Current state of one job."""
    try:
        job = STATE.jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return JobView(
        job_id=job.job_id,
        status=job.status.value if isinstance(job.status, JobStatus) else str(job.status),
        result=job.result,
        trace=job.trace,
        error=job.error,
    )
