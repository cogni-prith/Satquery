# SatQuery full stack — backend and frontend design

Date: 2026-08-31
Status: approved, implementing

## Context

The ML layer (`satquery` v0.1.0) is complete: 9 tools behind a versioned
`ToolRequest -> ToolResult` contract, 6 implemented, all 7 judging rows measured by one
`make eval` on freshly downloaded data. The team now owns the whole stack.

This spec covers `satquery-backend` (FastAPI) and `satquery-frontend` (React). Both are
**separate repositories**. `CLAUDE.md` forbids HTTP, frontend, queue and ORM code inside
`satquery-ml`, and that rule stands even though the same team now owns everything: the ML
repo's value is that it is a clean importable library with a typed boundary, and that
boundary is what caught the adapter regression, the double-render bug and two misplaced
checkpoints during development.

Target, decided: **judged demo now, hosted later.** Build for a single-machine live demo
where nothing may fail in front of anyone, but keep the job-handling and storage seams
swappable so hosting later is a module change rather than a rewrite.

## The binding constraint

RTX 4060 Laptop, **8188 MiB VRAM**. 15 GB system RAM. One machine, also running a browser
and possibly a screen share.

EarthDial-4B is 7.8 GB on disk and loads 4-bit NF4 at roughly 4 GB of VRAM. It fits once.
Everything below follows from that.

Measured latencies, from real runs: EarthDial cold load ~40 s; `vlm.caption` 3.5 s;
`fusion.extraction` 2.7 s; `change.vqa_head` milliseconds.

## Topology

Two processes on one machine.

```
browser ── HTTP ──> FastAPI (holds no models, no GPU)
                       |  asyncio.Queue + futures keyed by job id
                       v
                    GPU worker thread (models resident for process lifetime)
```

The API process holds nothing heavy, so it stays responsive while inference runs. The worker
is **single-threaded on purpose**: serialising inference is not a limitation here, it is the
only way 8 GB stays safe when two people click at once. A second concurrent request queues
rather than racing for VRAM.

VRAM budget, resident for the process lifetime:

| component | VRAM |
|---|---|
| EarthDial-4B, 4-bit NF4 | ~4.0 GB |
| fusion dual-encoder | ~0.4 GB |
| change VQA head | ~0.2 GB |
| **total resident** | **~4.6 GB** |
| headroom for activations, browser compositor | ~3.5 GB |

All three load once at startup (~45 s) and are never evicted. Reloading mid-demo is the
failure this design exists to prevent.

## API

| method | path | purpose |
|---|---|---|
| `POST` | `/api/images` | multipart upload; returns `image_id` plus parsed metadata |
| `POST` | `/api/query` | submit a query over uploaded images; returns `job_id` |
| `GET` | `/api/jobs/{job_id}` | poll status; returns `ToolResult` and `Trace` when done |
| `GET` | `/api/tools` | registry snapshot, so the frontend renders capabilities from truth |
| `GET` | `/api/health` | `models_loaded`, `vram_used_mb`, `contract_version` |

Job-based rather than synchronous. A 3.5 s call held open across a laptop's wifi during a
demo is a hung spinner waiting to happen; a job id with polling degrades visibly and
recoverably.

`ToolResult` and `Trace` cross the wire **unchanged** — serialised straight from the
Pydantic models the ML layer already produces. No reshaping, no backend-side view model.
This is what makes "auditable execution summary" a real artifact rather than a claim: the
trace the frontend renders is the trace the tools emitted.

`POST /api/images` parses each upload through `satquery.io.raster.read_image_ref`, so GSD
comes from the affine transform and the returned metadata is the same `ImageRef` the tools
will see. Upload warnings (no CRS, unrecognised bands) surface immediately in the UI rather
than at query time.

## The router

Two stages, per `CLAUDE.md`. Not a free-form ReAct loop.

**Stage one — deterministic gate.** `REGISTRY.candidates(input_config, modalities, gsd_m,
task)` narrows the tool set from parsed raster metadata. One image gives VQA, caption,
grounding; two images with different timestamps give change; two with different sensors give
fusion. Pure lookup, no model, always succeeds.

**Stage two — constrained selection.** Choose among the candidates by keyword match against
the query, scoring each candidate's task type against query terms. On no confident match,
fall back to the first candidate the gate returned, which is deterministic and always valid.

Every stage appends a `TraceStep`, **including fallback paths**, so the trace is well formed
even when selection is uncertain. A judge asking "how did it decide that?" sees the gate's
candidate list, the selection, and the reason.

The constrained-classification-via-LLM variant described in `CLAUDE.md` is deferred: with 6
implemented tools and a deterministic gate that usually narrows to 1–3 candidates, keyword
selection covers the demo, and a second model call would add latency and a failure mode for
no measured gain. The seam is a single `select()` function, so upgrading is local.

## Modules

```
satquery-backend/
  app/
    main.py              FastAPI app, lifespan starts and stops the worker
    config.py            settings from env; no hardcoded paths
    api/
      images.py          upload, parse, store; returns ImageRef metadata
      query.py           submit a job, poll a job
      meta.py            /api/tools and /api/health
    core/
      jobs.py            JobStore protocol + in-memory implementation
      blobs.py           BlobStore protocol + local-directory implementation
      schemas.py         request/response models for the HTTP layer only
    router/
      gate.py            stage one, wraps REGISTRY.candidates
      select.py          stage two, keyword selection with deterministic fallback
      trace.py           builds the Trace across both stages
    worker/
      runtime.py         owns the GPU: loads models once, runs one job at a time
      dispatch.py        asyncio.Queue bridge between API and worker thread
  tests/                 everything except the GPU worker, which is mocked
```

Each module has one responsibility. Nothing in `app/` imports torch except
`worker/runtime.py`.

## Seams for hosting later

| seam | demo implementation | hosted swap |
|---|---|---|
| `JobStore` | in-memory dict, TTL eviction | Redis |
| `BlobStore` | local directory under a configured root | S3 |
| `Dispatcher` | `asyncio.Queue` to one worker thread | Celery or a separate worker service |

Each is a Protocol with a single implementation now. Swapping one is a new class, not a
rewrite of callers.

## Error handling and degradation

What a judge sees, in each failure:

- **Models still loading** — `/api/health` reports `models_loaded: false`; the UI shows a
  loading state and disables submit. The frontend gates on this rather than failing a query.
- **Tool raises** — `BaseTool.run` already converts this into a `ToolResult` with `error`
  set and a well-formed trace. The API returns it as a completed job with an error, not a
  500. The UI shows the error and the trace.
- **Stub tool selected** — same path; the error names what is missing. Never a fabricated
  answer.
- **GPU OOM** — worker catches, returns an error result, and continues. Serialised execution
  makes this unlikely, and the worker does not die.
- **Malformed upload** — `read_image_ref` raises; the upload endpoint returns 400 with the
  parse error. Nothing enters the job queue.

## Testing

The heavy models cannot run in CI, so the worker is mocked at the dispatch boundary. Tested:
router gate and selection against the real registry; job lifecycle; upload parsing against a
synthetic GeoTIFF fixture; API contract shapes; that a tool error becomes a completed job
with an error rather than a 500.

Not tested in CI: actual inference. That is covered by `make eval` in the ML repo, which is
the correct place for it.

## What we are not building

- **Auth, users, multi-tenancy** — single-machine demo; no user model exists to protect.
- **WebSockets or streaming** — polling a job id is enough at 3.5 s latencies and has fewer
  failure modes on a laptop's wifi.
- **A database** — jobs are ephemeral, images are files. A schema would be ceremony.
- **Docker Compose** — two processes started by one script.
- **LLM-based router stage two** — deferred, seam left in place; see the router section.

## Frontend

React with Vite, TypeScript, no component library. Four surfaces:

1. **Upload** — drag or pick rasters; shows parsed GSD, modality, band names, and any
   ingest warnings immediately.
2. **Query** — free-text box; shows which tools the gate considers reachable given the
   current images, read from `/api/tools`.
3. **Result** — the answer, confidence, and evidence: boxes drawn over the image, mask and
   index rasters as downloadable links.
4. **Trace** — the execution summary, step by step, always shown. This is a scored judging
   row, so it is a first-class surface rather than a debug panel.

The frontend holds no domain logic. It renders what the contract returns.
