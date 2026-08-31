# satquery-backend

FastAPI service for **SatQuery AI** (Smart India Hackathon 2026, ISRO SIH26167).

It holds no models itself. It `pip`-installs the `satquery` package and calls its tools
through the versioned `ToolRequest -> ToolResult` contract — the same path `make eval` uses
in the ML repo, so there is exactly one inference implementation.

## Why the ML code is not in here

`satquery-ml` forbids HTTP, frontend, queue and ORM code, and that rule holds even though
one team now owns every repo. The ML layer's value is that it is a clean importable library
with a typed boundary; during development that boundary caught an adapter regression, a
double-render bug and two misplaced checkpoints. Dissolving it into a web app would cost the
thing doing the most work.

## Design

The binding constraint is **8188 MiB of VRAM**. EarthDial-4B needs ~2.4 GB of it loaded
4-bit, which fits once — so exactly one component may hold models, and it loads them at
startup and never reloads.

```
browser ── HTTP ──> FastAPI (no models, no GPU)
                       |  asyncio.Queue
                       v
                    GPU worker (models resident, one job at a time)
```

Inference is serialised on purpose. Two concurrent calls on an 8 GB card is how a live demo
becomes an OOM; a queue is how it becomes a short wait.

`ToolResult` and `Trace` cross the wire **unchanged** — serialised straight from the models
the ML layer produces. No backend-side view model, because the trace is a scored judging
artifact and a second definition of it would be free to drift.

## Running

    ./run.sh          # http://localhost:8000, models load in ~45 s

Watch `/api/health` for `models_loaded`. The frontend gates its submit button on it.

## API

| method | path | purpose |
|---|---|---|
| `POST` | `/api/images` | upload a raster; returns parsed `ImageRef` metadata |
| `POST` | `/api/query` | submit a query; returns a `job_id` |
| `GET` | `/api/jobs/{id}` | poll; returns `ToolResult` and `Trace` |
| `GET` | `/api/tools` | registry snapshot |
| `GET` | `/api/health` | `models_loaded`, `vram_used_mb`, `contract_version` |

## The router

Two stages, per the problem statement. Stage one is a deterministic gate over parsed raster
metadata (`REGISTRY.candidates`) — a lookup, no model, always succeeds. Stage two selects
among the candidates by keyword, with a deterministic fallback that says so in the trace.
Both stages append a `TraceStep`, including on failure paths.

## Tests

    PYTHONPATH=<ml-repo>/src:. python -m pytest tests -q     # 17 tests

The GPU worker is mocked; real inference is covered by `make eval` in the ML repo, which is
the correct place for it.
