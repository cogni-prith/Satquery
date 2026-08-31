# satquery-frontend

React UI for **SatQuery AI** (Smart India Hackathon 2026, ISRO SIH26167).

Talks to `satquery-backend`. Holds no domain logic: it uploads rasters, submits a query,
and renders whatever the contract returns.

## What it does not do

There is no hardcoded list of capabilities here. Which tools exist, what they accept and
whether they are implemented all come from `GET /api/tools` — the same registry the router
reads — so there is no second source of truth free to drift.

Nor does it reshape results. `ToolResult` and `Trace` are rendered as the ML layer emitted
them.

## The four surfaces

1. **Imagery** — upload one raster (VQA, captioning, grounding) or two (change, or
   optical-plus-SAR fusion). Parsed GSD, modality, bands and any ingest warnings appear
   immediately, while the user is still looking at the file they picked.
2. **Question** — free text. The submit button is gated on `/api/health` reporting
   `models_loaded`, because the backend takes ~45 s to become ready and a spinner during
   that window is indistinguishable from a hang.
3. **Result** — answer, confidence, boxes, and the rasters written to disk. A tool that
   errored renders as a refusal, never as a blank answer: the ML layer refuses to fabricate
   and the UI must not hide that behind an empty box.
4. **Trace** — the execution summary, step by step, colour-coded by status. A first-class
   surface rather than a debug panel: "auditable execution summary" is a separately scored
   judging row, and it renders on failure paths too.

## Running

    npm install
    npm run dev        # http://localhost:5173

Vite proxies `/api` to `localhost:8000`, so the deployed build can be served same-origin
with no code change. Start the backend first.

## Note on confidence

Where the fusion tool supplies a confidence, it is **agreement between the learned model and
a deterministic spectral index**, not a softmax. The UI labels it as such. A number called
"confidence" that means two different things in two places is worse than no number at all.
