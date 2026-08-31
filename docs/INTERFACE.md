# SatQuery ML — Backend Interface

The contract between `satquery-ml` and the FastAPI backend. Everything the backend needs to
call this package is here.

- **Contract version:** `CONTRACT_VERSION = "1.0.0"` (`satquery.serve.contracts`). Pin it.
- **Source of truth:** `src/satquery/serve/contracts.py` and `src/satquery/models/registry.py`.
  This document describes them; where they disagree, the code wins and this document is a bug.
- **Registry snapshot below generated from:**
  `PYTHONPATH=src python -m satquery.models.registry --json` — 9 tools, 1 implemented, 8 honest stubs.

---

## 1. Scope and stability

### What crosses the boundary

The backend imports this package in-process. There is no HTTP surface here, no queue, no
database — the ML layer is a library. Three things cross the boundary:

| Direction | Type | Module |
|---|---|---|
| backend → tool | `ToolRequest` | `satquery.serve.contracts` |
| tool → backend | `ToolResult` | `satquery.serve.contracts` |
| backend reads | `ToolSpec` via `REGISTRY` | `satquery.models.registry` |

The backend additionally builds a `Trace` (from `satquery.serve.contracts`) and serialises it
on every call. `Trace` is constructed by the caller, not returned by a tool, because one user
query can fan out to more than one tool.

Every tool has exactly one shape:

```python
def run(request: ToolRequest) -> ToolResult
```

**`ToolResult` is the only return type.** If a piece of information is not on `ToolResult`, the
backend cannot see it, the frontend cannot render it, and a judge cannot score it.

### Stability

Changing `serve/contracts.py` or `models/registry.py` is a **cross-team event**. The rule, from
`CLAUDE.md`:

> Update `docs/INTERFACE.md` in the same commit and say so in the commit message.

`CONTRACT_VERSION` is bumped on any breaking change to the models. Adding an optional field with
a default is not breaking; removing a field, renaming one, tightening a validator, or changing an
enum member's value is.

Tool-level versions are independent: `ToolSpec.version` is the semantic version of one tool's
implementation, and `ToolSpec.key` is `name@version`. The registry can hold several versions of
the same tool at once; `REGISTRY.get_spec(name)` with no version returns the highest.

---

## 2. Quick start for the backend

```python
from satquery.models.registry import REGISTRY
from satquery.serve.contracts import ImageRef, InputConfig, Modality, ToolRequest, Trace

# Stage one of the router: a pure, deterministic filter over declared metadata.
candidates = REGISTRY.candidates(InputConfig.SINGLE, [Modality.OPTICAL_RGB], gsd_m=0.6)

# Stage two picks one name from `candidates`; then:
result = REGISTRY.get("vlm.grounding").run(request)
```

Three lines, and that is the whole integration.

### Knowing what is real

`ToolSpec.implemented` is the honesty flag. The registry lists every spec whether or not its
weights exist, so the backend can build against the final shape of the API today:

```python
for spec in REGISTRY.list_specs():
    print(spec.name, spec.version, "implemented" if spec.implemented else "STUB")

ready = REGISTRY.list_specs(implemented_only=True)  # today: [indices.deterministic]
```

Calling a tool that has no implementation bound raises `NotImplementedError` from
`REGISTRY.get(...)`, naming what is missing:

```
NotImplementedError: vlm.caption@0.1.0 is registered as a spec but has no implementation
bound. Missing: the caption tool class in satquery.serve.tools and, for a learned tool,
trained weights under the artifact root.
```

It never returns a fabricated answer. That is deliberate: a placeholder string would pass a
smoke test and quietly poison an eval run weeks later.

Two distinct failure surfaces, and the backend must handle both:

| Where | What happens |
|---|---|
| `REGISTRY.get(name)` — no factory bound | raises `NotImplementedError` |
| `tool.run(request)` — anything inside the tool | returns `ToolResult` with `error` set; **never raises** |

`REGISTRY.get` also raises `KeyError` for an unknown name, listing the known tools.

### CLI

```
PYTHONPATH=src python -m satquery.models.registry
PYTHONPATH=src python -m satquery.models.registry --json
PYTHONPATH=src python -m satquery.models.registry --implemented-only
```

`--json` emits `{"contract_version": ..., "tool_count": ..., "tools": [ToolSpec, ...]}`. This is
also what `make registry` prints. If the backend wants the registry over the wire, serve that
JSON; do not re-encode it by hand.

---

## 3. Type reference

All models are Pydantic v2 with `model_config = ConfigDict(extra="forbid")`. An unexpected key
is a `ValidationError`, not a silently ignored field.

### Enum `Modality`

Sensor family of a single raster input.

| Member | Value |
|---|---|
| `OPTICAL_RGB` | `optical_rgb` |
| `MULTISPECTRAL` | `multispectral` |
| `SAR` | `sar` |
| `PANCHROMATIC` | `panchromatic` |

### Enum `InputConfig`

Shape of the input set, decided by `io/validate.py` before any tool runs.

| Member | Value |
|---|---|
| `SINGLE` | `single` |
| `CROSS_MODAL_PAIR` | `cross_modal_pair` |
| `BI_TEMPORAL_PAIR` | `bi_temporal_pair` |

### Enum `TaskType`

What the user is asking for. Stage one of the router narrows to a subset of these.

| Member | Value |
|---|---|
| `VQA` | `vqa` |
| `CAPTION` | `caption` |
| `GROUNDING` | `grounding` |
| `CHANGE_DESCRIPTION` | `change_description` |
| `CHANGE_VQA` | `change_vqa` |
| `CHANGE_MASK` | `change_mask` |
| `FUSION_EXTRACTION` | `fusion_extraction` |

### Enum `StepStatus`

Outcome of one step in an execution trace.

| Member | Value |
|---|---|
| `OK` | `ok` |
| `WARNING` | `warning` |
| `ERROR` | `error` |
| `SKIPPED` | `skipped` |

### `BoundingBox`

Axis-aligned box in the **pixel grid** of one input image, absolute pixels in `xyxy` order —
not normalised, not geographic. This matches how VRSBench scores grounding (`acc@tau` on
horizontal boxes). The backend converts to map coordinates with that image's affine transform
when it needs to.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `x_min` | `float` | yes | Left edge, pixels. |
| `y_min` | `float` | yes | Top edge, pixels. |
| `x_max` | `float` | yes | Right edge, pixels; strictly greater than `x_min`. |
| `y_max` | `float` | yes | Bottom edge, pixels; strictly greater than `y_min`. |
| `label` | `str` | yes | Class or referring phrase this box answers. |
| `score` | `float \| None` | no (`None`) | Detector confidence, `0.0..1.0`. |
| `image_index` | `int` | no (`0`) | Index into `ToolRequest.images` this box is drawn on; `>= 0`. |

Read-only properties: `xyxy -> tuple[float, float, float, float]`, `area -> float`.

**Validator.** `_check_ordering` rejects a degenerate box: `x_max <= x_min` or `y_max <= y_min`
raises `ValueError`. A zero-area box is not a valid answer and will not be accepted.

### `ImageRef`

One georeferenced raster input, as parsed from disk by `io/raster.py`. Everything the
deterministic gate needs is on this model.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `path` | `Path` | yes | Absolute path to the raster on shared storage. |
| `modality` | `Modality` | yes | Sensor family. |
| `gsd_m` | `float \| None` | no (`None`) | Ground sampling distance in metres, computed from `transform`; `> 0`. |
| `crs` | `str \| None` | no (`None`) | CRS as an authority string, e.g. `"EPSG:32643"`. |
| `transform` | `tuple[float × 6] \| None` | no (`None`) | Affine `(a, b, c, d, e, f)` with `x = a*col + b*row + c`, `y = d*col + e*row + f`. |
| `timestamp` | `datetime \| None` | no (`None`) | Acquisition time, UTC. |
| `band_names` | `list[str]` | no (`[]`) | Canonical band names in stored order, already mapped by `io/modality.py`. |
| `width` | `int \| None` | no (`None`) | Raster width in pixels; `> 0`. |
| `height` | `int \| None` | no (`None`) | Raster height in pixels; `> 0`. |
| `warnings` | `list[str]` | no (`[]`) | Non-fatal problems found while ingesting this raster. |
| `gsd_token` | `str` | computed | **Serialised on the wire.** See section 8. |

Read-only properties (not serialised): `band_count -> int`, `shape -> tuple[int, int] | None`.

`gsd_m` is computed from `transform`, never from the filename and never assumed from the sensor
name. When it cannot be computed it stays `None` and `gsd_token` degrades to `<gsd:unknown>`.

### `Evidence`

Everything visual a tool produced, so a judge can check the answer against a map.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `boxes` | `list[BoundingBox]` | no (`[]`) | Localisations. |
| `mask_path` | `Path \| None` | no (`None`) | Single-channel mask raster under the artifact root. |
| `overlay_path` | `Path \| None` | no (`None`) | Rendered RGB overlay for display. |
| `index_maps` | `dict[str, Path]` | no (`{}`) | Deterministic index rasters by name, e.g. `{"ndwi": ..., "ndbi": ...}`. |

Property: `is_empty -> bool`, true when the tool returned text only (no boxes, no mask, no
overlay, no index maps).

### `ToolRequest`

One call from the backend into one tool.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `query` | `str` | yes | The user's natural-language question, **verbatim**. `min_length=1`. |
| `images` | `list[ImageRef]` | yes | One image or an ordered pair. `min_length=1`, `max_length=2`. Bi-temporal pairs: earlier image first. |
| `task` | `TaskType \| None` | no (`None`) | Set once the router has classified; `None` before that. |
| `params` | `dict[str, Any]` | no (`{}`) | Tool parameters, validated against `ToolSpec.param_schema`. |
| `request_id` | `str` | no (uuid4 hex) | Correlates request, result and trace. Generated by the backend. |

**Validators.** `images` accepts only 1 or 2 entries — zero images or three raises
`ValidationError`. There is no three-image or time-series configuration in contract 1.0.0.
An empty `query` is rejected.

### `ToolResult`

The only object that crosses the boundary back. Keep it complete.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `request_id` | `str` | yes | Echoes `ToolRequest.request_id`. |
| `tool_name` | `str` | yes | The tool that produced this. |
| `tool_version` | `str` | yes | Its `ToolSpec.version`. |
| `answer` | `str \| None` | no (`None`) | Natural-language answer for the user. |
| `evidence` | `Evidence` | no (empty) | Boxes, mask, overlay, index maps. |
| `confidence` | `float \| None` | no (`None`) | `0.0..1.0`. For tools with a deterministic index counterpart this is the **agreement between the learned output and the index output**. |
| `params_used` | `dict[str, Any]` | no (`{}`) | Parameters actually applied **after defaults** — not the requested params. |
| `latency_ms` | `float` | no (`0.0`) | `>= 0`. Filled in by `BaseTool.run`. |
| `warnings` | `list[str]` | no (`[]`) | Non-fatal caveats. Spec-conformance warnings are prepended by `BaseTool.run`. |
| `error` | `str \| None` | no (`None`) | Set when the tool failed; `answer` is then unreliable. |

Property: `ok -> bool`, true when `error is None`.

**Validator.** `_answer_or_error` rejects a *successful* result that carries neither an answer
nor evidence:

```
ValueError: a successful ToolResult must carry an answer or evidence;
set `error` if the tool could not produce either
```

In other words a tool may not return silent emptiness. It either says something, shows
something, or admits failure.

### `TraceStep`

One tool invocation inside an auditable execution summary.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `step` | `int` | yes | Zero-based position in the trace; `>= 0`. Assigned by `Trace.add_step`. |
| `tool_name` | `str` | yes | |
| `tool_version` | `str` | yes | |
| `params` | `dict[str, Any]` | no (`{}`) | Parameters actually applied. |
| `latency_ms` | `float` | no (`0.0`) | `>= 0`. |
| `confidence` | `float \| None` | no (`None`) | `0.0..1.0`. |
| `status` | `StepStatus` | no (`ok`) | |
| `message` | `str \| None` | no (`None`) | Why a step warned, errored or was skipped. |

### `Trace`

The auditable execution summary serialised on **every** call. The problem statement does not
require or evaluate internal reasoning, only the observable trace — so this must always be well
formed, including when a tool fails.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `request_id` | `str` | yes | |
| `input_config` | `InputConfig` | yes | As decided by `io/validate.py`. |
| `task` | `TaskType \| None` | no (`None`) | |
| `steps` | `list[TraceStep]` | no (`[]`) | In execution order. |
| `created_at` | `datetime` | no (now, UTC) | |
| `total_latency_ms` | `float` | computed | **Serialised.** Sum of every step's latency, rounded to 3 dp. |

Methods — build a trace with these, never by hand:

| Method | Purpose |
|---|---|
| `add_step(*, tool_name, tool_version, params=None, latency_ms=0.0, confidence=None, status=OK, message=None) -> TraceStep` | Append a step, assigning its index. Use for skipped or gate-level steps. |
| `record(result: ToolResult) -> TraceStep` | Append the step implied by a `ToolResult`, so trace and result cannot disagree. |
| `to_json(*, indent=2) -> str` | Serialise. This is what gets persisted for every call. |

`record` derives `status` automatically: `error` set → `ERROR`; otherwise warnings present →
`WARNING`; otherwise `OK`. `message` becomes `result.error`, else the warnings joined with
`"; "`, else `None`.

### `ToolSpec`

Declarative description of one tool, rich enough that the router gate is a lookup rather than a
guess.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `name` | `str` | yes | Stable identifier, e.g. `"vlm.vqa"`. |
| `version` | `str` | yes | Semantic version of the implementation. |
| `task` | `TaskType` | yes | |
| `accepted_modalities` | `list[Modality]` | yes | `min_length=1`. Deduplicated and sorted by value on validation. |
| `accepted_input_configs` | `list[InputConfig]` | yes | `min_length=1`. Deduplicated and sorted by value on validation. |
| `min_gsd_m` | `float \| None` | no (`None`) | Finest GSD in metres this tool is valid for; `> 0`. `None` = no lower bound. |
| `max_gsd_m` | `float \| None` | no (`None`) | Coarsest GSD in metres; `> 0`. `None` = no upper bound. |
| `param_schema` | `dict[str, Any]` | no (`{}`) | JSON Schema for the permitted `ToolRequest.params`. |
| `returns` | `list[str]` | no (`[]`) | `ToolResult` fields this tool populates, dotted (e.g. `evidence.boxes`). |
| `description` | `str` | no (`""`) | One line for the router's stage-two constrained classifier. |
| `requires_gpu` | `bool` | no (`True`) | |
| `implemented` | `bool` | no (`False`) | `False` while the tool is an honest stub. |

Property: `key -> str`, `"name@version"`.

**Validators.**
- `_sort_modalities` / `_sort_configs` deduplicate and sort — so the serialised order is stable
  and the backend can compare specs byte-for-byte across runs.
- `_check_gsd_range` rejects `min_gsd_m > max_gsd_m` with
  `min_gsd_m (X) must not exceed max_gsd_m (Y)`.

Methods used by the gate:

| Method | Behaviour |
|---|---|
| `accepts_gsd(gsd_m: float \| None) -> bool` | `None` always returns `True`. See section 4. |
| `accepts(*, input_config, modalities, gsd_m=None, task=None) -> bool` | Task must match if given; `input_config` must be in the accepted list; **every** supplied modality must be accepted; then `accepts_gsd`. |

### `Tool` protocol

`runtime_checkable` `Protocol` with `spec: ToolSpec` and `run(request: ToolRequest) -> ToolResult`.
`run` must return a `ToolResult` even on failure, with `error` set.

### `ToolRegistry` (`satquery.models.registry`)

`REGISTRY` is the process-wide instance built from `BUILTIN_SPECS`. Backend-relevant surface:

| Call | Returns / raises |
|---|---|
| `REGISTRY.get_spec(name, version=None)` | `ToolSpec`; highest version when `version` is `None`. `KeyError` if unknown. |
| `REGISTRY.get(name, version=None)` | Live `Tool`, constructed and cached on first use. `KeyError` if unknown, `NotImplementedError` if no factory bound. |
| `REGISTRY.list_specs(implemented_only=False)` | `list[ToolSpec]`, ordered by name then version. |
| `REGISTRY.candidates(input_config, modalities, gsd_m=None, task=None)` | `list[ToolSpec]`, ordered by name then version. |
| `name in REGISTRY` | `bool`, matching either `name` or `name@version`. |
| `len(REGISTRY)` | Number of registered specs. |

Factories are lazy and cached, so model weights load on first use of a tool and only once.

### Serialisation helpers

`Trace.to_json(indent=2)` for traces. `satquery.serve.contracts.dumps(model, indent=2)` for any
other contract model — a thin wrapper that handles `Path` and `datetime`. Equivalent to
`model.model_dump_json(indent=2)`, which is what produced every JSON block below.

---

## 4. The router gate

The backend owns the router. This package owns its input. It is **two stages and it is not a
free-form ReAct loop.**

### Stage one — deterministic gate

Parsed raster metadata narrows the candidate tool set. One call, no model, no query text:

```python
candidates = REGISTRY.candidates(input_config, modalities, gsd_m=coarsest_gsd, task=None)
```

Pure filter over declared metadata, deterministic and ordered by name then version. Pass
`gsd_m` as the **coarsest** of the inputs.

Gate rules — what the input shape alone tells you:

| Parsed input | `InputConfig` | Reachable tasks |
|---|---|---|
| One image | `SINGLE` | `vqa`, `caption`, `grounding` (plus `fusion_extraction` via `indices.deterministic`) |
| Two images, **different timestamps**, same sensor | `BI_TEMPORAL_PAIR` | `change_description`, `change_vqa`, `change_mask` |
| Two images, **different sensors** | `CROSS_MODAL_PAIR` | `fusion_extraction` |

`io/validate.py` makes this decision authoritatively — it sees timestamps, CRS and footprints.
`satquery.models.base.infer_input_config_from_images` is the cheap fallback used inside a tool
when it is called directly without having been routed; it compares modalities only and treats a
same-modality pair as bi-temporal.

### Unknown GSD never excludes a tool

`ToolSpec.accepts_gsd(None)` returns `True`, always. From the docstring:

> The hidden evaluation set may carry transforms we cannot interpret, and refusing to run at
> all is a worse failure than running with a `<gsd:unknown>` token and saying so in the
> warnings.

So a `None` GSD narrows nothing. The tool still runs, `BaseTool.check_request` emits
`image N has no computable GSD; instruction will carry <gsd:unknown>`, and the warning reaches
the backend on `ToolResult.warnings` and the trace step's `message`.

An out-of-range **known** GSD is a different case: it excludes the tool from `candidates`, but
if the backend calls the tool anyway it is a warning, not a refusal — `check_request` emits
`image N GSD X.XX m is outside the declared range [...]` and the tool proceeds.

### Stage two — constrained classification

Given the stage-one candidate list, the backend asks its LLM to choose. Constraints:

1. Classify against a **strict Pydantic schema** — the chosen tool name must be one of the
   candidate names, and `params` must validate against that spec's `param_schema`.
2. **One retry** on a schema violation, feeding back the validation error.
3. Then **keyword fallback** — a deterministic rule over the query text. The router must always
   produce a well-formed trace, so there is no third path where it gives up.

`ToolSpec.description` is written to be the single line the stage-two classifier sees. Two
descriptions encode routing preferences directly:

- `detector.openvocab` — "The router prefers this over `vlm.grounding` when the query names a
  concrete object class rather than a spatial relation."
- `change.vqa_head` — "Scored separately from `vlm.change_description`; **the router calls
  both**."

### The CDVQA doctrine

For a bi-temporal question over the closed six-class CDVQA answer set, route **both** tools and
report both: `change.vqa_head` produces the scored closed-set answer in milliseconds,
`vlm.change_description` produces the prose the user reads. One `Trace` with two steps. Example 5
below is exactly this.

The six frozen classes, in the order the head's output layer uses
(`satquery.preprocess.constants.CDVQA_ANSWERS`):

`non-vegetated ground surface`, `buildings`, `playgrounds`, `water`, `low vegetation`, `trees`.

---

## 5. The registry table

Generated from `PYTHONPATH=src python -m satquery.models.registry --json`. Contract version
`1.0.0`, 9 tools, **1 implemented, 8 honest stubs**.

| Tool | Ver | Task | Modalities | Input configs | GSD (m) | Returns | GPU | Implemented |
|---|---|---|---|---|---|---|---|---|
| `change.mask` | 0.1.0 | `change_mask` | multispectral, optical_rgb | bi_temporal_pair | 0.1 – 30.0 | `evidence.mask_path`, `evidence.overlay_path`, `confidence` | yes | **no** |
| `change.vqa_head` | 0.1.0 | `change_vqa` | multispectral, optical_rgb | bi_temporal_pair | 0.1 – 120.0 | `answer`, `confidence` | yes | **no** |
| `detector.openvocab` | 0.1.0 | `grounding` | multispectral, optical_rgb | single | 0.1 – 30.0 | `evidence.boxes`, `evidence.overlay_path`, `confidence` | yes | **no** |
| `fusion.extraction` | 0.1.0 | `fusion_extraction` | multispectral, optical_rgb, panchromatic, sar | cross_modal_pair | 0.1 – 120.0 | `answer`, `evidence.mask_path`, `evidence.overlay_path`, `evidence.index_maps`, `confidence` | yes | **no** |
| `indices.deterministic` | 0.1.0 | `fusion_extraction` | multispectral, sar | cross_modal_pair, single | any – any | `answer`, `evidence.mask_path`, `evidence.index_maps`, `confidence` | **no** | **yes** |
| `vlm.caption` | 0.1.0 | `caption` | multispectral, optical_rgb, sar | single | 0.1 – 120.0 | `answer`, `confidence` | yes | **no** |
| `vlm.change_description` | 0.1.0 | `change_description` | multispectral, optical_rgb, sar | bi_temporal_pair | 0.1 – 120.0 | `answer`, `confidence` | yes | **no** |
| `vlm.grounding` | 0.1.0 | `grounding` | multispectral, optical_rgb | single | 0.1 – 120.0 | `answer`, `evidence.boxes`, `evidence.overlay_path`, `confidence` | yes | **no** |
| `vlm.vqa` | 0.1.0 | `vqa` | multispectral, optical_rgb, sar | single | 0.1 – 120.0 | `answer`, `confidence` | yes | **no** |

Notes the table cannot carry:

- The 0.1 – 120.0 m envelope is deliberately wide. Training is 10 m Sentinel and roughly 0.3 m
  aerial; the hidden evaluation set is sub-metre Cartosat-2S and RISAT SAR. A narrow declared
  range would make the gate refuse exactly the imagery we are scored on, so the range is wide
  and the per-image warning carries the caveat.
- `indices.deterministic` declares **no** GSD bounds at all. Closed-form indices have no training
  distribution to fall outside of. This is the safety net that still produces a defensible answer
  with a visible map on unseen sensor data — and it is the only tool that runs on CPU.
- `change.mask` and `detector.openvocab` cap at 30 m rather than 120 m.
- `fusion.extraction` is the only tool accepting `panchromatic`, because pan-sharpening feeds it.
- Two tools share `task=grounding` (`vlm.grounding`, `detector.openvocab`) and two share
  `task=fusion_extraction` (`fusion.extraction`, `indices.deterministic`). Filtering by task
  alone does not disambiguate; that is stage two's job.

`param_schema` for each tool is in the `--json` output; it is not duplicated here because a
hand-copied schema would drift.

---

## 6. Worked examples

Every JSON block below was produced by constructing the real Pydantic models and calling
`model_dump_json(indent=2)`. They all validate against contract `1.0.0`.

Two caveats:

> **Field values are illustrative, not measured.** No benchmark number in this repo is real yet;
> every score in every table is `TBD` until an eval has actually run. The confidences, latencies
> and answers below show the *shape* the backend should code against, nothing more.

> **Eight of these nine tools are stubs today.** Examples 1 through 5 show what the backend will
> receive once the weights exist. Calling those tools right now raises `NotImplementedError` from
> `REGISTRY.get(...)` — see section 7. Example 6 uses `indices.deterministic`, the one tool with
> `implemented=True`.

Paths below assume `SATQUERY_DATA_ROOT=/srv/satquery/data` and
`SATQUERY_ARTIFACT_ROOT=/srv/satquery/artifacts`. Real paths come from
`satquery.utils.paths`; nothing in this package hardcodes a location.

### Source of the five queries

> These five representative queries were recovered from a **third-party mirror** of the SIH 2026
> problem statement list, **not from sih.gov.in directly**. Confirm them against the official
> portal before anyone relies on the exact wording.

| # | Query | Task | Input config |
|---|---|---|---|
| 1 | "Describe the land-cover and major objects visible in this image." | `caption` | `SINGLE` |
| 2 | "Highlight the water body referred to in the query." | `grounding` | `SINGLE` |
| 3 | "What changed between these two dates, and where did the change occur?" | `change_description` (+ `change_mask`) | `BI_TEMPORAL_PAIR` |
| 4 | "Use the optical and SAR images together to identify built-up and water-covered regions." | `fusion_extraction` | `CROSS_MODAL_PAIR` |
| 5 | "Has the built-up area increased, decreased, or remained unchanged?" | `change_vqa` (+ `change_description`) | `BI_TEMPORAL_PAIR` |

---

### Example 1 — caption, `SINGLE`

Gate: one image → `SINGLE`. `REGISTRY.candidates(InputConfig.SINGLE, [Modality.MULTISPECTRAL], 1.6)`
returns `indices.deterministic`, `vlm.caption`, `vlm.grounding`, `vlm.vqa`; stage two picks
`vlm.caption`.

**ToolRequest**

```json
{
  "query": "Describe the land-cover and major objects visible in this image.",
  "images": [
    {
      "path": "/srv/satquery/data/cartosat2s/scene_0417_ms.tif",
      "modality": "multispectral",
      "gsd_m": 1.6,
      "crs": "EPSG:32643",
      "transform": [
        1.6,
        0.0,
        712340.0,
        0.0,
        -1.6,
        1423980.0
      ],
      "timestamp": null,
      "band_names": [
        "B02",
        "B03",
        "B04",
        "B08"
      ],
      "width": 512,
      "height": 512,
      "warnings": [],
      "gsd_token": "<gsd:1.6m>"
    }
  ],
  "task": "caption",
  "params": {
    "max_new_tokens": 128,
    "temperature": 0.0
  },
  "request_id": "8c1f2a9d4b6e47f0a3d5c7e9b1204f6a"
}
```

**ToolResult**

```json
{
  "request_id": "8c1f2a9d4b6e47f0a3d5c7e9b1204f6a",
  "tool_name": "vlm.caption",
  "tool_version": "0.1.0",
  "answer": "The scene is predominantly dense urban built-up land with a regular street grid. A river crosses the north-east corner, bordered by a narrow strip of low vegetation. Two large flat-roofed industrial buildings sit in the south-west quadrant, adjoined by a surfaced parking area.",
  "evidence": {
    "boxes": [],
    "mask_path": null,
    "overlay_path": null,
    "index_maps": {}
  },
  "confidence": 0.71,
  "params_used": {
    "max_new_tokens": 128,
    "temperature": 0.0,
    "num_beams": 1
  },
  "latency_ms": 1840.4,
  "warnings": [
    "image 0 GSD 1.60 m is far from the 10 m training distribution"
  ],
  "error": null
}
```

Note `params_used` carries `num_beams: 1`, a default the caller never asked for. That is the
point of the field: it is what was applied, not what was requested.

**Trace**

```json
{
  "request_id": "8c1f2a9d4b6e47f0a3d5c7e9b1204f6a",
  "input_config": "single",
  "task": "caption",
  "steps": [
    {
      "step": 0,
      "tool_name": "vlm.caption",
      "tool_version": "0.1.0",
      "params": {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "num_beams": 1
      },
      "latency_ms": 1840.4,
      "confidence": 0.71,
      "status": "warning",
      "message": "image 0 GSD 1.60 m is far from the 10 m training distribution"
    }
  ],
  "created_at": "2026-08-27T09:30:00Z",
  "total_latency_ms": 1840.4
}
```

A warning on the result promotes the step to `status: "warning"` automatically — `Trace.record`
derives it, so the trace cannot contradict the result.

---

### Example 2 — grounding, `SINGLE`

Gate: one image → `SINGLE`. The query is a referring expression ("the water body referred to in
the query") rather than a bare object class, so stage two prefers `vlm.grounding` over
`detector.openvocab`.

**ToolRequest**

```json
{
  "query": "Highlight the water body referred to in the query.",
  "images": [
    {
      "path": "/srv/satquery/data/vrsbench/images/val_02931.tif",
      "modality": "optical_rgb",
      "gsd_m": 0.6,
      "crs": "EPSG:32643",
      "transform": [
        0.6,
        0.0,
        712340.0,
        0.0,
        -0.6,
        1423980.0
      ],
      "timestamp": null,
      "band_names": [
        "B04",
        "B03",
        "B02"
      ],
      "width": 512,
      "height": 512,
      "warnings": [],
      "gsd_token": "<gsd:0.6m>"
    }
  ],
  "task": "grounding",
  "params": {
    "max_boxes": 5,
    "score_threshold": 0.25
  },
  "request_id": "1d7b40e6c2f34a8fb0921e5d6c8a3f47"
}
```

**ToolResult**

```json
{
  "request_id": "1d7b40e6c2f34a8fb0921e5d6c8a3f47",
  "tool_name": "vlm.grounding",
  "tool_version": "0.1.0",
  "answer": "One water body found: an irregular reservoir occupying the centre-left of the tile.",
  "evidence": {
    "boxes": [
      {
        "x_min": 88.0,
        "y_min": 161.5,
        "x_max": 307.0,
        "y_max": 344.0,
        "label": "the water body",
        "score": 0.82,
        "image_index": 0
      }
    ],
    "mask_path": null,
    "overlay_path": "/srv/satquery/artifacts/overlays/1d7b40e6c2f34a8fb0921e5d6c8a3f47_grounding.png",
    "index_maps": {}
  },
  "confidence": 0.82,
  "params_used": {
    "max_boxes": 5,
    "score_threshold": 0.25
  },
  "latency_ms": 1327.9,
  "warnings": [],
  "error": null
}
```

Box coordinates are absolute pixels against `images[0]` — `image_index: 0` says which. To draw
them on a map, apply that image's `transform`.

**Trace**

```json
{
  "request_id": "1d7b40e6c2f34a8fb0921e5d6c8a3f47",
  "input_config": "single",
  "task": "grounding",
  "steps": [
    {
      "step": 0,
      "tool_name": "vlm.grounding",
      "tool_version": "0.1.0",
      "params": {
        "max_boxes": 5,
        "score_threshold": 0.25
      },
      "latency_ms": 1327.9,
      "confidence": 0.82,
      "status": "ok",
      "message": null
    }
  ],
  "created_at": "2026-08-27T09:30:00Z",
  "total_latency_ms": 1327.9
}
```

---

### Example 3 — change description, `BI_TEMPORAL_PAIR`

Gate: two images, same modality, **different timestamps** → `BI_TEMPORAL_PAIR`. The query asks
both *what* changed and *where*, so the backend routes `vlm.change_description` for the prose and
`change.mask` for the pixels. Two `ToolResult`s, one `Trace`.

**ToolRequest** (earlier image first — the ordering is part of the contract)

```json
{
  "query": "What changed between these two dates, and where did the change occur?",
  "images": [
    {
      "path": "/srv/satquery/data/second/pairs/0142_t1.tif",
      "modality": "optical_rgb",
      "gsd_m": 0.5,
      "crs": "EPSG:32643",
      "transform": [
        0.5,
        0.0,
        712340.0,
        0.0,
        -0.5,
        1423980.0
      ],
      "timestamp": "2023-03-14T05:21:00Z",
      "band_names": [
        "B04",
        "B03",
        "B02"
      ],
      "width": 512,
      "height": 512,
      "warnings": [],
      "gsd_token": "<gsd:0.5m>"
    },
    {
      "path": "/srv/satquery/data/second/pairs/0142_t2.tif",
      "modality": "optical_rgb",
      "gsd_m": 0.5,
      "crs": "EPSG:32643",
      "transform": [
        0.5,
        0.0,
        712340.0,
        0.0,
        -0.5,
        1423980.0
      ],
      "timestamp": "2025-11-02T05:19:00Z",
      "band_names": [
        "B04",
        "B03",
        "B02"
      ],
      "width": 512,
      "height": 512,
      "warnings": [],
      "gsd_token": "<gsd:0.5m>"
    }
  ],
  "task": "change_description",
  "params": {
    "max_new_tokens": 192
  },
  "request_id": "a5e903cc71bd4d2e8f6047b9c2d81e35"
}
```

**ToolResult** — step 0, `vlm.change_description`

```json
{
  "request_id": "a5e903cc71bd4d2e8f6047b9c2d81e35",
  "tool_name": "vlm.change_description",
  "tool_version": "0.1.0",
  "answer": "Low vegetation across the western half of the tile has been replaced by buildings and non-vegetated ground surface. A new access road runs north to south through the centre. The tree cover along the eastern edge and the pond in the south-east corner are unchanged.",
  "evidence": {
    "boxes": [],
    "mask_path": null,
    "overlay_path": null,
    "index_maps": {}
  },
  "confidence": 0.68,
  "params_used": {
    "max_new_tokens": 192,
    "temperature": 0.0,
    "num_beams": 1
  },
  "latency_ms": 2210.6,
  "warnings": [],
  "error": null
}
```

**ToolResult** — step 1, `change.mask`

```json
{
  "request_id": "a5e903cc71bd4d2e8f6047b9c2d81e35",
  "tool_name": "change.mask",
  "tool_version": "0.1.0",
  "answer": null,
  "evidence": {
    "boxes": [],
    "mask_path": "/srv/satquery/artifacts/masks/a5e903cc71bd4d2e8f6047b9c2d81e35_change.tif",
    "overlay_path": "/srv/satquery/artifacts/overlays/a5e903cc71bd4d2e8f6047b9c2d81e35_change.png",
    "index_maps": {}
  },
  "confidence": 0.74,
  "params_used": {
    "threshold": 0.5
  },
  "latency_ms": 196.3,
  "warnings": [],
  "error": null
}
```

`answer` is `null` here and the result still validates, because `evidence` is non-empty. That is
`_answer_or_error` doing its job: say something *or* show something.

**Trace**

```json
{
  "request_id": "a5e903cc71bd4d2e8f6047b9c2d81e35",
  "input_config": "bi_temporal_pair",
  "task": "change_description",
  "steps": [
    {
      "step": 0,
      "tool_name": "vlm.change_description",
      "tool_version": "0.1.0",
      "params": {
        "max_new_tokens": 192,
        "temperature": 0.0,
        "num_beams": 1
      },
      "latency_ms": 2210.6,
      "confidence": 0.68,
      "status": "ok",
      "message": null
    },
    {
      "step": 1,
      "tool_name": "change.mask",
      "tool_version": "0.1.0",
      "params": {
        "threshold": 0.5
      },
      "latency_ms": 196.3,
      "confidence": 0.74,
      "status": "ok",
      "message": null
    }
  ],
  "created_at": "2026-08-27T09:30:00Z",
  "total_latency_ms": 2406.9
}
```

---

### Example 4 — optical + SAR fusion, `CROSS_MODAL_PAIR`

Gate: two images, **different sensors** → `CROSS_MODAL_PAIR`. Only `fusion.extraction` and
`indices.deterministic` accept that configuration, and only `fusion.extraction` accepts
`optical_rgb`/`panchromatic` alongside SAR.

The SAR image goes through the frozen rendering pipeline before the model sees it: refined Lee
filter (window 7), dB conversion, 2nd–98th percentile stretch, pseudo-RGB `R=VV, G=VH,
B=VV/VH ratio in dB`. Both polarisations are present here, so the single-pol fallback does not
fire.

**ToolRequest**

```json
{
  "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
  "images": [
    {
      "path": "/srv/satquery/data/cartosat2s/scene_0417_ms.tif",
      "modality": "multispectral",
      "gsd_m": 1.6,
      "crs": "EPSG:32643",
      "transform": [
        1.6,
        0.0,
        712340.0,
        0.0,
        -1.6,
        1423980.0
      ],
      "timestamp": "2025-12-04T05:12:00Z",
      "band_names": [
        "B02",
        "B03",
        "B04",
        "B08",
        "B11",
        "B12"
      ],
      "width": 512,
      "height": 512,
      "warnings": [],
      "gsd_token": "<gsd:1.6m>"
    },
    {
      "path": "/srv/satquery/data/risat/scene_0417_grd.tif",
      "modality": "sar",
      "gsd_m": 2.5,
      "crs": "EPSG:32643",
      "transform": [
        2.5,
        0.0,
        712340.0,
        0.0,
        -2.5,
        1423980.0
      ],
      "timestamp": "2025-12-05T17:44:00Z",
      "band_names": [
        "VV",
        "VH"
      ],
      "width": 512,
      "height": 512,
      "warnings": [],
      "gsd_token": "<gsd:2.5m>"
    }
  ],
  "task": "fusion_extraction",
  "params": {
    "targets": [
      "built_up",
      "water"
    ]
  },
  "request_id": "6b2c8f14ae7d4930b5c1e08d7f3a2496"
}
```

**ToolResult**

```json
{
  "request_id": "6b2c8f14ae7d4930b5c1e08d7f3a2496",
  "tool_name": "fusion.extraction",
  "tool_version": "0.1.0",
  "answer": "Built-up land covers roughly 38 percent of the footprint, concentrated along the northern and central corridors; water covers roughly 9 percent, all of it the river in the north-east and one rectangular tank in the south. The learned extraction and the NDWI/NDBI index layer agree on 91 percent of pixels.",
  "evidence": {
    "boxes": [],
    "mask_path": "/srv/satquery/artifacts/masks/6b2c8f14ae7d4930b5c1e08d7f3a2496_fusion.tif",
    "overlay_path": "/srv/satquery/artifacts/overlays/6b2c8f14ae7d4930b5c1e08d7f3a2496_fusion.png",
    "index_maps": {
      "ndwi": "/srv/satquery/artifacts/indices/6b2c8f14ae7d4930b5c1e08d7f3a2496_ndwi.tif",
      "ndbi": "/srv/satquery/artifacts/indices/6b2c8f14ae7d4930b5c1e08d7f3a2496_ndbi.tif",
      "sar_water": "/srv/satquery/artifacts/indices/6b2c8f14ae7d4930b5c1e08d7f3a2496_sar_water.tif"
    }
  },
  "confidence": 0.91,
  "params_used": {
    "targets": [
      "built_up",
      "water"
    ]
  },
  "latency_ms": 3104.8,
  "warnings": [
    "single-pol fallback not triggered; VV and VH both present"
  ],
  "error": null
}
```

`confidence` here is **not** a softmax. Per the `ToolResult.confidence` docstring, for a tool
with a deterministic index counterpart it is the agreement between the learned output and the
index output — the 0.91 and the "agree on 91 percent of pixels" in the answer are the same
number. `index_maps` ships the rasters that back the claim, so a judge can check it.

**Trace**

```json
{
  "request_id": "6b2c8f14ae7d4930b5c1e08d7f3a2496",
  "input_config": "cross_modal_pair",
  "task": "fusion_extraction",
  "steps": [
    {
      "step": 0,
      "tool_name": "fusion.extraction",
      "tool_version": "0.1.0",
      "params": {
        "targets": [
          "built_up",
          "water"
        ]
      },
      "latency_ms": 3104.8,
      "confidence": 0.91,
      "status": "warning",
      "message": "single-pol fallback not triggered; VV and VH both present"
    }
  ],
  "created_at": "2026-08-27T09:30:00Z",
  "total_latency_ms": 3104.8
}
```

---

### Example 5 — change VQA, both tools routed, `BI_TEMPORAL_PAIR`

This is the CDVQA doctrine made concrete. The scored answer comes from the discriminative
`change.vqa_head` (closed six-class set, ~40 ms); the prose the user reads comes from
`vlm.change_description` (~2 s). **Both run, both are reported, the trace has two steps.**

**ToolRequest**

```json
{
  "query": "Has the built-up area increased, decreased, or remained unchanged?",
  "images": [
    {
      "path": "/srv/satquery/data/second/pairs/0142_t1.tif",
      "modality": "optical_rgb",
      "gsd_m": 0.5,
      "crs": "EPSG:32643",
      "transform": [
        0.5,
        0.0,
        712340.0,
        0.0,
        -0.5,
        1423980.0
      ],
      "timestamp": "2023-03-14T05:21:00Z",
      "band_names": [
        "B04",
        "B03",
        "B02"
      ],
      "width": 512,
      "height": 512,
      "warnings": [],
      "gsd_token": "<gsd:0.5m>"
    },
    {
      "path": "/srv/satquery/data/second/pairs/0142_t2.tif",
      "modality": "optical_rgb",
      "gsd_m": 0.5,
      "crs": "EPSG:32643",
      "transform": [
        0.5,
        0.0,
        712340.0,
        0.0,
        -0.5,
        1423980.0
      ],
      "timestamp": "2025-11-02T05:19:00Z",
      "band_names": [
        "B04",
        "B03",
        "B02"
      ],
      "width": 512,
      "height": 512,
      "warnings": [],
      "gsd_token": "<gsd:0.5m>"
    }
  ],
  "task": "change_vqa",
  "params": {
    "answer_set": [
      "buildings",
      "non-vegetated ground surface",
      "low vegetation"
    ]
  },
  "request_id": "f0947ab3c58e4126ad3b60c9e2718d45"
}
```

`answer_set` is a subset of `CDVQA_ANSWERS` (19 values, not six); the spec's `param_schema` enumerates the legal
strings, and omitting the parameter defaults to all six.

**ToolResult** — step 0, `change.vqa_head` (the scored answer)

```json
{
  "request_id": "f0947ab3c58e4126ad3b60c9e2718d45",
  "tool_name": "change.vqa_head",
  "tool_version": "0.1.0",
  "answer": "buildings",
  "evidence": {
    "boxes": [],
    "mask_path": null,
    "overlay_path": null,
    "index_maps": {}
  },
  "confidence": 0.88,
  "params_used": {
    "answer_set": [
      "buildings",
      "non-vegetated ground surface",
      "low vegetation"
    ]
  },
  "latency_ms": 41.7,
  "warnings": [],
  "error": null
}
```

**ToolResult** — step 1, `vlm.change_description` (the prose)

```json
{
  "request_id": "f0947ab3c58e4126ad3b60c9e2718d45",
  "tool_name": "vlm.change_description",
  "tool_version": "0.1.0",
  "answer": "The built-up area has increased. Buildings now occupy the western half of the tile where low vegetation stood at the earlier date; no built-up parcel present in the earlier image has been removed.",
  "evidence": {
    "boxes": [],
    "mask_path": null,
    "overlay_path": null,
    "index_maps": {}
  },
  "confidence": 0.66,
  "params_used": {
    "max_new_tokens": 128,
    "temperature": 0.0,
    "num_beams": 1
  },
  "latency_ms": 2085.2,
  "warnings": [],
  "error": null
}
```

**Trace** — one request, two steps, both correlated by `request_id`

```json
{
  "request_id": "f0947ab3c58e4126ad3b60c9e2718d45",
  "input_config": "bi_temporal_pair",
  "task": "change_vqa",
  "steps": [
    {
      "step": 0,
      "tool_name": "change.vqa_head",
      "tool_version": "0.1.0",
      "params": {
        "answer_set": [
          "buildings",
          "non-vegetated ground surface",
          "low vegetation"
        ]
      },
      "latency_ms": 41.7,
      "confidence": 0.88,
      "status": "ok",
      "message": null
    },
    {
      "step": 1,
      "tool_name": "vlm.change_description",
      "tool_version": "0.1.0",
      "params": {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "num_beams": 1
      },
      "latency_ms": 2085.2,
      "confidence": 0.66,
      "status": "ok",
      "message": null
    }
  ],
  "created_at": "2026-08-27T09:30:00Z",
  "total_latency_ms": 2126.9
}
```

Which answer the frontend shows is the backend's call. Which one the CDVQA eval scores is not:
that is step 0.

---

### Example 6 — `indices.deterministic`, the implemented tool

The only tool with `implemented=True`, the only one with `requires_gpu=False`, and the only one
with no declared GSD bounds. Closed-form NDWI and NDBI over the frozen band order — no weights,
nothing to fall out of distribution.

**ToolRequest** — note `task` is `null`, which is legal: `task` is set once the router has
classified, and a direct call need not classify at all.

```json
{
  "query": "Where is the water in this scene?",
  "images": [
    {
      "path": "/srv/satquery/data/bigearthnet/S2A_MSIL2A_20180526T94031_63_43.tif",
      "modality": "multispectral",
      "gsd_m": 10.0,
      "crs": "EPSG:32643",
      "transform": [
        10.0,
        0.0,
        712340.0,
        0.0,
        -10.0,
        1423980.0
      ],
      "timestamp": null,
      "band_names": [
        "B02",
        "B03",
        "B04",
        "B08",
        "B11",
        "B12"
      ],
      "width": 120,
      "height": 120,
      "warnings": [],
      "gsd_token": "<gsd:10.0m>"
    }
  ],
  "task": null,
  "params": {
    "indices": [
      "ndwi",
      "ndbi"
    ],
    "write_maps": true
  },
  "request_id": "2e5a71c0d94f43b7962c8ea50f1b6d38"
}
```

**ToolResult**

```json
{
  "request_id": "2e5a71c0d94f43b7962c8ea50f1b6d38",
  "tool_name": "indices.deterministic",
  "tool_version": "0.1.0",
  "answer": "NDWI exceeds 0.0 on 11.4 percent of pixels, forming one connected component in the south-east. NDBI exceeds 0.0 on 27.1 percent of pixels.",
  "evidence": {
    "boxes": [],
    "mask_path": "/srv/satquery/artifacts/masks/2e5a71c0d94f43b7962c8ea50f1b6d38_ndwi_binary.tif",
    "overlay_path": null,
    "index_maps": {
      "ndwi": "/srv/satquery/artifacts/indices/2e5a71c0d94f43b7962c8ea50f1b6d38_ndwi.tif",
      "ndbi": "/srv/satquery/artifacts/indices/2e5a71c0d94f43b7962c8ea50f1b6d38_ndbi.tif"
    }
  },
  "confidence": null,
  "params_used": {
    "indices": [
      "ndwi",
      "ndbi"
    ],
    "write_maps": true
  },
  "latency_ms": 63.5,
  "warnings": [],
  "error": null
}
```

`confidence` is `null` and that is correct, not an omission. Confidence in this contract means
learned-versus-index agreement; a closed-form index has nothing to disagree with. When
`fusion.extraction` runs, *its* confidence is the agreement with these maps.

**Trace**

```json
{
  "request_id": "2e5a71c0d94f43b7962c8ea50f1b6d38",
  "input_config": "single",
  "task": "fusion_extraction",
  "steps": [
    {
      "step": 0,
      "tool_name": "indices.deterministic",
      "tool_version": "0.1.0",
      "params": {
        "indices": [
          "ndwi",
          "ndbi"
        ],
        "write_maps": true
      },
      "latency_ms": 63.5,
      "confidence": null,
      "status": "ok",
      "message": null
    }
  ],
  "created_at": "2026-08-27T09:30:00Z",
  "total_latency_ms": 63.5
}
```

> **Binding status.** The spec is `implemented=True`, but the factory lives in
> `satquery/serve/tools.py`, which is not committed yet. Until it lands,
> `REGISTRY.get("indices.deterministic")` raises `ModuleNotFoundError` for
> `satquery.serve.tools`. The request and result above are constructed from the real models and
> validate; the live binding is the last step.

---

## 7. Error handling

### A tool never raises out of `run()`

`BaseTool.run` wraps `_run` in try/except and catches **everything** — `NotImplementedError`
gets its own branch, and a bare `except Exception` catches the rest, because a tool must never
take the service down. Both paths return a `ToolResult` built by `_error_result`.

What the backend sees on failure:

| Field | Value |
|---|---|
| `error` | Non-`None`. `NotImplementedError` becomes `"tool is not implemented: {message}"`; anything else becomes `"{ExceptionType}: {message}"`. |
| `answer` | `None` |
| `evidence` | Empty `Evidence()` |
| `confidence` | `None` |
| `params_used` | `dict(request.params)` — the requested params, since no defaults were applied |
| `latency_ms` | Time to failure, still measured |
| `warnings` | The spec-conformance warnings from `check_request`, preserved |
| `ok` | `False` |

The `_answer_or_error` validator permits `answer=None` with empty evidence **only** when `error`
is set — which is exactly the failure shape.

```json
{
  "request_id": "8c1f2a9d4b6e47f0a3d5c7e9b1204f6a",
  "tool_name": "vlm.caption",
  "tool_version": "0.1.0",
  "answer": null,
  "evidence": {
    "boxes": [],
    "mask_path": null,
    "overlay_path": null,
    "index_maps": {}
  },
  "confidence": null,
  "params_used": {
    "max_new_tokens": 128,
    "temperature": 0.0
  },
  "latency_ms": 0.412,
  "warnings": [
    "image 0 GSD 1.60 m is far from the 10 m training distribution"
  ],
  "error": "tool is not implemented: vlm.caption requires the LoRA adapter under <artifact_root>/checkpoints/vlm_caption; no weights found"
}
```

### The trace stays well formed

`Trace.record` maps `error` to `StepStatus.ERROR` and copies the message across. A failed call
still produces a complete, serialisable trace — which is the part the problem statement actually
evaluates.

```json
{
  "request_id": "8c1f2a9d4b6e47f0a3d5c7e9b1204f6a",
  "input_config": "single",
  "task": "caption",
  "steps": [
    {
      "step": 0,
      "tool_name": "vlm.caption",
      "tool_version": "0.1.0",
      "params": {
        "max_new_tokens": 128,
        "temperature": 0.0
      },
      "latency_ms": 0.412,
      "confidence": null,
      "status": "error",
      "message": "tool is not implemented: vlm.caption requires the LoRA adapter under <artifact_root>/checkpoints/vlm_caption; no weights found"
    }
  ],
  "created_at": "2026-08-27T09:30:00Z",
  "total_latency_ms": 0.412
}
```

### The one place that does raise

`REGISTRY.get(name)` — *acquiring* the tool, not running it — raises:

| Exception | When |
|---|---|
| `KeyError` | No tool with that name or `name@version`. Message lists the known tools. |
| `NotImplementedError` | Spec registered, no factory bound. Message names the missing tool class and, for a learned tool, the missing weights under the artifact root. |

Recommended backend pattern: check `spec.implemented` before routing to a tool, and wrap
`REGISTRY.get` in try/except so a stub degrades to a fallback tool rather than a 500. When
degrading, add a `SKIPPED` step to the trace with the reason, so the audit trail records the
substitution.

### Warnings are not errors

`BaseTool.check_request` returns warnings, never exceptions, for a request outside the declared
spec — wrong input config, unaccepted modality, out-of-range or missing GSD, plus any
`ImageRef.warnings` from ingest. Deliberate: on the hidden evaluation set an out-of-range GSD or
an unparsed transform is likely, and answering with a caveat beats refusing to answer. These
warnings are prepended to `ToolResult.warnings` and surface on the trace step's `message`.

---

## 8. The GSD token

Every instruction string handed to a model is prefixed with the ground sampling distance of its
input:

```
<gsd:10.0m> Describe the land-cover and major objects visible in this image.
<gsd:0.6m> Highlight the water body referred to in the query.
<gsd:unknown> Describe the land-cover and major objects visible in this image.
```

Format, frozen in `satquery.preprocess.constants`: `GSD_TOKEN_FORMAT = "<gsd:{value:.1f}m>"`,
`GSD_TOKEN_UNKNOWN = "<gsd:unknown>"`.

This matters more than it looks. Training data is Sentinel at 10 m; the hidden ISRO evaluation
set is Cartosat-2S at sub-metre to roughly 2 m and RISAT SAR. The token is the single strongest
signal the model has about that ~20x resolution gap, and it must be **byte-identical at training
time and inference time**.

### Rules

1. **Computed from the affine transform**, never from the filename and never assumed from the
   sensor name. `gsd_from_transform` takes the norms of the transform's two column vectors, so
   it stays correct for rotated transforms, and returns the geometric mean of the two pixel edges.
2. **Unknown means unknown.** A missing, non-finite, non-positive, or sub-representable GSD
   (below 0.05 m, which would format as the misleading `<gsd:0.0m>`) yields `<gsd:unknown>`. Never
   a guess.
3. **The backend must not construct the token itself.** Read `ImageRef.gsd_token`. It is a
   `computed_field`, so it is present in `model_dump()` and in the JSON — every example above
   shows it. Formatting it by hand in backend code is exactly the drift this section exists to
   prevent.
4. Prefixing is **idempotent**: `prefix_instruction` returns an already-tokenised string
   unchanged, so a second pass cannot produce two tokens and break the pattern the model was
   trained on.

A geographic CRS (degrees, not metres) needs a centre latitude to convert; without one, GSD is
reported as unknown rather than guessed, and when it is converted a warning says so and asks for
a projected CRS.

---

## Change log

| Contract version | Change |
|---|---|
| 1.0.0 | Initial contract. 9 registered tool specs, 1 implemented. |
