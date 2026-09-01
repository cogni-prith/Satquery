# satquery-ml

The **machine learning layer** of SatQuery AI, our entry for Smart India Hackathon 2026,
ISRO problem statement **SIH26167** — *"SatQuery AI: An Interactive Vision-Language
Assistant for Multimodal Remote Sensing Image Analysis through Text Queries."*

You point it at satellite imagery, ask a question in plain English, and it answers with
text plus a map you can check the answer against.

> **This repo is ML only.** A different team owns the FastAPI backend. A third owns the
> React frontend. They import this package and call its tools. See
> [What this repo will never contain](#what-this-repo-will-never-contain).

---

## Start here (about 60 seconds)

```
make install
make test
make smoke
```

- `make install` installs the CPU-only dependencies into the **`tf-torch` miniconda
  environment**, which is what every `make` target uses. It deliberately does *not*
  touch PyTorch, which is already there.
- `make test` runs 214 unit tests. Takes under a second.
- `make smoke` builds fake satellite images in a temp folder, pushes them through the
  whole ingest and preprocessing pipeline, and prints what came out. **This is the best
  single command for seeing what the repo actually does.** Read its output.

Then look at the tool catalogue the backend team consumes:

```
make registry
```

Run `make` on its own to list every available command.

Check what you are actually running against:

```
make env
```

> **Environment.** Everything runs in the `tf-torch` miniconda env
> (`~/miniconda3/envs/tf-torch`), which has Python 3.10, torch 2.13 with working CUDA on
> an RTX 4060, TensorFlow 2.15 and transformers 5.1. Override per-invocation with
> `make PY=/path/to/python <target>`.
>
> Note this is **Python 3.10**, while the architecture specifies 3.11. The code is written to
> run on both: `serve/contracts.py` imports the real `enum.StrEnum` on 3.11+ and falls
> back to an equivalent shim on 3.10. A working CUDA build of torch is worth more than
> the version convention.
>
> **Pinned versions that matter.** `torchvision` must match torch exactly
> (`0.28.0+cu130` for torch `2.13.0+cu130`); a mismatch breaks torchvision, timm, peft
> and every transformers model class at once. `transformers` must be **< 5**: version 5
> rewrote the generation loop and the EarthDial/InternVL remote code is written against
> the 4.x contract. Verified working on **transformers 4.49.0**.

---

## Where everything lives

### First: two things that look like duplicates but aren't

You will notice `data`, `train` and `eval` each appear twice in the tree. This is
deliberate, not a mistake:

| Path | What it is |
|---|---|
| `src/satquery/data/` | **Code.** Python that loads datasets. |
| `configs/data/` | **Settings.** YAML files saying *which* dataset, where, what split. |
| `data/` (top level) | **Nothing.** An empty folder where real datasets get symlinked in. |

`configs/` mirrors the code package names on purpose, so the settings for
`src/satquery/train/` are always at `configs/train/`. Once you see the pattern it stops
being confusing.

### The tree

```
satquery-ml/
├── the architecture            Project law. The rules everything here obeys. Read it.
├── README.md            You are here.
├── Makefile             Every command you need. Run `make` to list them.
├── pyproject.toml       Dependencies. Base group = CPU only; `gpu` group = torch etc.
├── .env.example         Copy to `.env` and set your data/artifact paths.
│
├── src/satquery/        ALL the code. Nothing importable lives outside here.
│   ├── serve/
│   │   ├── contracts.py   ★ THE API. Every type crossing to the backend.
│   │   └── tools.py         Tool implementations + registry wiring.
│   ├── models/
│   │   ├── registry.py    ★ THE CATALOGUE. What tools exist and when to use each.
│   │   ├── base.py          Shared tool plumbing (timing, error capture).
│   │   ├── vlm/             Vision-language backbone: VQA, captioning, grounding.
│   │   ├── change/          Bi-temporal change: CDVQA head + change mask.
│   │   ├── grounding/       Open-vocabulary object detector.
│   │   └── fusion/          Optical + SAR joint extraction.
│   ├── preprocess/
│   │   ├── constants.py   ★ FROZEN NUMBERS. Never copy these anywhere else.
│   │   ├── gsd.py           Resolution: compute it, format it, snap it.
│   │   ├── sar.py           Radar rendering (speckle filter → dB → pseudo-RGB).
│   │   ├── optical.py       Band ordering, pan-sharpening, contrast stretch.
│   │   └── indices.py       NDWI / NDBI / NDVI. The safety net.
│   ├── io/                  Reading GeoTIFFs, identifying sensors, pairing images.
│   ├── data/                Dataset loaders and the training mix.
│   ├── train/               LoRA fine-tuning and the change-head trainer.
│   ├── eval/                Scoring harness, metrics, report writer.
│   └── utils/               Paths, seeding, logging.
│
├── configs/             YAML settings, mirroring the package names above.
├── scripts/             The commands `make` runs. Entry points only.
├── tests/               214 tests. All CPU, all on synthetic data.
├── docs/
│   ├── INTERFACE.md     ★ What the backend team reads. The API contract.
│   ├── DATA_CARDS.md      One card per dataset.
│   └── EVAL_PROTOCOL.md   How every score is computed.
│
├── data/                Empty. Symlink real datasets in. Git-ignored.
├── artifacts/           Empty. Checkpoints and results land here. Git-ignored.
└── scratch/             Empty. Your mess goes here. Git-ignored.
```

**The four files marked ★ are the ones that matter most.** If you read nothing else,
read `preprocess/constants.py` and `serve/contracts.py`.

---

## What works right now

The repo is a **scaffold**: the plumbing is real and tested, the learned models are not
trained yet.

| Area | Status |
|---|---|
| `serve/contracts.py` — the API types | **Working**, fully validated |
| `models/registry.py` — the tool catalogue | **Working**, 9 tools registered |
| `preprocess/` — GSD, SAR, optical, indices | **Working**, 100+ tests |
| `io/` — GeoTIFF ingest, sensor ID, pair checks | **Working**, tested |
| `data/schema.py` — the training record | **Working** |
| `eval/answer_norm.py`, grounding + VQA metrics | **Working**, tested |
| `indices.deterministic` tool | **Working** end to end |
| `vlm.vqa`, `vlm.caption`, `vlm.grounding` | **Working.** `make infer` runs them |
| EarthDial-4B checkpoint | **Downloaded** (7.8 GB), repaired, and **generating answers** |
| Everything else (5 of 9 tools, all training, all loaders) | **Honest stubs** |

**"Honest stub"** means: calling it raises `NotImplementedError` naming exactly what is
missing. It never returns a fake answer. This is a rule, not an accident — a stub that
returns something plausible passes a smoke test and silently ruins an evaluation run
three weeks later.

```
>>> REGISTRY.get("vlm.vqa").run(request).error
"tool is not implemented: vlm.vqa is not implemented. Missing: constrained-decoding
 generation over the VQA answer set against a loaded backbone. ... the fine-tuned LoRA
 adapter ... has not been trained, so there is no weight to run and a fabricated answer
 would corrupt eval."
```

---

## What to do next

In order. Each step unblocks the ones below it.

### 1. Point the repo at some disk (5 minutes)

```
cp .env.example .env
```

Edit `.env` and set `SATQUERY_DATA_ROOT` (where datasets will live) and
`SATQUERY_ARTIFACT_ROOT` (where checkpoints and results go). Both must be outside this
repo — datasets are hundreds of gigabytes.

### 2. Get the datasets (hours, mostly waiting)

```
make install && .venv/bin/python scripts/download_datasets.py
```

This **prints instructions and downloads nothing**, on purpose. It tells you the exact
command per dataset and where to put the result. Run those commands yourself. Start with
**VRSBench** and **CDVQA** — they are the smallest and unblock the most.

Then check the mix config loads: `.venv/bin/python scripts/build_mix.py`

### 3. Write one dataset loader (the real first coding task)

Open `src/satquery/data/datasets/vrsbench.py`. It has the full method signatures and
docstrings; `__len__` and `__getitem__` raise `NotImplementedError`.

Your job: make `__getitem__` return a `Sample` (defined in `src/satquery/data/schema.py`
— read that first). Everything downstream already knows how to consume a `Sample`, so
once one loader is real, the mixer, collator and eval harness all start working for it.

Add a test in `tests/` that constructs a couple of records and checks the schema holds.

### 4. Repair torchvision, then smoke-test the model

The EarthDial-4B checkpoint is **already downloaded** (7.8 GB, in the HF cache) and its
missing remote code is **already patched in** (`make fetch-model`). Two things still
stand between you and a working inference call:

The `tf-torch` env ships `torchvision 0.19.0` against `torch 2.13.0`. Those do not match,
and the mismatch silently breaks `timm` and `peft` too. Install the matching build from
the same CUDA index the torch came from:

```
~/miniconda3/envs/tf-torch/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu130 "torchvision==0.28.0"
```

Verify with `~/miniconda3/envs/tf-torch/bin/python -c "import torchvision, timm, peft"`.

**Why this is a hard blocker, not a nicety:** every transformers model class resolves
through `transformers.modeling_utils`, which imports torchvision. With the ABI broken,
even `from transformers import LlamaForCausalLM` fails. No model of any kind can load in
this env until it is fixed.

**Second risk, only checkable after the above.** The checkpoint was saved with
transformers 4.37.2; the env has 5.1.0. On the first load attempt the custom config code
logged `vision_config is None` and `llm_config is None ... Initializing the LlamaConfig`
— but this model is InternViT + **Phi-3**, not Llama. That means transformers 5 is not
parsing the nested configs from `config.json`, and would build the wrong architecture.
If that persists once torchvision is fixed, pin transformers to the 4.x line the
checkpoint expects, ideally in a dedicated env so the shared `tf-torch` keeps 5.1:

```
conda create -n satquery --clone tf-torch
```

Then `make PY=~/miniconda3/envs/satquery/bin/python test`.

Then load the EarthDial-4B backbone in `src/satquery/models/vlm/backbone.py`.

> ⚠️ The HuggingFace repo IDs in the configs are **unverified** — they came from
> the architecture and nobody has confirmed them against the Hub. Check them before you trust a
> download. If EarthDial misbehaves, the documented fallback order is GeoChat, then
> Qwen2.5-VL-7B. Do not restructure the repo around a fallback until the primary has
> actually failed an inference test.

### 5. Train the LoRA adapter

```
make train-lora
```

This is the artifact that proves *"RS adaptation of a vision or VL component"* to the
judges. Without it we fail the problem statement outright — a generic VLM with no remote
sensing adaptation does not satisfy SIH26167.

### 6. Train the change head, then score everything

```
make train-change
make eval
```

`make eval` already runs today and honestly reports every score as `TBD`. As each piece
lands, cells fill in.

---

## Three rules that will bite you

**1. Never copy a number out of `preprocess/constants.py`.**
Those values must be byte-identical when training and when evaluating. If they drift, the
model quietly gets worse and it is close to impossible to debug. `constants_fingerprint()`
hashes the whole set; it is logged at training time and checked at eval time so drift
fails loudly instead of silently.

**2. Never invent a benchmark number.**
If an evaluation has not run, the cell says `TBD`. A fabricated score in a README is worse
than no README.

**3. Never make a stub return something plausible.**
Raise `NotImplementedError` and name what is missing.

---

## The hard problem, in one paragraph

We train on Sentinel imagery at **10 metres per pixel**. ISRO will evaluate on Cartosat-2S
at **under 1 metre** and RISAT radar. That is more than a 20× resolution gap, plus
different colour bands and different radar calibration. Three defences: every prompt is
tagged with its resolution (`<gsd:10.0m>`), training images are randomly rescaled across
that whole range, and the deterministic spectral indices give a defensible answer even
when the learned model is out of its depth. Check every design decision against *"does
this survive a 20× resolution change?"*

---

## Results

Filled in only by a committed run of `make eval`.

| Requirement | Benchmark | Metric | Score |
|---|---|---|---|
| RS adaptation | before/after on the mix | delta | TBD |
| Single-image VQA (mandatory) | RSVQA | exact match | TBD |
| Single-image VQA (mandatory) | VRSBench | LLM-judge acc | TBD |
| Captioning | VRSBench | BLEU-4 / CIDEr | TBD |
| Grounding | VRSBench | acc@0.5 | TBD |
| Bi-temporal change (mandatory) | CDVQA test-1 | accuracy | TBD |
| Bi-temporal change (mandatory) | CDVQA test-2 | accuracy | TBD |
| Optical + SAR fusion | BigEarthNet.txt | built-up / water IoU | TBD |

---

## What this repo will never contain

Do not add these here. They belong to the backend and frontend teams:

- HTTP server code (FastAPI, Flask, Django)
- Frontend assets (React, HTML, CSS)
- Job queues (Celery, Redis)
- Database models, migrations, ORM code
- Auth, sessions, user management
- Multi-service Docker Compose topology

A single-service Dockerfile for the training environment is fine. If a task looks like it
needs one of the above, the answer is almost always to widen the tool contract in
`serve/contracts.py` instead.
