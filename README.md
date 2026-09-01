# SatQuery AI

An interactive vision-language assistant for multimodal remote sensing image analysis
through text queries. Smart India Hackathon 2026, ISRO problem statement **SIH26167**.

Ask a question about satellite imagery in English and get an answer that is **measured**,
not generated — with the tool that produced every number named beside it.

## Layout

| directory | what it is | branch with its own history |
|---|---|---|
| `ml/` | the ML layer: tools, symbolic measurement, training and eval | `ml` |
| `backend/` | FastAPI service, router, single-GPU worker | `backend` |
| `frontend/` | React dashboard: routing diagram, evidence maps, execution trace | `frontend` |
| `v1/` | the earlier VLM-first build, kept for its trained heads and eval harness | `v1` |

Each directory was a standalone repository and is vendored here with `git subtree`, so
`main` is browsable while every project keeps its full commit history on its own branch.

## The idea

Perception is discriminative and answers are **computed**. A language model is used at the
two ends only — parsing the query, and wording the result — and it never sees the image.
That is enforced by a signature, not a prompt: `verbalize(record)` takes an `AnswerRecord`
and nothing else, and a test asserts the signature has not widened.

Every value carries the tool that produced it. A `Fact` with a blank provenance is refused
at construction.

Confidence is **agreement between two independent estimates** — a learned mask against a
closed-form spectral index — never a softmax maximum or a token logprob. Where no second
signal exists, no confidence is reported rather than a number that means nothing.

## Measured results

Reproduced by `make eval` on freshly downloaded data. A benchmark that has not run reports
TBD; nothing here is an estimate.

| requirement | benchmark | metric | n | score |
|---|---|---|--:|--:|
| Single-image VQA (mandatory) | RSVQA | exact match | 200 | 0.655 |
| Single-image VQA (mandatory) | VRSBench | exact match | 200 | 0.620 |
| Second single-image task | VRSBench | acc@0.5 | 120 | 0.367 |
| Second single-image task | VRSBench | BLEU-4 / ROUGE-L | 60 | 0.116 / 0.345 |
| Bi-temporal change (mandatory) | CDVQA test-1 | exact match | 500 | 0.678 |
| Bi-temporal change (mandatory) | CDVQA test-2 | exact match | 500 | 0.644 |
| Optical + SAR pair | reBEN validation | IoU built-up / water | 200 | 0.654 / 0.904 |

Land-cover segmentation, scored on the reBEN validation split (4,000 unseen patches):

| class | IoU | | class | IoU |
|---|--:|---|---|--:|
| water | **0.950** | | built-up | **0.602** |
| vegetation | 0.821 | | wetland | 0.367 |
| agriculture | 0.815 | | | |

Object detection, VRSBench holdout at IoU 0.5: recall **0.644**, precision **0.593**.
Read the precision as a floor — VRSBench annotates one object per referring expression, so
a correct detection of a real but unlabelled object counts against the model.

Per class, never a single mean: a mean over five classes here is dominated by forest and
farmland and can look healthy while water and built-up, which every area answer rests on,
are near zero.

## Eleven tools

`seg.landcover` · `indices.deterministic` · `change.radiometric` · `change.mask` ·
`change.vqa_head` · `fusion.extraction` · `detector.openvocab` · `vlm.vqa` · `vlm.caption` ·
`vlm.grounding` · `vlm.change_description`

Routing is a deterministic gate over modality, band names and GSD — no model decides which
tool runs, so "why did it use that?" always has a checkable answer.

## Things it will tell you about itself

These are in the code and in the answers, because a system that hides them is harder to
trust, not easier:

- **NDBI cannot separate concrete from dry bare soil.** On real Alentejo imagery it read
  39.7% built-up on a cell CORINE labels as agro-forestry with no urban class at all, so
  the deterministic path reports built-up as an explicit upper bound.
- **The segmenter learned a designation, not an observation.** CORINE labels a reservoir a
  water body whether or not it holds water. During the 2017 drought CORINE says 55.6%
  water, the segmenter predicts 34.8%, NDWI observes 18.7% — so for "how much water is
  there now", the index is the right instrument and the tool says so.
- **The detector is closed-vocabulary** despite its name: 26 fixed VRSBench classes, and
  the router sends anything outside them to the open-ended VLM instead.
- **An RGB screenshot has no near-infrared**, so no spectral index applies. Rather than
  refuse, a radiometric path reports *where* the scene differs and states plainly that it
  cannot say *what* changed.

## Trained weights

Not in this repository — five trained models totalling ~500 MB, kept outside git. Point
`SATQUERY_ARTIFACT_ROOT` at a directory holding them:

```
$SATQUERY_ARTIFACT_ROOT/
  train/lora_stage1/adapter/          EarthDial-4B LoRA, step 2000
  train/landcover/final/              SegFormer-b0, CORINE Level-1
  train/detector/model.pt             Faster R-CNN, 26 VRSBench classes
  train/change_head/checkpoint-4000/  CDVQA classifier
  checkpoints/fusion/                 optical + SAR dual encoder
```

`serve/tools.py` unregisters any tool whose weights are absent, so a partial set narrows
what the system offers rather than breaking it. Every packing and training script under
`ml/scripts/` rebuilds a model from public data; none of them download anything, they
print the command and stop.

## Running it

```bash
cd ml       && make install && make test && make smoke
cd backend  && uvicorn app.main:app --port 8000
cd frontend && npm install && npm run dev
```

The backend reads `SATQUERY_ARTIFACT_ROOT` and `SATQUERY_DATA_ROOT`; set
`SATQUERY_LOAD_VLM=1` to load the 4 GB vision-language backbone. Without weights present
the service still starts and simply offers fewer tools.
