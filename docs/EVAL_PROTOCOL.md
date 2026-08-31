# Evaluation protocol

How `satquery-ml` is scored, by whom, and under what constraints. One command runs everything:

```
make eval
```

If a change breaks that command, the change is not done.

---

## What the judges score

Every mandatory row needs a matching artifact on disk. A row without an artifact is an unproven claim.

| Requirement | Component | Proof artifact |
|---|---|---|
| RS adaptation of a vision or VL component | LoRA on BigEarthNet.txt | training log, adapter weights, before/after table |
| Single-image VQA (**mandatory**) | `models/vlm` | RSVQA and VRSBench VQA scores |
| One more single-image task | captioning **and** grounding, we do both | VRSBench caption metrics, grounding acc@0.5 |
| Bi-temporal change (**mandatory**) | `models/change` | CDVQA test-1 and test-2 accuracy |
| Optical and SAR pair analysis | `models/fusion` | joint built-up and water extraction |
| Agentic orchestration | `models/registry.py` | execution trace per call |
| Auditable execution summary | `serve/contracts.py` `Trace` | trace serialised on every call |

A generic VLM with no remote-sensing adaptation fails the problem statement outright. The fine-tune is
not optional polish, and the before/after table is the evidence that it happened.

---

## How each metric is computed

### RSVQA — closed-set VQA, exact match after normalisation

Prediction and ground truth both pass through `src/satquery/eval/answer_norm.py` before comparison;
the score is the fraction of exact string matches over the normalised pair. Normalising only one side
is a silent bug that under-reports accuracy.

Decoding is constrained to the closed answer set via the `answer_set` parameter on the `vlm.vqa` tool
spec, so the model cannot spend its accuracy budget on formatting. Presence questions are binary;
count questions are compared against the benchmark's own answer buckets, never as free integers.
Reported per question family and as an overall figure.

### VRSBench VQA — open-set, LLM judge

VRSBench VQA answers are free text and are **not** exact-matched and **not** normalised through
`answer_norm.py`. Each `(question, ground truth, prediction)` triple goes to an LLM judge that returns
a binary correct/incorrect; the score is the fraction judged correct. The judge model, its prompt and
its decoding parameters are pinned in `configs/eval/full_suite.yaml`, because changing any of the
three changes the number and makes two runs incomparable. Judge verdicts are written to the run
directory alongside the score so a disputed item can be re-read.

### VRSBench captioning — BLEU-4, METEOR, ROUGE-L, CIDEr

Standard captioning metrics against the human-verified references, all four reported; no single one of
them is trustworthy alone. Tokenisation is fixed in the eval config. CIDEr's IDF statistics are
computed over the evaluation reference set being scored, so a change to `max_samples` changes CIDEr —
record `max_samples` next to every CIDEr number.

### VRSBench grounding — acc@0.5 on horizontal boxes

Predicted and reference boxes are axis-aligned `xyxy` in absolute pixels of the source raster grid,
exactly as `BoundingBox` in `serve/contracts.py` defines them. A prediction counts as correct when its
IoU with the reference box is at or above 0.5. Headline metric is acc@0.5; acc@0.7 is reported
alongside it as a tightness check. Boxes are never normalised to `[0, 1]` and never converted to
geographic coordinates before scoring — that conversion is the backend's job at display time, not
ours at scoring time.

### CDVQA — accuracy on both official test splits, closed six-class set

The scored answer comes from the discriminative Siamese head (`change.vqa_head`), not from the
generative VLM. Accuracy is exact match over the frozen **19-value** answer set in
`CDVQA_ANSWERS` (VERIFIED against the published annotations; CLAUDE.md's "six classes"
are only the land-cover subset, 23.4% of answers):

`non-vegetated ground surface`, `buildings`, `playgrounds`, `water`, `low vegetation`, `trees`

**test-1 and test-2 are reported separately.** They are two official splits with different
characteristics and averaging them into one number hides a real difference. Per-class accuracy is
reported alongside overall accuracy, because a closed six-class set with an unbalanced prior lets a
degenerate model look respectable on the aggregate.

The free-form change description from `vlm.change_description` is generated for the same pairs and
reported as a qualitative artifact. It is not the CDVQA score.

### Fusion — joint built-up and water extraction

An optical/SAR co-registered pair goes to `fusion.extraction`, which produces built-up and water
masks. Each mask is scored against the deterministic index counterpart from `indices.deterministic`:
NDBI for built-up, NDWI (and the SAR VV backscatter threshold) for water, all with the frozen
thresholds in `preprocess/constants.py`. Reported as IoU between the learned mask and the index mask,
per target, plus the mask and overlay paths written under the artifact root so a judge can look at the
map rather than take the number on trust.

That agreement figure is also what populates `ToolResult.confidence` for this tool. It is deliberately
an agreement measure and not a softmax: on unseen sensor data a softmax is confidently wrong, whereas
disagreement between a learned output and a closed-form index is genuine information.

---

## The ten-minute budget

The full suite must not exceed **ten minutes on a single GPU**. This is a hard constraint, not a
target: an eval nobody can afford to run is an eval nobody runs, and then the scores table goes stale
during the week that matters.

`configs/eval/full_suite.yaml` enforces it with a per-task `max_samples` cap. Each task declares its
own cap; the harness subsamples deterministically (seeded through `satquery.utils.seed`) so two runs
at the same cap score the same items. Every number in the results table is therefore a number *at a
stated `max_samples`*, and the cap is recorded next to the score. A score at a different cap is a
different number — do not compare them.

Raise a cap only by editing the YAML, never by passing a flag in a one-off command that nobody else
will reproduce. If the suite overruns, cut sample counts before cutting tasks: a mandatory row with
few samples is still evidence; a missing mandatory row is a failed requirement.

---

## The anti-drift rule

`constants_fingerprint()` in `src/satquery/preprocess/constants.py` is a SHA-256 over every frozen
preprocessing constant — GSD token format, speckle filter window and look count, dB conversion, stretch
percentiles, SAR pseudo-RGB layout, band order, index epsilons and thresholds, pan-sharpening choice,
pair-validation limits, and the CDVQA class tuple.

**The rule:**

1. Every training run logs the fingerprint into its run directory (`ConstantsFingerprintCallback` in
   `src/satquery/train/callbacks.py` does this).
2. The eval harness recomputes the fingerprint in its own process and asserts it matches the one
   recorded in the checkpoint being scored.
3. A mismatch is a hard failure, not a warning. The scores from that run are not comparable to any
   other run and must not go in the table.

**Why this matters more than it looks like it does.** The failure it catches is silent. If the stretch
percentiles change from 2–98 to 1–99 between training and evaluation, nothing crashes, no shape
mismatches, no exception is raised — every image is simply slightly different from what the model
learned on, and accuracy drops a few points for reasons no one can find. Multiply that by a SAR
speckle window, a swapped band order, or a GSD token formatted with a different precision, and you get
a week of debugging a model that was never broken. The fingerprint turns an invisible accuracy leak
into a loud failure at the top of the run.

Corollary: never inline a constant, never add a train-only or eval-only branch, and never add a
sensor-specific override. Any of those defeats the fingerprint by making it agree while the behaviour
differs.

---

## Bi-temporal change (MANDATORY): CDVQA test-1 and test-2

**Proof artifact for the mandatory "bi-temporal change" judging row.** Scored on the FULL
official test splits, with code independent of the training loop's own metric.

**Provenance.** Two-tower head: weight-shared ResNet-18 over both dates giving
`[f_t1, f_t2, |f_t1 - f_t2|]`, fused with a GRU question encoder (46-word vocabulary,
built from the training split only). 12.12M parameters, 0.73 GB VRAM. Trained on 65,967
CDVQA training questions over 1,600 SECOND image pairs; `artifacts/train/change_head/
checkpoint-3000`, selected as the val-accuracy peak. Val trend across checkpoints:
0.4595 -> 0.5190 -> 0.6775 -> 0.6775 -> 0.6815 -> **0.6900** -> 0.6765 -> 0.6690, i.e. it
began overfitting after step 3000 and training was stopped there.

| Split | n | Majority | Per-type majority | **Head** | Gain over per-type |
|---|---|---|---|---|---|
| test_1 | 39,686 | 0.312 | 0.508 | **0.6784** | +0.170 |
| test_2 | 31,036 | 0.179 | 0.456 | **0.6320** | +0.176 |

Scored separately and never averaged: they are two different question sets over the same
968 image pairs.

Accuracy by question type (test_1 / test_2):

| Type | test_1 | test_2 |
|---|---|---|
| change_or_not | 0.833 | 0.831 |
| decrease_or_not | 0.731 | 0.733 |
| increase_or_not | 0.717 | 0.712 |
| change_ratio_types | 0.714 | 0.712 |
| change_to_what | 0.591 | 0.591 |
| largest_change | 0.439 | 0.438 |
| change_ratio | 0.355 | 0.358 |
| **smallest_change** | **0.268** | **0.268** |
| *land-cover answers only* | *0.434* | *0.434* |

### How to read this

- **The baseline matters.** A model that answers the most common answer for each question
  type scores 0.508 on test_1. The raw accuracy is only meaningful against that.
- **The failure modes are coherent, not random.** Detecting *whether* something changed is
  strong (0.83). Ranking *which* class changed least is near-chance (0.27). Those are
  genuinely different difficulties: the second requires quantifying and ordering change
  across all six land-cover classes from a single forward pass.
- **The six-class framing would have cost most of this.** The land-cover subset alone
  scores 0.434 on 8,799 of 39,686 questions. A six-way head built to CLAUDE.md's spec
  could not have answered the other 76.6% at all.
- **test_1 and test_2 differ mainly by question mix**, not by difficulty: per-type
  accuracies are nearly identical, and the 4.6-point gap in the headline is explained by
  test_2 having proportionally more `change_ratio_types` and fewer `change_or_not`.

## Checkpoint sweep: does more training help?

All five models scored on the SAME 150 VRSBench validation samples, through one corrected
code path, greedy decoding. Reproduce with
`$SATQUERY_ARTIFACT_ROOT/baselines/eval_adapter.py [--adapter <checkpoint>]`.

| Model | Grounding acc@0.5 | acc@0.7 | VQA exact match |
|---|---|---|---|
| base, no adapter | 0.1467 | 0.0467 | 0.3667 |
| checkpoint-500 | 0.3333 | 0.1133 | 0.5933 |
| checkpoint-1000 | 0.3267 | 0.1200 | 0.5733 |
| checkpoint-1500 | **0.3533** | 0.1200 | 0.5933 |
| checkpoint-2000 | 0.3200 | 0.1200 | **0.6000** |

### The finding: adaptation saturates by step 500

The gap from base to *any* adapter is large and unambiguous -- grounding roughly doubles,
VQA rises about 60%. The gap *between* adapters is not.

Spread across all four adapted checkpoints is 0.033 (grounding) and 0.027 (VQA). The 95%
confidence interval at n = 150 is about **+/-0.075**, i.e. roughly 11 samples. Every
difference between checkpoints is several times smaller than the noise floor, so this
sweep cannot distinguish them, and the apparent "best" cell in each column lands on a
different checkpoint -- 1500 for grounding, 2000 for VQA -- which is what noise looks like.

**Practical consequence.** Steps 500 to 2000 cost 1h36m of GPU time and bought nothing
measurable. The useful run is ~1h37m, not 3h13m. Future experiments should train to ~500
steps, measure, and only extend if a larger evaluation justifies it.

**Confound worth stating.** checkpoint-500 trained on VRSBench alone; checkpoints 1000 and
later trained on VRSBench *plus* CDVQA, because the CDVQA loader landed mid-run. So "more
steps did not help" is entangled with "the broader mix did not help either". Separating
them needs a controlled run, which has not been done.

**What would settle it.** n = 150 is too small to rank these. Scoring the full 16,159-record
referring split would shrink the interval to about +/-0.007 and could genuinely separate a
0.02 difference. That is a ~3 hour eval and has not been run.

### A bug this sweep caught

The first attempt reported checkpoint-1000 at exactly `0.1467 / 0.0467 / 0.3667` -- the base
model's numbers to four decimal places. The model config names an adapter so that
`make eval` picks it up, so `load()` attached it and `attach_adapter()` then wrapped a
*second* PeftModel around the checkpoint under test. Nested PeftModels do not compose: both
go inert and the model behaves as the unadapted base.

Left unnoticed, the table would have read "the fine-tune achieved nothing at any
checkpoint" -- plausible, since the loss curve had gone flat, and entirely wrong.
`EarthDialBackbone.attach_adapter` now refuses to nest adapters, and the eval script
asserts a non-zero LoRA tensor count after attaching before it will report a number.

## RS adaptation: before and after (LoRA step 500)

**This is the proof artifact for the mandatory "RS adaptation of a vision or VL component"
judging row.** Both columns are measured runs on the SAME 150 VRSBench validation samples,
same prompts, same metrics, same decoding. Reproduce with
`$SATQUERY_ARTIFACT_ROOT/baselines/eval_adapter.py [--adapter <checkpoint>]`.

**Provenance.** `akshaydudhane/EarthDial_4B_RGB`, NF4 4-bit, greedy. Adapter:
`artifacts/train/lora_stage1/checkpoint-500`, LoRA r=16 alpha=32 on the LM attention and
MLP projections (8.91M trainable, 0.41% of the model), trained 500 optimizer steps at
effective batch 16 -- roughly 8,000 samples, about 5.6% of one VRSBench epoch. Final
training loss 0.7965. RTX 4060 8 GB, 1h37m.

| Metric | Base | Adapted (step 500) | Change |
|---|---|---|---|
| Grounding acc@0.5 | 0.1467 | **0.3333** | **2.3x** |
| Grounding acc@0.7 | 0.0467 | **0.1133** | **2.4x** |
| Grounding boxes parsed | 150/150 | 150/150 | unchanged |
| VQA exact match | 0.3667 | **0.5933** | **1.6x** |

VQA by question type:

| Type | Base | Adapted | | Type | Base | Adapted |
|---|---|---|---|---|---|---|
| image | 1.000 | 1.000 | | object quantity | 0.250 | **0.500** |
| object existence | 0.923 | 0.923 | | object position | 0.194 | **0.361** |
| object color | 0.643 | **0.786** | | object shape | 0.333 | 0.333 |
| object category | 0.111 | **0.667** | | object size | 0.000 | **0.667** |
| reasoning | 0.500 | 0.500 | | rural or urban | 0.000 | **0.500** |
| scene type | 0.000 | **0.667** | | object direction | 0.000 | 0.000 |

### How to read this honestly

The gains are real but they are not all the same kind of gain.

- **Grounding is genuine capability.** Localising a box more accurately cannot be faked by
  learning vocabulary, and the parse rate was already 150/150 before adaptation, so the
  improvement is not a format artifact. 2.3x at acc@0.5 is the strongest single result here.
- **Part of the VQA gain is vocabulary alignment, not new perception.** The three
  categories that went 0.000 -> 0.5-0.67 (`object size`, `rural or urban`, `scene type`)
  are exactly the ones flagged in the baseline as likely phrasing mismatches under
  exact-match scoring. The model was probably not blind to scene type before; it phrased
  its answer differently from the reference. Learning the reference's phrasing is a real
  gain **for this benchmark**, and it is what a judge will score, but it is a weaker claim
  than improved understanding.
- **`object direction` did not move at all (0.000 -> 0.000).** That is a genuine capability
  gap, not a phrasing problem, and 500 steps did not touch it.
- **n = 150.** Per-type buckets are small; individual cells are indicative, not reliable.
- **Exact match still understates open-set VQA.** VRSBench's official protocol uses an LLM
  judge. Both columns are lower bounds, measured identically, so the *ratio* is meaningful
  even though the absolute values are conservative.

### Not a ceiling

Training was stopped at 500 of 2000 planned steps to measure early. The optimizer state is
on disk, so the run is resumable. Training loss had plateaued around 0.8 since roughly step
60, so further gains from more steps of the same recipe are uncertain -- but the mix was
also incomplete (VRSBench only; BigEarthNet.txt and CDVQA were absent), so this is not the
configuration CLAUDE.md specifies.

## Measured baseline: EarthDial-4B, no adapter

Real numbers from a real run, not estimates. Recorded so the RS-adaptation "before /
after" table the judges want has a genuine "before". Reproduce with the script kept at
`$SATQUERY_ARTIFACT_ROOT/baselines/baseline.py`.

**Provenance.** `akshaydudhane/EarthDial_4B_RGB`, NF4 4-bit, no LoRA adapter, greedy
decoding, VRSBench validation split, **n = 150 per task** (not the full split), run
2026-08-29 on an RTX 4060 8 GB.

| Task | Metric | Base model |
|---|---|---|
| Grounding (referring) | boxes parsed | 150 / 150 |
| Grounding (referring) | acc@0.5 | **0.147** |
| Grounding (referring) | acc@0.7 | **0.047** |
| VQA | exact match | **0.367** |

VQA by question type:

| Type | Acc | Type | Acc |
|---|---|---|---|
| image | 1.000 | object shape | 0.333 |
| object existence | 0.923 | object quantity | 0.250 |
| object color | 0.643 | object position | 0.194 |
| reasoning | 0.500 | object category | 0.111 |
| object direction | 0.000 | object size | 0.000 |
| rural or urban | 0.000 | scene type | 0.000 |

**Three caveats, or these numbers will be misread.**

1. **Exact match understates open-set VQA.** VRSBench scores VQA with an LLM judge; this
   is string equality after normalisation. The four zeros are almost certainly phrasing
   mismatches rather than total failure -- a model answering "urban area" against a
   reference of "urban" scores zero here and would score one under the judge. Treat
   0.367 as a **lower bound**.
2. **n = 150, so per-type buckets are small.** Some types have only a handful of
   samples; those cells are indicative, not reliable. The full split is 37,409 VQA and
   16,159 referring records.
3. **This is the unadapted base model.** It is the number the fine-tune has to beat, and
   the reason CLAUDE.md calls the fine-tune mandatory rather than polish.

The shape of the result is the useful part: the model is strong on existence and
presence (0.92) and near-useless on counting, direction, size and scene type. That is
where the adaptation has to earn its keep.

## Results

> A cell is filled only by a committed run of `make eval`; a fabricated score is worse than no score.

### Single image

| Task | Dataset | Split | Metric | max_samples | Score |
|---|---|---|---|---|---|
| VQA | RSVQA | test | exact match (normalised) | TBD | TBD |
| VQA | VRSBench | test | LLM-judge accuracy | TBD | TBD |
| Captioning | VRSBench | test | BLEU-4 | TBD | TBD |
| Captioning | VRSBench | test | METEOR | TBD | TBD |
| Captioning | VRSBench | test | ROUGE-L | TBD | TBD |
| Captioning | VRSBench | test | CIDEr | TBD | TBD |
| Grounding | VRSBench | test | acc@0.5 | TBD | TBD |
| Grounding | VRSBench | test | acc@0.7 | TBD | TBD |

### Bi-temporal change

| Task | Dataset | Split | Metric | max_samples | Score |
|---|---|---|---|---|---|
| Change VQA | CDVQA | test-1 | accuracy (6-class) | TBD | TBD |
| Change VQA | CDVQA | test-2 | accuracy (6-class) | TBD | TBD |
| Change mask (optional) | LEVIR-CD | test | IoU / F1 | TBD | TBD |

### Optical + SAR fusion

| Target | Source | Metric | max_samples | Score |
|---|---|---|---|---|
| Built-up | BigEarthNet.txt pairs | IoU vs NDBI index mask | TBD | TBD |
| Water | BigEarthNet.txt pairs | IoU vs NDWI / SAR-threshold mask | TBD | TBD |

### RS adaptation, before/after

| Metric | Base EarthDial-4B | + LoRA on BigEarthNet.txt | Delta |
|---|---|---|---|
| RSVQA exact match | TBD | TBD | TBD |
| VRSBench VQA (judge) | TBD | TBD | TBD |
| VRSBench CIDEr | TBD | TBD | TBD |
| VRSBench acc@0.5 | TBD | TBD | TBD |

### Run metadata (filled per committed run)

| Field | Value |
|---|---|
| Commit | TBD |
| Constants fingerprint | TBD |
| GPU | TBD |
| Wall clock, full suite | TBD |

---

## The central risk

We train on Sentinel at **10 m** GSD in small tiles. The hidden ISRO evaluation set is **Cartosat-2S**
optical (sub-metre to roughly 2 m) and **RISAT** SAR, pre-georeferenced and co-registered, annotations
undisclosed. That is more than an order of magnitude of resolution change, plus different band centres
and different SAR calibration. Nothing in the results table above measures this gap directly, because
we cannot see the evaluation set. Every design decision must be checked against "does this survive a
20x resolution change".

Three mitigations, and they are deliberately independent so that no single one failing sinks us:

1. **Aggressive scale resampling in the training mix.** Every source is resampled across the canonical
   scale ladder in `CANONICAL_GSD_SCALES_M` (0.5 m to 60 m), so the model sees sub-metre and coarse
   versions of the same content during adaptation rather than meeting sub-metre imagery for the first
   time at evaluation.
2. **The GSD token.** Every instruction is prefixed with `<gsd:{value:.1f}m>` computed from the
   affine transform — never the filename, never the sensor name — so resolution is an explicit input
   the model can condition on instead of a hidden variable it must infer. When GSD genuinely cannot be
   computed the token is `<gsd:unknown>`; a guessed token is worse than an absent one because the
   model has been trained to trust it.
3. **The deterministic index safety net.** NDWI, NDBI, NDVI and the SAR backscatter threshold are
   closed-form. They have no weights, no GPU requirement, and no training distribution to fall outside
   of, so they work identically on Sentinel and on Cartosat-2S. When the learned model is uncertain on
   unseen sensor data, index agreement still produces a defensible answer with a visible map — and the
   agreement between learned output and index output is our confidence signal.
