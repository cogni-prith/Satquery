# Data cards

One card per dataset used by `satquery-ml`. Read this before writing or changing a loader in
`src/satquery/data/datasets/`.

## Ground rules that apply to every dataset on this page

- **Nothing is downloaded automatically.** `scripts/download_datasets.py` prints the acquisition
  instructions for a dataset and verifies checksums of files that are *already present*. It never
  fetches. Some of these archives are hundreds of gigabytes and several sit behind a registration
  or licence click-through that a script must not click on your behalf.
- **Data lives outside the repo.** The dataset tree is rooted at `$SATQUERY_DATA_ROOT` and is
  symlinked into `data/`. Resolve it in code with `satquery.utils.paths.dataset_dir(name)`, never
  with a literal path.
- **Every loader emits `satquery.data.schema.Sample`.** Nothing downstream — mixer, collator,
  model, loss, eval harness — is allowed to know which dataset a record came from. `Sample.source`
  exists for logging, mix accounting and per-dataset eval slicing only.
- **The GSD token is not the loader's job.** A loader fills `ImageRef.gsd_m` from the raster's
  affine transform (via `satquery.preprocess.gsd`); `Sample.prompt()` prefixes the frozen token.
  A loader that formats a token itself is a bug.
- **Counts.** Every headline number below is quoted from the dataset's own documentation as
  recorded in `CLAUDE.md`. Where `CLAUDE.md` does not state a number, the cell reads `TBD` and
  stays `TBD` until someone counts the files on disk. No number here is inferred.
- **Licences.** A licence cell reads `TBD - verify before release` unless it has been read off the
  dataset's own repository or paper by a human. Guessing a licence is not a shortcut, it is a legal
  claim.

---

## BigEarthNet.txt

| Field | Value |
|---|---|
| Role | Primary remote-sensing adaptation set. The LoRA fine-tune that keeps us from failing the problem statement outright runs on this. |
| Canonical URL | https://txt.bigearth.net |
| Paper | arXiv **2603.29630** — see the accuracy note below, this identifier is **UNVERIFIED** |
| Built on | BigEarthNet v2.0 (reBEN), Sentinel-1 + Sentinel-2, 10 European countries |
| Headline counts | 464,044 co-registered S1/S2 pairs; ~9.6M annotations; 15 tasks |
| Image size | TBD |
| GSD | 10 m (Sentinel-2 10 m bands; Sentinel-1 GRD) |
| Modalities | `MULTISPECTRAL` (S2), `SAR` (S1), co-registered pairs |
| Licence | TBD - verify before release |
| Splits we use | Train split for the LoRA adaptation; official validation split for the before/after table. Test split is not used for tuning. |

**Accuracy note, mandatory.** `CLAUDE.md` records the arXiv identifier as `2603.29630`. That is
**unverified and structurally suspicious**: an arXiv YYMM prefix of `2603` places the paper in
March 2026, and the five-digit sequence number is at the top of the plausible range. It may be a
transcription error for a different identifier. **Confirm this against arxiv.org before it appears
in any submission document, slide, or README.** A wrong citation in an ISRO submission is a
credibility hit that costs more than the citation is worth. Until confirmed, cite the project by
its URL (txt.bigearth.net), not by arXiv id.

**Task and answer mapping.** BigEarthNet.txt spans 15 tasks; the ones we consume map as:

| BigEarthNet.txt task family | `TaskType` | `AnswerType` |
|---|---|---|
| Captioning | `CAPTION` | `OPEN_TEXT` |
| Binary VQA | `VQA` | `BINARY` |
| Multiple-choice VQA | `VQA` | `MULTIPLE_CHOICE` (populate `choices`, `answer_text` must be one of them) |
| Referring expression detection | `GROUNDING` | `BOXES` |
| Co-registered S1/S2 pair tasks | `FUSION_EXTRACTION` | per-task; see the loader |

**Loader normalisation.** Map Sentinel band names to the frozen internal order through
`io/modality.py` — never by position. S1 rasters go through `preprocess/sar.py::render_sar`, the
same function RISAT will go through at evaluation time. Referring-expression boxes are converted to
absolute `xyxy` pixels in the source raster's grid, matching `BoundingBox`. Multiple-choice
`choices` are kept in their presentation order because shuffling them changes the prompt.

**Gotchas.**
- S1 and S2 for a patch are co-registered but are *separate files*; pairing them is the loader's
  responsibility and a mis-pair is silent.
- Everything here is 10 m. This dataset is the single largest contributor to the resolution gap
  described in `CLAUDE.md`; aggressive scale resampling on this source is not optional.
- Annotation volume (~9.6M) means the loader must index lazily. Do not materialise the annotation
  set in memory.
- The reBEN patch grid is European. Land-cover priors learned here do not transfer to Indian
  scenes for free.

---

## VRSBench

| Field | Value |
|---|---|
| Role | Single-image train **and** eval: captioning, open-set VQA, and referring-expression grounding. |
| Canonical URL | https://github.com/lx709/VRSBench |
| Headline counts | 29,614 aerial images; 52,472 referring expressions; 123,221 QA pairs; human-verified captions |
| Image size | 512 x 512 |
| GSD | Mixed aerial resolutions, roughly sub-metre; per-image GSD must come from the transform where one exists, otherwise `<gsd:unknown>` |
| Modalities | `OPTICAL_RGB` |
| Licence | TBD - verify before release |
| Splits we use | Official train for fine-tuning; official test for the reported caption, VQA and grounding numbers. |

**Task and answer mapping.**

| VRSBench subset | `TaskType` | `AnswerType` |
|---|---|---|
| Captions | `CAPTION` | `OPEN_TEXT` |
| VQA | `VQA` | `OPEN_TEXT` (open-set, LLM-judged) |
| Referring expressions | `GROUNDING` | `BOXES` |

**Loader normalisation.** Boxes are horizontal (axis-aligned) and are emitted as absolute pixel
`xyxy` on the 512 x 512 grid, which is what `acc@0.5` is computed against. The referring phrase goes
into `BoundingBox.label` and into `Sample.instruction`. Captions and QA go to `answer_text` as
literal strings with no normalisation — VRSBench VQA is open-set and normalising it would corrupt
the judge's input.

**Gotchas.**
- Many VRSBench images are plain RGB with no usable geotransform. `gsd_m` is then `None` and the
  prompt correctly carries `<gsd:unknown>`. Do **not** backfill a guessed GSD from "it's aerial, so
  ~0.3 m" — a wrong token is worse than an absent one.
- VRSBench VQA is open-set and scored by an LLM judge, not exact match. Do not route it through
  `eval/answer_norm.py`.
- Grounding is on **horizontal** boxes. If a rotated-box variant appears in a future release, it is
  a different metric and needs its own row.

---

## RSVQA

| Field | Value |
|---|---|
| Role | Single-image **eval only**: closed-set VQA. This is a mandatory-row proof artifact. |
| Canonical URL | https://rsvqa.sylvainlobry.com — **unverified**, `CLAUDE.md` gives no URL for RSVQA; confirm before it appears in a submission |
| Headline counts | TBD |
| Image size | TBD |
| GSD | TBD (the low-resolution and high-resolution subsets differ; record per subset once verified) |
| Modalities | `OPTICAL_RGB` |
| Licence | TBD - verify before release |
| Splits we use | Official test split(s) only. RSVQA is not in the training mix. |

**Task and answer mapping.** `TaskType.VQA`; `AnswerType.CLOSED_SET` for the counting/area/comparison
families and `AnswerType.BINARY` for presence questions.

**Loader normalisation.** Answers are normalised through `src/satquery/eval/answer_norm.py` on
**both** sides — prediction and ground truth — before exact match. This is the whole ballgame for
RSVQA: the same answer appears as `"yes"`, `"Yes"`, `"yes."` and `"yes, there is"`, and an
un-normalised comparison silently under-reports accuracy by a large margin.

**Gotchas.**
- Numeric-count answers are bucketed into ranges in the original benchmark. Do not compare them as
  free integers.
- Because RSVQA is eval-only, any accidental inclusion in the training mix invalidates the score.
  Loader must hard-refuse `split="train"`.

---

## CDVQA

| Field | Value |
|---|---|
| Role | Bi-temporal change VQA **eval**. Mandatory row in the judging table. |
| Canonical URL | https://github.com/YZHJessica/CDVQA |
| Headline counts | 2,968 image pairs; 122k+ auto-generated QA pairs |
| Image size | 512 x 512 |
| Source imagery | The SECOND public subset |
| GSD | TBD (inherited from SECOND; record once verified) |
| Modalities | `OPTICAL_RGB`, bi-temporal pair |
| Licence | TBD - verify before release |
| Splits we use | Both official test splits: **test-1** and **test-2**. Both are reported separately; averaging them hides a real difference. |

**Closed answer set, mandatory note.** CDVQA answers are closed over exactly six land-cover classes.
Copied verbatim from `CDVQA_ANSWERS` in `src/satquery/preprocess/constants.py`, **in this order**.
NOTE: the real answer set is **19 values**, not six. The six land-cover classes below cover
only 23.4% of answers; `yes`/`no` is 52.2% and change-ratio buckets are 24.4%. The dataset
spells them `NVG_surface`, `low_vegetation` etc., not in prose —
the order is the discriminative head's output-layer ordering and permuting it silently scrambles a
reloaded checkpoint:

1. `non-vegetated ground surface`
2. `buildings`
3. `playgrounds`
4. `water`
5. `low vegetation`
6. `trees`

**Task and answer mapping.** `TaskType.CHANGE_VQA` with `AnswerType.CLOSED_SET` for the scored
answer. The same pair is additionally routed to `TaskType.CHANGE_DESCRIPTION` /
`AnswerType.OPEN_TEXT` for the free-form description, which is reported but is not the CDVQA score.

**Loader normalisation.** Earlier acquisition first in `Sample.images`, always — every paired task in
the schema assumes it. Answers are mapped onto the frozen six-class tuple; an answer string that does
not map is a loader error, never a silent drop.

**Gotchas.**
- `CLAUDE.md` is explicit: **do not route CDVQA through the VLM alone.** A generative model against a
  six-way closed set loses to a Siamese encoder plus a small classification head, and the head runs in
  milliseconds. Route both, report both.
- The QA pairs are auto-generated, so they carry template artefacts. A model can score well by
  learning the templates; that is a real risk for the ISRO hidden set and a reason to weight the
  free-form description in our own judgement even though it is not the scored number.
- Two official test splits, not one. Reporting a single CDVQA number is a reporting error.

---

## LEVIR-CD

| Field | Value |
|---|---|
| Role | **Optional.** Change-mask segmentation head training only. First thing to cut if we fall behind. |
| Canonical URL | https://chenhao.in/LEVIR/ — **unverified**, `CLAUDE.md` gives no URL for LEVIR-CD; confirm before it appears in a submission |
| Headline counts | TBD |
| Image size | TBD |
| GSD | TBD |
| Modalities | `OPTICAL_RGB`, bi-temporal pair |
| Licence | TBD - verify before release |
| Splits we use | Official train/val/test, for the mask head only. Not in the VLM mix. |

**Task and answer mapping.** `TaskType.CHANGE_MASK` with `AnswerType.MASK`; the target is a
single-channel raster path in `Sample.mask_path`.

**Loader normalisation.** Binary building-change masks, earlier image first. The loader emits a path,
not a loaded array — mask decoding belongs to the collator.

**Gotchas.** LEVIR-CD is building-change only. A model trained on it will not describe vegetation or
water change, so it cannot substitute for the CDVQA head. The problem statement marks change *mask*
as optional; do not let it consume time that the mandatory rows need.

---

## SECOND

| Field | Value |
|---|---|
| Role | **Optional.** Semantic change-mask training, and the source imagery behind CDVQA. |
| Canonical URL | https://captain-whu.github.io/SCD/ — **unverified**, `CLAUDE.md` gives no URL for SECOND; confirm before it appears in a submission |
| Headline counts | TBD |
| Image size | 512 x 512 |
| GSD | TBD |
| Modalities | `OPTICAL_RGB`, bi-temporal pair |
| Licence | TBD - verify before release |
| Splits we use | Official train split for the mask head only. |

**Task and answer mapping.** `TaskType.CHANGE_MASK`, `AnswerType.MASK`.

**Gotchas — read this one carefully.** CDVQA is built on the SECOND public subset. Training the change
head on SECOND and then evaluating on CDVQA risks **evaluating on imagery seen in training**. Any use
of SECOND for training must exclude every pair that appears in a CDVQA test split, and the exclusion
must be by image identity, not by filename convention. If the exclusion cannot be established, do not
train on SECOND at all; the contaminated number is worthless.

---

## The default mix

`configs/data/mix_v1.yaml` defines the training mix:

| Source | Share |
|---|---|
| BigEarthNet.txt | 40% |
| VRSBench | 40% |
| CDVQA | 20% |

with aggressive random resampling across scales on every source. The resampling is the first of the
three mitigations for the resolution gap (10 m Sentinel training vs sub-metre Cartosat-2S and RISAT
evaluation); see `docs/EVAL_PROTOCOL.md` for the other two. Change the mix in the YAML, never in a
loader.

---

## Loader status

Every dataset loader is currently an honest stub: it raises `NotImplementedError` naming exactly what
is missing. No loader returns a synthetic `Sample`. A loader that fabricated records would pass a
smoke test today and destroy an eval run in three weeks. Module paths below are the planned homes
under `src/satquery/data/datasets/`; `make registry` and the test suite are the authority on what
exists right now.

| Loader | Status | Exact missing input |
|---|---|---|
| `data/datasets/bigearthnet_txt.py` | stub | BigEarthNet.txt archive + annotation index under `$SATQUERY_DATA_ROOT/bigearthnet_txt`; S1/S2 patch pairing manifest |
| `data/datasets/vrsbench.py` | stub | VRSBench images and the caption / VQA / referring-expression JSON under `$SATQUERY_DATA_ROOT/vrsbench` |
| `data/datasets/rsvqa.py` | stub | RSVQA test images and question/answer JSON under `$SATQUERY_DATA_ROOT/rsvqa` |
| `data/datasets/cdvqa.py` | stub | CDVQA bi-temporal pairs and the test-1 / test-2 QA JSON under `$SATQUERY_DATA_ROOT/cdvqa` |
| `data/datasets/levir_cd.py` | stub | LEVIR-CD image pairs and binary masks under `$SATQUERY_DATA_ROOT/levir_cd` |
| `data/datasets/second.py` | stub | SECOND image pairs and semantic change masks under `$SATQUERY_DATA_ROOT/second`, **plus** the CDVQA-test exclusion list |

Get any of them by running `python scripts/download_datasets.py --dataset <name>`, following the
instructions it prints, and re-running it to verify checksums.

A loader is done when it emits the unified schema, has `__len__` and `__getitem__` smoke tests, and
has its row on this page filled in with real counts.
