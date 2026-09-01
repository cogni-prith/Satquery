# Interface contract

What the perception layer hands to the symbolic layer, and what the symbolic layer hands
to the verbalizer. These two boundaries are the whole architecture; everything else is an
implementation detail behind one of them.

## Boundary 1 — perception to symbolic

Every specialist returns arrays and boxes, never prose.

| producer | returns |
|---|---|
| `models.segmentation.LandCoverSegmenter` | `(H, W)` int mask of class ids |
| `models.change.SemanticChangeNet` | two `(H, W)` masks, co-registered |
| `models.detection.OpenVocabDetector` | list of `{label, score, row_min, row_max, col_min, col_max}` |
| `models.clip.DualEncoderCLIP` | image and text embeddings, `(N, D)` |
| `preprocess.indices` | `(H, W)` float index rasters (NDWI, NDVI, NDBI) |

The GSD travels with the imagery, read from the affine transform. When there is no
transform the GSD is `None` and the token is `<gsd:unknown>`. It is never guessed, because
every area in every answer is a pixel count multiplied by it.

## Boundary 2 — symbolic to verbalizer

`verbalize(record: AnswerRecord) -> str`. One parameter. This is the hallucination
firewall, and it is enforced by the signature rather than by a prompt: the verbalizer has
no image parameter, so no instruction can make it look at one. A test asserts the
signature has not widened.

`AnswerRecord` holds only computed values. Every `Fact` carries a `provenance` string
naming the tool that produced it, and a blank provenance is refused at construction. If a
number is in the answer, `validate_provenance()` can name what measured it.

The optional LLM rewrite (`verbalize.llm.rewrite`) takes the same record plus the template
draft, and its output passes `check_numbers`: every number it emits must already appear in
the record or the draft, within a 1% rounding tolerance. Small integers (0–10) pass freely
as ordinary English.

## Confidence

Agreement between two independent estimates of the same quantity — a learned mask and a
deterministic spectral index — as IoU, banded by `thresholds.confidence_band`. Never a
softmax maximum and never a token logprob: those measure how sure the model is, which is
uncorrelated with whether it is right on a sensor it has not seen.
