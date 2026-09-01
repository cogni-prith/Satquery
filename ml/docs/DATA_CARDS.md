# Data cards

One entry per dataset: what it is, what it is used for, and what it cannot support.
The last column matters most — it is what stops a benchmark being cited for a claim it
does not license.

## RSVQA-LR
Sentinel-2 patches with template-generated questions (presence, count, comparison, rural
or urban). Used for single-image VQA. **Limit:** the questions are generated from a fixed
grammar, so a high score measures fit to that grammar as much as scene understanding.

## VRSBench
High-resolution aerial imagery with human-verified captions, VQA pairs and referring
expressions. Used for VQA, captioning and grounding. **Limit:** sub-metre aerial optical
only — it says nothing about Sentinel-scale or SAR performance.

## CDVQA
Bi-temporal question answering over SECOND imagery. Used for change VQA. **Limit:** the
answer set is small and closed, so it is scored by exact match and routed through the
symbolic layer rather than a generative model — a generated free-text answer would be
graded on string luck.

## SECOND
Semantic change detection: two dates, per-pixel class labels at each. The imagery behind
CDVQA. Used to train `SemanticChangeNet`. **Limit:** small (~4,600 pairs) and urban-biased.

## BigEarthNet-V2 (reBEN)
Paired Sentinel-1 SAR and Sentinel-2 optical patches with CORINE land-cover labels, read
from an LMDB built by `rico-hdl`. Used for optical + SAR fusion and land-cover
segmentation. **Limit:** labels are patch-level multi-label, not per-pixel — a per-pixel
IoU against them is an approximation and should be reported as one.

## Reading order on a spinning disk
The LMDB is keyed by patch. Sequential reads run ~1,000 keys/s; random reads over a large
key space fall to roughly one patch per second on an external HDD. Sample patches, then
take all their bands, rather than sampling rows.
