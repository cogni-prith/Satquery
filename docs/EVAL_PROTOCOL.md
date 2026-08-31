# Eval protocol

## One command

```
make eval
```

Runs every suite in `configs/eval/full_suite.yaml` and writes one report. There is no
second, unofficial path that produces a different number.

## Rules

**A benchmark that has not run reports TBD.** Not 0.0, not an estimate, not a number from
the paper. A fabricated score in a README is worse than an empty cell, because the empty
cell is honest about what is left to do.

**Report n with every score.** At n=150 the 95% confidence interval on an exact-match
score is roughly ±0.075. A 0.03 difference between two checkpoints at that n is not a
difference, and reporting it as an improvement is how a run gets extended by two hours for
nothing.

**A 100% tool-failure rate is a failure, not a 0.0.** The harness raises rather than
recording a score, because a table full of plausible zeros reads like a weak model instead
of a broken wiring.

**Verify the adapter actually attached.** The most dangerous failure mode this project has
seen produced a complete, plausible results table from the *unadapted base model* — one
call in two hundred failed on a moved artifact path and the rest silently ran on base
weights. It was caught by the shape of the numbers matching the base model's, not by
anything in the harness. The backbone loader now discards base weights when an adapter
fails to attach, and the run fails loudly instead.

**Fingerprint the frozen constants.** Every run logs `constants_fingerprint()`. If
preprocessing changed between train and eval, the fingerprints differ and the comparison
is void.

## Metrics

| task | metric |
|---|---|
| VQA, change VQA | exact match after `eval.answer_norm` |
| grounding | accuracy at IoU 0.5 and 0.7 |
| captioning | BLEU-4, ROUGE-L |
| segmentation, fusion | per-class IoU, plus learned-vs-index agreement |

Segmentation IoU is reported per class, never as a single mean. A water IoU of 0.955 on
reBEN is flattered by 11% of patches being over 90% water; the figure on the balanced
subset is 0.828, and that is the defensible one.
