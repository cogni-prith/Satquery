# satquery-ml v2

The model layer of **SatQuery AI**, our entry for Smart India Hackathon 2026, ISRO
problem statement SIH26167.

**This is a rewrite of v1, not an increment.** v1 was VLM-first: EarthDial-4B generated
answers, and a LoRA adapter was the centrepiece. v2 is specialist-first. Perception is
discriminative, answers are computed, and a language model runs in exactly two places —
parsing the query and wording the result.

## Why the pivot

The representative queries are quantitative. *"Has the built-up area increased,
decreased, or remained unchanged?"* has an exact answer: segment built-up at T1 and T2,
count pixels, multiply by GSD squared, compare against a calibrated threshold. A
generative model guessing that is a defect, not a design.

It also answers the project's central risk directly. Training data is Sentinel at 10 m;
the hidden evaluation set is Cartosat-2S and RISAT at sub-metre to 2 m. A segmentation
model degrading on an unseen sensor produces a visibly worse mask. A generative model
degrading produces a confident wrong sentence, which is invisibly worse.

## The flow

    query + images
      -> intent parser (small LLM, constrained JSON)
      -> perception bank (segmentation, detection, change, dual encoder)
      -> symbolic answer layer (counts, areas, thresholds, spatial predicates)
      -> verbalizer (templates first, LLM rewrite optional)
      -> answer + evidence + trace

`symbolic/` is where the answer is decided. Pure Python, CPU-testable, no training
dependency — the highest-value code here, and implemented first for that reason.

## The hallucination firewall

`verbalize/templates.verbalize()` takes an `AnswerRecord` and **nothing else**. No image,
no path, no feature map, no tool handle. That absence is enforced by the signature and
asserted in `tests/test_verbalize.py`, because a prompt asking a model not to hallucinate
is a request while an absent parameter is a guarantee.

Every `Fact` in an `AnswerRecord` carries a `provenance` string naming the tool or
computation that produced it. `validate_provenance()` refuses a record where any is
blank. This is what makes "evidence-grounded" true rather than aspirational.

## What was carried over from v1

Unchanged and already tested: all of `preprocess/` (frozen constants, GSD token, the
five-step SAR pipeline, optical, spectral indices), all of `io/`, the fusion dual
encoder, the dataset loaders, and the registry. The pivot changed how answers are
*decided*, not how rasters are *read*.

## Status

    symbolic/     implemented, 24 tests
    verbalize/    templates implemented, LLM rewrite an honest stub
    preprocess/   carried from v1, unchanged
    io/           carried from v1, unchanged
    models/       clip, segmentation, change, detection, captioner all to build

168 tests pass on CPU in under a second.

## Confidence

Agreement between independent signals, never a model score. For any claim about water,
built-up or vegetation there are two paths: the learned segmentation and the
deterministic spectral index. Confidence is their IoU plus a categorical band. A softmax
maximum reported as confidence would make the trace lie about what the system knows.
