"""Stage one of the router: the deterministic gate.

Parsed raster metadata narrows the candidate tool set. This is a lookup, not a model, and
it always succeeds -- `CLAUDE.md` is explicit that the router is two staged steps rather
than a free-form ReAct loop, and that the registry specs must be rich enough for the gate
to be a lookup table rather than a guess.

The rules, from the problem statement:
  one image                        -> vqa, caption, grounding
  two images, different timestamps -> change
  two images, different sensors    -> fusion
"""

from __future__ import annotations

from satquery.serve.contracts import ImageRef, InputConfig, Modality, TaskType, ToolSpec


def classify_input(images: list[ImageRef]) -> InputConfig:
    """Decide the input configuration from the images alone.

    Sensor difference is checked before timestamp difference. A Sentinel-1 and Sentinel-2
    pair of the same scene usually differs in BOTH -- they are separate acquisitions -- so
    testing timestamps first would classify every fusion pair as bi-temporal change and
    silently route it to the wrong tool.

    Raises:
        ValueError: Not one or two images. The contract permits no other count.
    """
    if len(images) == 1:
        return InputConfig.SINGLE
    if len(images) != 2:
        raise ValueError(f"expected one or two images, got {len(images)}")

    first, second = images
    sar = {first.modality is Modality.SAR, second.modality is Modality.SAR}
    if len(sar) == 2:
        return InputConfig.CROSS_MODAL_PAIR

    if first.timestamp and second.timestamp and first.timestamp != second.timestamp:
        return InputConfig.BI_TEMPORAL_PAIR

    # Two same-sensor images with no usable timestamps. Bi-temporal is the honest reading:
    # a pair of one modality is a before/after by construction, and the tools that serve it
    # report the missing timestamp as a warning rather than inventing an ordering.
    return InputConfig.BI_TEMPORAL_PAIR


def gate(images: list[ImageRef], task: TaskType | None = None) -> tuple[InputConfig, list[ToolSpec]]:
    """Return the input configuration and every tool that can legally serve it.

    Only implemented tools are returned. A stub would fail honestly if called, but offering
    it as a candidate invites the selector to pick it, which turns a capability gap into a
    user-visible error for no reason.
    """
    from satquery.models.registry import REGISTRY

    config = classify_input(images)
    modalities = [image.modality for image in images]
    gsd = max((i.gsd_m for i in images if i.gsd_m is not None), default=None)

    candidates = REGISTRY.candidates(config, modalities, gsd, task)
    return config, [spec for spec in candidates if spec.implemented]
