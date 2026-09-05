"""Stage one of the router: the deterministic gate.

Parsed raster metadata narrows the candidate tool set. This is a lookup, not a model, and
it always succeeds -- `the architecture` is explicit that the router is two staged steps rather
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

    candidates = [
        spec for spec in REGISTRY.candidates(config, modalities, gsd, task) if spec.implemented
    ]
    # Highest preference first. Where a learned tool and its deterministic fallback are
    # both legal, the choice must be stated on the spec rather than fall out of whichever
    # name happens to sort first. `preference` exists only on the v2 ToolSpec, and this
    # service is meant to run against either package, so its absence means "no stated
    # preference" rather than a crash.
    candidates.sort(key=lambda spec: getattr(spec, "preference", 0), reverse=True)
    return config, candidates


def explain_refusal(images: list[ImageRef], task: TaskType | None = None) -> str:
    """Say why no tool matched, in terms of the image the user actually uploaded.

    "No implemented tool can serve this" is true and useless. The gate knows exactly which
    constraint each tool failed, and the most common refusal by far -- an RGB screenshot,
    which carries no near-infrared band and so supports no spectral index -- has a
    one-sentence explanation the user can act on.
    """
    from satquery.models.registry import REGISTRY

    config = classify_input(images)
    modalities = {image.modality for image in images}

    if modalities == {Modality.OPTICAL_RGB}:
        return (
            "This is a three-band RGB image, which carries no near-infrared band. Every "
            "index this build can compute -- NDWI, NDBI, NDVI -- needs one, so there is "
            "nothing here to measure. Upload a multispectral raster (Sentinel-2, Landsat) "
            "or a SAR scene. The learned models that would read an RGB screenshot are not "
            "trained yet."
        )

    implemented = [spec for spec in REGISTRY.list_specs(implemented_only=True)]
    if not implemented:
        return "No tool in this build is implemented yet, so nothing can serve any input."

    names = ", ".join(spec.name for spec in implemented)
    return (
        f"No implemented tool accepts {config.value} input with modalities "
        f"{sorted(m.value for m in modalities)}"
        + (f" for task {task.value}" if task else "")
        + f". Implemented in this build: {names}."
    )
