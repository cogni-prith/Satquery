"""Bind implemented tools to their specs. Imported for its side effects.

`ToolRegistry` loads this module the first time a tool instance is requested, so a spec
without a binding here stays a spec: asking for it raises `NotImplementedError` naming
what is missing rather than returning something that answers.

Three tools, when their weights exist: the trained segmenter, the spectral index path
and the radiometric fallback. Every learned tool needs weights that do not
exist yet, and binding a stub would make the router offer a tool that cannot answer.
"""

from __future__ import annotations

from satquery.models.indices.deterministic import DeterministicIndexTool
from satquery.models.indices.radiometric import RadiometricChangeTool
from satquery.models.registry import REGISTRY

REGISTRY.bind(
    "indices.deterministic",
    lambda: DeterministicIndexTool(REGISTRY.get_spec("indices.deterministic")),
)
REGISTRY.bind(
    "change.radiometric",
    lambda: RadiometricChangeTool(REGISTRY.get_spec("change.radiometric")),
)


def _bind_landcover() -> None:
    """Bind the trained segmenter, but only if its weights are actually on disk.

    Unregistering the spec when the weights are absent is deliberate. Leaving it bound
    would let the gate offer a tool that raises on every call, and leaving it registered
    as implemented would make the capability list claim something the deployment cannot
    do. A missing model should narrow what the system offers, not break what it promises.
    """
    from satquery.models.segmentation.landcover import LandCoverSegmenter
    from satquery.models.segmentation.tool import LandCoverTool
    from satquery.utils.paths import artifact_root

    weights = artifact_root() / "train" / "landcover" / "final"
    if not weights.is_dir():
        REGISTRY.unregister("seg.landcover")
        return

    segmenter = LandCoverSegmenter()
    try:
        segmenter.load(weights)
    except Exception as exc:  # a broken checkpoint must not take the service down
        from satquery.utils.logging import get_logger

        get_logger(__name__).warning("segmenter at %s did not load: %s", weights, exc)
        REGISTRY.unregister("seg.landcover")
        return

    REGISTRY.bind(
        "seg.landcover",
        lambda: LandCoverTool(REGISTRY.get_spec("seg.landcover"), segmenter),
    )


_bind_landcover()
