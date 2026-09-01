"""Bind implemented tools to their specs. Imported for its side effects.

`ToolRegistry` loads this module the first time a tool instance is requested, so a spec
without a binding here stays a spec: asking for it raises `NotImplementedError` naming
what is missing rather than returning something that answers.

Binding is conditional on weights actually being present. Where they are not, the spec is
unregistered rather than left bound to something that raises on every call -- a deployment
missing a model should offer less, not promise what it cannot do.
"""

from __future__ import annotations

import contextlib
import os

from satquery.models.indices.deterministic import DeterministicIndexTool
from satquery.models.indices.radiometric import RadiometricChangeTool
from satquery.models.registry import REGISTRY
from satquery.utils.logging import get_logger

_LOG = get_logger(__name__)

#: The four tools that share one EarthDial backbone.
_VLM_TOOLS = ("vlm.vqa", "vlm.caption", "vlm.grounding", "vlm.change_description")


def _drop(*names: str) -> None:
    """Unregister specs whose implementation turned out to be unavailable."""
    for name in names:
        with contextlib.suppress(KeyError):
            REGISTRY.unregister(name)


# -- closed-form tools: no weights, so nothing to be missing -----------------------------

REGISTRY.bind(
    "indices.deterministic",
    lambda: DeterministicIndexTool(REGISTRY.get_spec("indices.deterministic")),
)
REGISTRY.bind(
    "change.radiometric",
    lambda: RadiometricChangeTool(REGISTRY.get_spec("change.radiometric")),
)


def _bind_landcover() -> None:
    """Bind the trained segmenter, but only if its weights are on disk."""
    from satquery.models.segmentation.landcover import LandCoverSegmenter
    from satquery.models.segmentation.tool import LandCoverTool
    from satquery.utils.paths import artifact_root

    weights = artifact_root() / "train" / "landcover" / "final"
    if not weights.is_dir():
        _LOG.info("no segmenter at %s; seg.landcover unregistered", weights)
        _drop("seg.landcover")
        return

    segmenter = LandCoverSegmenter()
    try:
        segmenter.load(weights)
    except Exception as exc:  # a broken checkpoint must not take the service down
        _LOG.warning("segmenter at %s did not load: %s", weights, exc)
        _drop("seg.landcover")
        return

    REGISTRY.bind(
        "seg.landcover",
        lambda: LandCoverTool(REGISTRY.get_spec("seg.landcover"), segmenter),
    )
    _LOG.info("bound seg.landcover")


def _bind_vlm() -> None:
    """Load EarthDial once and bind the four VLM tools onto it.

    One backbone shared by all four. A factory per tool would load a 4 GB model four times
    and exhaust the card on the first request.

    Failure unregisters rather than raises: a machine with no GPU, or a deployment without
    the LoRA adapter, should serve the deterministic tools and report the VLM ones as
    unavailable, not fail at import and take the service down with it.
    """
    try:
        from satquery.models.vlm.backbone import BackboneConfig, EarthDialBackbone
        from satquery.models.vlm.tasks import (
            CaptionTool,
            ChangeDescriptionTool,
            GroundingTool,
            VqaTool,
        )
        from satquery.utils.paths import configs_dir

        config = BackboneConfig.from_yaml(configs_dir() / "model" / "earthdial_4b_rgb.yaml")
        backbone = EarthDialBackbone(config)
        backbone.load()
    except Exception as exc:
        _LOG.warning("VLM tools unavailable (%s); unregistering %s", exc, list(_VLM_TOOLS))
        _drop(*_VLM_TOOLS)
        return

    REGISTRY.bind("vlm.vqa", lambda: VqaTool(backbone=backbone))
    REGISTRY.bind("vlm.caption", lambda: CaptionTool(backbone=backbone))
    REGISTRY.bind("vlm.grounding", lambda: GroundingTool(backbone=backbone))
    REGISTRY.bind("vlm.change_description", lambda: ChangeDescriptionTool(backbone=backbone))
    _LOG.info("bound %d VLM tools on one EarthDial backbone", len(_VLM_TOOLS))


def _bind_change_vqa() -> None:
    """Bind v1's trained CDVQA head, if its checkpoint is on disk.

    A discriminative classifier over the frozen 19-value CDVQA answer set, not a
    generative model. The answer set is closed, so a classifier is both more accurate and
    answers in milliseconds. Measured 0.678 and 0.644 exact match on CDVQA test-1 and
    test-2.
    """
    from satquery.models.change.siamese import ChangeVqaTool
    from satquery.utils.paths import artifact_root

    base = artifact_root() / "train" / "change_head"
    if not any(base.glob("checkpoint-*")):
        _LOG.info("no CDVQA head under %s; change.vqa_head unregistered", base)
        _drop("change.vqa_head")
        return
    REGISTRY.bind("change.vqa_head", ChangeVqaTool)
    _LOG.info("bound change.vqa_head")


def _bind_fusion() -> None:
    """Bind v1's trained optical+SAR dual encoder, if its checkpoint is on disk.

    The other place this system holds two independent estimates of one quantity: a learned
    extraction and the spectral index behind it. Measured on reBEN validation, IoU 0.654
    built-up and 0.904 water, with index agreement 0.803.
    """
    from satquery.models.fusion.dual_encoder import FusionExtractionTool

    try:
        probe = FusionExtractionTool()
        checkpoint = probe.checkpoint_path()
    except Exception as exc:
        _LOG.warning("fusion tool would not construct (%s); unregistered", exc)
        _drop("fusion.extraction")
        return

    if not checkpoint.is_file():
        _LOG.info("no fusion checkpoint at %s; unregistered", checkpoint)
        _drop("fusion.extraction")
        return

    REGISTRY.bind("fusion.extraction", FusionExtractionTool)
    _LOG.info("bound fusion.extraction")


_bind_landcover()
_bind_change_vqa()
_bind_fusion()

# Opt-in. Loading a 4 GB backbone is not something an import should do to a test run or to
# a CLI that only wanted to print the registry; the serving process asks for it by name.
if os.environ.get("SATQUERY_LOAD_VLM", "").lower() in {"1", "true", "yes"}:
    _bind_vlm()
else:
    _drop(*_VLM_TOOLS)
