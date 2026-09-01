#!/usr/bin/env python3
"""Run any of the three trained models on your own imagery.

    python scripts/predict.py vqa      --image scene.tif --question "How many aircraft?"
    python scripts/predict.py caption  --image scene.tif
    python scripts/predict.py ground   --image scene.tif --phrase "the runway"
    python scripts/predict.py change   --before t1.tif --after t2.tif --question "What changed?"
    python scripts/predict.py fusion   --optical s2.tif --sar s1.tif

Point `--models` at the exported model directory, or set `SATQUERY_MODELS`. Everything
goes through the same `ToolRequest` -> `ToolResult` contract the backend uses, so what
you see here is exactly what the service returns.

**The three models are not alike in what they need beside them.**

`vqa`, `caption` and `ground` share one 35 MB LoRA adapter, which is *not a model*: it
is a set of low-rank deltas that only mean anything applied on top of EarthDial-4B. That
backbone is ~8 GB and is fetched from HuggingFace on first use and cached; the adapter
cannot substitute for it. `change` and `fusion` are self-contained -- their ImageNet
encoder architectures come from `timm`, but every weight that matters is in the
checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
from satquery.io.raster import read_image_ref
from satquery.serve.contracts import TaskType, ToolRequest
from satquery.utils.logging import configure_logging, get_logger

LOG = get_logger("predict")

#: Where the exported models live, unless `--models` or $SATQUERY_MODELS says otherwise.
DEFAULT_MODELS = Path("/media/cyborg-prithwish/Expansion/satquery-models")


def _models_root(argument: str | None) -> Path:
    root = Path(argument or os.environ.get("SATQUERY_MODELS") or DEFAULT_MODELS).expanduser()
    if not root.is_dir():
        raise SystemExit(
            f"model directory not found: {root}\n"
            "Pass --models /path/to/satquery-models, or set SATQUERY_MODELS."
        )
    return root


def _refs(*paths: str):
    """Parse each raster once, so GSD comes from the transform rather than a guess."""
    return [read_image_ref(path) for path in paths]


def _report(result) -> int:
    """Print a ToolResult the way the backend would consume it."""
    if not result.ok:
        LOG.error("%s failed: %s", result.tool_name, result.error)
        return 1

    print(f"\n  tool       : {result.tool_name} v{result.tool_version}")
    print(f"  answer     : {result.answer}")
    if result.confidence is not None:
        print(f"  confidence : {result.confidence:.4f}")
    if result.evidence.boxes:
        for box in result.evidence.boxes:
            print(
                f"  box        : [{box.x_min:.0f}, {box.y_min:.0f}, "
                f"{box.x_max:.0f}, {box.y_max:.0f}]  {box.label}"
            )
    if result.evidence.mask_path:
        print(f"  mask       : {result.evidence.mask_path}")
    for name, path in result.evidence.index_maps.items():
        print(f"  {name:<11}: {path}")
    print(f"  latency    : {result.latency_ms:.0f} ms")
    for warning in result.warnings:
        print(f"  warning    : {warning}")
    return 0


def _vlm_tool(kind: str, models: Path):
    """Build a VLM tool with the LoRA adapter attached to the EarthDial backbone."""
    from satquery.models.vlm.backbone import BackboneConfig, EarthDialBackbone
    from satquery.models.vlm.tasks import CaptionTool, GroundingTool, VqaTool

    adapter = models / "vlm_lora_adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise SystemExit(f"LoRA adapter not found at {adapter}")

    LOG.info("loading EarthDial-4B (about 8 GB, cached after the first run)")
    # The backbone is described by a config file, not by defaults: the repo id, the
    # tiling limits and the chat template all have to match the checkpoint the adapter
    # was trained against, and a mismatch there is silent.
    config = BackboneConfig.from_yaml("configs/model/earthdial_4b_rgb.yaml")
    # Clear the config's own adapter_path before loading. `load()` would otherwise
    # attach whatever that field names, and this script then attaches the adapter the
    # user actually asked for on top -- which the backbone refuses, because a nested
    # adapter leaves both sets of weights inert and the model silently reverts to base
    # behaviour while still answering confidently.
    config = replace(config, adapter_path=None)
    backbone = EarthDialBackbone(config)
    backbone.load()
    LOG.info("attaching adapter from %s", adapter)
    backbone.attach_adapter(adapter)

    return {"vqa": VqaTool, "caption": CaptionTool, "ground": GroundingTool}[kind](
        backbone=backbone
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["vqa", "caption", "ground", "change", "fusion"])
    parser.add_argument("--models", default=None, help="Exported model directory.")
    parser.add_argument("--image", help="Single image, for vqa / caption / ground.")
    parser.add_argument("--before", help="Earlier date, for change.")
    parser.add_argument("--after", help="Later date, for change.")
    parser.add_argument("--optical", help="Optical raster, for fusion.")
    parser.add_argument("--sar", help="Raw SAR raster (VV/VH or HH/HV), for fusion.")
    parser.add_argument("--question", default="What is in this image?")
    parser.add_argument("--phrase", default=None, help="Referring phrase, for ground.")
    args = parser.parse_args(argv)

    configure_logging()
    models = _models_root(args.models)

    if args.mode in {"vqa", "caption", "ground"}:
        if not args.image:
            raise SystemExit(f"--image is required for {args.mode}")
        tool = _vlm_tool(args.mode, models)
        query = args.phrase if args.mode == "ground" else args.question
        task = {"vqa": TaskType.VQA, "caption": TaskType.CAPTION, "ground": TaskType.GROUNDING}[
            args.mode
        ]
        request = ToolRequest(images=_refs(args.image), task=task, query=query or "describe")

    elif args.mode == "change":
        if not (args.before and args.after):
            raise SystemExit("--before and --after are required for change")
        from satquery.models.change.siamese import ChangeVqaTool

        tool = ChangeVqaTool(checkpoint=models / "change_vqa_head")
        request = ToolRequest(
            images=_refs(args.before, args.after),
            task=TaskType.CHANGE_VQA,
            query=args.question,
        )

    else:  # fusion
        if not (args.optical and args.sar):
            raise SystemExit("--optical and --sar are required for fusion")
        from satquery.models.fusion.dual_encoder import FusionConfig, FusionExtractionTool

        tool = FusionExtractionTool(FusionConfig.from_yaml("configs/model/fusion.yaml"))
        request = ToolRequest(
            images=_refs(args.optical, args.sar),
            task=TaskType.FUSION_EXTRACTION,
            query=args.question,
        )

    return _report(tool.run(request))


if __name__ == "__main__":
    sys.exit(main())
