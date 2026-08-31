#!/usr/bin/env python3
"""Run one real inference through the VLM tools. Proves the backbone actually works.

Needs the EarthDial checkpoint on disk (`make fetch-model`) and a GPU. Unlike
`scripts/smoke_test.py`, which must pass on a clean machine with nothing downloaded,
this deliberately loads several gigabytes of weights.

Point it at your own GeoTIFF with `--image`, or let it build a synthetic scene. Note
that answers on the synthetic scene mean very little: it is band arithmetic, not a
photograph of the Earth, so it exercises the plumbing rather than the model's accuracy.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from satquery.io.raster import read_image_ref, write_raster
from satquery.io.validate import validate_inputs
from satquery.models.vlm.backbone import BackboneConfig, EarthDialBackbone
from satquery.models.vlm.tasks import CaptionTool, GroundingTool, VqaTool
from satquery.serve.contracts import ToolRequest, Trace
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("infer")

DEFAULT_QUESTIONS: tuple[str, ...] = (
    "Describe the land-cover and major objects visible in this image.",
    "Is there water in this image? Answer yes or no.",
)


def _synthetic_scene(directory: Path) -> Path:
    """Write a 512x512 four-band scene: water on the left, vegetation on the right."""
    rng = np.random.default_rng(0)
    stack = rng.uniform(0.05, 0.30, (4, 512, 512)).astype("float32")
    stack[1, :, :170] = 0.30  # green high over water
    stack[3, :, :170] = 0.02  # NIR absorbed by water
    stack[3, :, 170:] = 0.55  # NIR high over vegetation
    stack[2, :, 170:] = 0.05  # red low over vegetation
    path = directory / "synthetic_scene.tif"
    write_raster(
        path,
        stack,
        crs="EPSG:32643",
        transform=(10.0, 0.0, 600000.0, 0.0, -10.0, 2000000.0),
        band_names=["B02", "B03", "B04", "B08"],
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """Load the backbone once and run every requested question through it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/earthdial_4b_rgb.yaml")
    parser.add_argument("--image", type=Path, default=None, help="GeoTIFF to ask about.")
    parser.add_argument("--question", action="append", default=None, help="Repeatable.")
    parser.add_argument("--ground", default=None, help="Referring phrase for the grounding tool.")
    args = parser.parse_args(argv)

    configure_logging()
    seed_everything()

    with tempfile.TemporaryDirectory(prefix="satquery-infer-") as tmp:
        path = args.image or _synthetic_scene(Path(tmp))
        ref = read_image_ref(path)
        config, warnings = validate_inputs([ref])

        LOG.info("image     %s", ref.path.name)
        LOG.info("modality  %s   gsd %s -> %s", ref.modality.value, ref.gsd_m, ref.gsd_token)
        LOG.info("config    %s", config.value)
        for warning in warnings:
            LOG.info("warning   %s", warning)

        backbone = EarthDialBackbone(BackboneConfig.from_yaml(args.config))
        started = time.perf_counter()
        backbone.load(allow_download=False)
        LOG.info("backbone loaded in %.1fs", time.perf_counter() - started)

        trace = Trace(request_id="infer-demo", input_config=config)
        questions = args.question or list(DEFAULT_QUESTIONS)

        jobs: list[tuple[type, str]] = [(CaptionTool, questions[0])]
        jobs += [(VqaTool, q) for q in questions[1:]]
        if args.ground:
            jobs.append((GroundingTool, args.ground))

        failures = 0
        for tool_class, question in jobs:
            tool = tool_class(backbone=backbone)
            result = tool.run(ToolRequest(query=question, images=[ref]))
            trace.record(result)
            LOG.info("-" * 70)
            LOG.info("%s", tool.spec.name)
            LOG.info("  Q  %s", question)
            if result.ok:
                LOG.info("  A  (%.0f ms) %s", result.latency_ms, result.answer)
                for box in result.evidence.boxes:
                    LOG.info("  box %s score=%s", [round(v, 1) for v in box.xyxy], box.score)
            else:
                failures += 1
                LOG.error("  ERROR %s", result.error)

        LOG.info("-" * 70)
        LOG.info("trace:\n%s", trace.to_json())

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
