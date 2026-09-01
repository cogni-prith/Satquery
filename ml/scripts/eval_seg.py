#!/usr/bin/env python3
"""Score the trained segmenter on the reBEN validation split.

The honest test. `train_seg.py` holds out 10% of its own pack, which shares regions,
acquisitions and distribution with what it trained on -- that number says the head
learned, not that it generalises. This one runs on patches from a different split
entirely.

Per-class IoU, reported per class. A mean over five classes on reBEN is dominated by
forest and farmland and can look healthy while water and built-up, which every area answer
in the system rests on, are near zero. An absent class scores nan, never 0.0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.models.segmentation.landcover import LandCoverSegmenter
from satquery.preprocess.constants import (
    LANDCOVER_CLASSES,
    LANDCOVER_IGNORE_INDEX,
    constants_fingerprint,
)
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("eval-seg")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="scratch/landcover/validation.npz")
    parser.add_argument("--weights", default="train/landcover/final")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)

    configure_logging()
    seed_everything(1337)
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint())

    import torch

    from satquery.utils.paths import artifact_root

    root = Path(__file__).resolve().parents[1]
    cache = root / args.cache
    if not cache.exists():
        LOG.error("no validation cache at %s; run pack_landcover.py --split validation", cache)
        return 1

    weights = artifact_root() / args.weights
    segmenter = LandCoverSegmenter()
    segmenter.load(weights)
    module = segmenter._module
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module.to(device).eval()

    blob = np.load(cache)
    images, masks = blob["images"], blob["masks"]
    LOG.info("scoring %d validation patches from %s", len(images), cache.name)

    classes = len(LANDCOVER_CLASSES)
    intersection = np.zeros(classes, dtype=np.int64)
    union = np.zeros(classes, dtype=np.int64)

    with torch.no_grad():
        for start in range(0, len(images), args.batch):
            batch = torch.from_numpy(images[start : start + args.batch].astype(np.float32))
            target = torch.from_numpy(masks[start : start + args.batch].astype(np.int64))
            logits = module(pixel_values=batch.to(device)).logits
            logits = torch.nn.functional.interpolate(
                logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False
            )
            predicted = logits.argmax(dim=1).cpu()

            valid = target != LANDCOVER_IGNORE_INDEX
            for index in range(classes):
                if index == LANDCOVER_IGNORE_INDEX:
                    continue
                p = (predicted == index) & valid
                t = (target == index) & valid
                intersection[index] += int((p & t).sum())
                union[index] += int((p | t).sum())

    print(f"\n{'class':<14}{'IoU':>9}{'pixels':>14}")
    scored = []
    for index, name in enumerate(LANDCOVER_CLASSES):
        if index == LANDCOVER_IGNORE_INDEX:
            continue
        if union[index] == 0:
            print(f"{name:<14}{'nan':>9}{'absent':>14}")
            continue
        iou = intersection[index] / union[index]
        scored.append(iou)
        print(f"{name:<14}{iou:>9.4f}{int(union[index]):>14,}")
    print(f"\n{'mean (present)':<14}{np.mean(scored):>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
