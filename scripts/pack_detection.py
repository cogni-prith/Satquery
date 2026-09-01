#!/usr/bin/env python3
"""Pack VRSBench referring boxes into a detection training cache. Downloads nothing.

VRSBench ships referring expressions with an object class and a normalised box corner per
record: 16,159 boxes over 26 classes across 9,318 images. That is a real detection set --
it is DOTA imagery underneath -- and it is what makes a trained detector possible here.

One thing it is NOT is open-vocabulary supervision. The classes are a closed list of 26.
A detector fitted to them detects those 26 things, and calling it open-vocabulary because
the tool is named `detector.openvocab` would be a claim the training data cannot support.
The spec description is corrected to say so.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.utils.logging import configure_logging, get_logger

LOG = get_logger("pack-detection")

ROOT = Path("/media/cyborg-prithwish/Expansion/satquery-data/vrsbench")
#: Model input size. VRSBench tiles are 512; 384 keeps a batch on an 8 GB card.
SIZE = 384


def parse_box(record: dict) -> tuple[float, float, float, float] | None:
    """Normalised xyxy from a record, or None when it cannot be read.

    `ground_truth` is EarthDial's `{<x1><y1><x2><y2>}` token form on a 0-100 grid.
    `obj_corner` is a polygon in unit coordinates. The token form is preferred because it
    is what the grounding tool already emits, so both paths share one convention.
    """
    text = record.get("ground_truth") or ""
    numbers = [int(n) for n in re.findall(r"<(\d+)>", text)]
    if len(numbers) == 4:
        x1, y1, x2, y2 = (n / 100.0 for n in numbers)
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2

    corner = record.get("obj_corner")
    if corner:
        try:
            points = np.asarray(ast.literal_eval(corner), dtype=np.float64).reshape(-1, 2)
        except Exception:
            return None
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        if x2 > x1 and y2 > y1:
            return float(x1), float(y1), float(x2), float(y2)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=4000, help="images to pack")
    parser.add_argument("--out", default="scratch/detection")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    configure_logging()
    annotations = ROOT / "VRSBench_EVAL_referring.json"
    if not annotations.exists():
        LOG.error("no VRSBench annotations at %s", annotations)
        return 1

    from PIL import Image

    records = json.loads(annotations.read_text())
    by_image: dict[str, list] = {}
    classes = Counter()
    unreadable = 0
    for record in records:
        box = parse_box(record)
        if box is None:
            unreadable += 1
            continue
        label = record.get("obj_cls") or "object"
        by_image.setdefault(record["image_id"], []).append((box, label))
        classes[label] += 1

    vocabulary = sorted(classes)
    index_of = {name: i for i, name in enumerate(vocabulary)}
    LOG.info(
        "%d boxes over %d classes, %d images; %d boxes unreadable",
        sum(classes.values()),
        len(vocabulary),
        len(by_image),
        unreadable,
    )

    # Find where the images actually live: the archives extract into their own subtree.
    roots = [p for p in (ROOT / "images").rglob("*") if p.is_dir()]
    lookup: dict[str, Path] = {}
    for directory in roots:
        for path in directory.glob("*.png"):
            lookup.setdefault(path.name, path)
    LOG.info("indexed %d image files", len(lookup))

    names = sorted(by_image)
    rng = np.random.default_rng(args.seed)
    if len(names) > args.images:
        names = [names[i] for i in sorted(rng.choice(len(names), args.images, replace=False))]

    images: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    missing = 0
    for count, name in enumerate(names, start=1):
        path = lookup.get(name)
        if path is None:
            missing += 1
            continue
        try:
            picture = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        except Exception:
            missing += 1
            continue
        entries = by_image[name]
        images.append(np.asarray(picture, dtype=np.uint8))
        # Boxes stay normalised: the image was resized, and a normalised box survives that
        # without a second place to get the scaling wrong.
        boxes.append(np.array([b for b, _ in entries], dtype=np.float32))
        labels.append(np.array([index_of[c] for _, c in entries], dtype=np.int64))
        if count % 500 == 0:
            LOG.info("  %d/%d packed", count, len(names))

    if not images:
        LOG.error("nothing packed; are the VRSBench images extracted?")
        return 1

    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.npz"
    np.savez(
        path,
        images=np.stack(images),
        boxes=np.array(boxes, dtype=object),
        labels=np.array(labels, dtype=object),
        vocabulary=np.array(vocabulary),
        allow_pickle=True,
    )
    LOG.info(
        "wrote %s: %d images, %d boxes, %d missing files",
        path,
        len(images),
        sum(len(b) for b in boxes),
        missing,
    )
    for name, total in classes.most_common(8):
        LOG.info("  %-24s %d", name, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
