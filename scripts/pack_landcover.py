#!/usr/bin/env python3
"""Pack reBEN patches and their CORINE reference maps into a local training cache.

Downloads nothing. Reads the LMDB already on the external drive and writes a compact
`.npz` beside the repo.

This exists because of one measurement. The LMDB lives on a spinning USB disk, and a
training loop samples patches in random order: at roughly one random patch per second
that is hours per epoch, and the GPU would idle through all of it. Read once in sorted
key order -- LMDB keys are patch ids, so sorted order is close to physical order -- at
around a thousand keys a second, and every epoch afterwards is served from memory.

Four 10 m bands are packed. B11 and B12 are 20 m and would need upsampling, which adds
no detail and doubles the cache; the four native bands carry blue, green, red and NIR,
which is what separates the five Level-1 classes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.preprocess.constants import CORINE_LEVEL1_TO_CLASS
from satquery.utils.logging import configure_logging, get_logger

LOG = get_logger("pack-landcover")

LMDB = Path("/media/cyborg-prithwish/Expansion/satquery-data/bigearthnet_lmdb")
METADATA = Path("/media/cyborg-prithwish/Expansion/satquery-data/bigearthnet_txt/metadata.parquet")
#: Native 10 m bands: blue, green, red, near-infrared.
BANDS = ("B02", "B03", "B04", "B08")
#: Sentinel-2 L2A is distributed as reflectance scaled by 10000.
REFLECTANCE_SCALE = 10000.0


def to_class_indices(reference: np.ndarray) -> np.ndarray:
    """Map CORINE Level-3 codes onto class indices.

    The leading digit of a Level-3 code is its Level-1 group, so 511 and 512 both become
    water. An unrecognised or zero code becomes 0, which the loss ignores -- teaching a
    pixel we have no label for is worse than not training on it.
    """
    out = np.zeros(reference.shape, dtype=np.uint8)
    for level1, index in CORINE_LEVEL1_TO_CLASS.items():
        low, high = level1 * 100, (level1 + 1) * 100
        out[(reference >= low) & (reference < high)] = index
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=int, default=12000, help="patches to pack")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--out", default="scratch/landcover")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    configure_logging()
    if not (LMDB / "data.mdb").exists():
        LOG.error("reBEN LMDB not found at %s. Mount the drive and re-run.", LMDB)
        return 1

    import pandas as pd

    from satquery.io.reben import ReBENStore

    frame = pd.read_parquet(METADATA)
    frame = frame[
        (frame.split == args.split)
        & (~frame.contains_cloud_or_shadow)
        & (~frame.contains_seasonal_snow)
    ]
    LOG.info("%d clean %s patches available", len(frame), args.split)

    rng = np.random.default_rng(args.seed)
    ids = frame.patch_id.to_numpy()
    if len(ids) > args.patches:
        ids = ids[rng.choice(len(ids), size=args.patches, replace=False)]
    # Sorted before reading. This is the entire performance story; see the docstring.
    ids = np.sort(ids)

    store = ReBENStore(LMDB, readahead=True)
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    skipped = 0

    for index, patch_id in enumerate(ids, start=1):
        try:
            stack, _ = store.optical(patch_id, BANDS)
            reference, _ = store.reference_map(patch_id)
        except Exception:
            skipped += 1
            continue
        labels = to_class_indices(np.asarray(reference))
        if not labels.any():
            skipped += 1  # entirely unlabelled: nothing to learn from
            continue
        images.append((stack / REFLECTANCE_SCALE).astype(np.float16))
        masks.append(labels)
        if index % 1000 == 0:
            LOG.info("  %d/%d read, %d kept", index, len(ids), len(images))

    if not images:
        LOG.error("nothing packed; check the reference maps are in the LMDB")
        return 1

    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}.npz"
    image_array = np.stack(images)
    mask_array = np.stack(masks)
    np.savez(path, images=image_array, masks=mask_array, bands=np.array(BANDS))

    counts = np.bincount(mask_array.ravel(), minlength=6)
    share = counts / counts.sum()
    LOG.info("wrote %s", path)
    LOG.info("  images %s  masks %s", image_array.shape, mask_array.shape)
    LOG.info("  skipped %d patches", skipped)
    # Printed because it decides whether the loss needs class weighting: reBEN is heavily
    # forest and farmland, and a head trained on the raw distribution can score well while
    # never predicting water at all.
    from satquery.preprocess.constants import LANDCOVER_CLASSES

    for name, fraction in zip(LANDCOVER_CLASSES, share, strict=False):
        LOG.info("  %-12s %5.1f%% of pixels", name, fraction * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
