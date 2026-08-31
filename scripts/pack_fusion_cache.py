#!/usr/bin/env python3
"""Pack a fusion training cache from the reBEN LMDB in one streaming pass.

Why this exists, measured rather than assumed: the reBEN LMDB lives on a spinning USB
drive, where sequential reads run at 1,009 keys/s (111 MB/s) but random reads -- the
access pattern of a shuffled training epoch -- collapse to about 1 patch/s. That is a
seek limit, not something LMDB tuning can recover. Training directly against it would
take roughly 130 hours for two epochs over the full split.

So the corpus is read once, in cursor order, at the drive's full streaming speed, and
written to a single packed `uint8` memmap on fast local storage. Training then does
random access against a flat file at SSD latency.

Layout. One row per patch, `PATCH_BYTES` wide::

    [ optical 4x120x120 | sar 3x120x120 | mask 120x120 ]

A single memmap rather than a file tree because the same 128 KB-cluster and small-file
costs that shaped the LMDB conversion apply again here: 150,000 patches would otherwise
be 450,000 files.

Selection is a **stride** through the split, never a head slice. reBEN rows are grouped
by Sentinel tile, so the first N rows are a handful of European scenes; a stride of
`len(split) // N` spreads the subset across every tile and country in the split, which
is what makes a capped run worth reporting at all.

Both the optical and SAR arrays are written having already been through the frozen
preprocessing -- `stretch_to_uint8` and `render_sar` -- so the packed cache holds
exactly the pixels the model would have seen through the raster path. `tests/` asserts
that equivalence rather than trusting this comment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from satquery.data.datasets.bigearthnet_fusion import corine_extraction_mask
from satquery.io.reben import REFERENCE_MAP_SUFFIX, ReBENStore
from satquery.preprocess.constants import OPTICAL_BAND_ORDER_4, constants_fingerprint
from satquery.preprocess.optical import stretch_to_uint8
from satquery.preprocess.sar import render_sar
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.paths import artifact_root, dataset_dir

LOG = get_logger("pack-fusion")

PATCH_SIDE = 120
OPTICAL_BYTES = 4 * PATCH_SIDE * PATCH_SIDE
SAR_BYTES = 3 * PATCH_SIDE * PATCH_SIDE
MASK_BYTES = PATCH_SIDE * PATCH_SIDE
PATCH_BYTES = OPTICAL_BYTES + SAR_BYTES + MASK_BYTES


def _select(frame, cap: int):
    """Return a stride-selected subset of `frame`, spread across the whole split."""
    if cap <= 0 or cap >= len(frame):
        return frame
    stride = len(frame) // cap
    return frame.iloc[::stride].head(cap)


def _scan_prefix(store: ReBENStore, prefix: str, wanted: dict[str, int]):
    """Yield `(key, index, bands)` for wanted keys under `prefix`, in cursor order.

    Range-scans rather than iterating the whole database, so the three passes together
    read the store roughly once instead of three times.
    """
    from safetensors.numpy import load

    env = store._environment()
    with env.begin(write=False) as transaction:
        cursor = transaction.cursor()
        if not cursor.set_range(prefix.encode("utf-8")):
            return
        for raw_key, payload in cursor:
            key = raw_key.decode("utf-8")
            if not key.startswith(prefix):
                break
            index = wanted.get(key)
            if index is not None:
                yield key, index, dict(load(bytes(payload)))


def pack(split: str, cap: int, out_dir: Path, lmdb_path: Path, root: Path) -> Path:
    """Write the packed cache for one split and return its directory."""
    import pandas as pd

    frame = pd.read_parquet(root / "metadata.parquet")
    frame = frame[frame["split"] == split]
    subset = _select(frame, cap).reset_index(drop=True)
    count = len(subset)

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "patches.u8"
    LOG.info(
        "packing %d of %d %r patches -> %s (%.1f GB)",
        count,
        len(frame),
        split,
        data_path,
        count * PATCH_BYTES / 2**30,
    )

    packed = np.lib.format.open_memmap(
        data_path, mode="w+", dtype=np.uint8, shape=(count, PATCH_BYTES)
    )
    filled = np.zeros((count, 3), dtype=bool)

    optical_index = {
        str(row.patch_id): position for position, row in enumerate(subset.itertuples())
    }
    sar_index = {str(row.s1_name): position for position, row in enumerate(subset.itertuples())}
    reference_index = {
        f"{row.patch_id}{REFERENCE_MAP_SUFFIX}": position
        for position, row in enumerate(subset.itertuples())
    }

    # readahead=True: this is a full cursor scan, and without OS prefetch the
    # same scan drops from about 1,009 keys/s to 54 on this drive.
    store = ReBENStore(lmdb_path, readahead=True)
    started = time.time()

    # -- pass 1: Sentinel-1, which sorts before everything else
    done = 0
    for key, position, bands in _scan_prefix(store, "S1", sar_index):
        missing = [pol for pol in ("VV", "VH") if pol not in bands]
        if missing:
            LOG.warning("%s missing %s, left unfilled", key, missing)
            continue
        from satquery.preprocess.sar import db_to_linear

        rendered, _ = render_sar(db_to_linear(bands["VV"]), db_to_linear(bands["VH"]))
        packed[position, OPTICAL_BYTES : OPTICAL_BYTES + SAR_BYTES] = np.transpose(
            rendered, (2, 0, 1)
        ).ravel()
        filled[position, 1] = True
        done += 1
        if done % 5000 == 0:
            LOG.info("  S1 %d/%d  %.0f patches/s", done, count, done / (time.time() - started))

    # -- pass 2: Sentinel-2 and the reference maps, interleaved under the same prefix
    both = {**optical_index, **reference_index}
    done = 0
    mark = time.time()
    for key, position, bands in _scan_prefix(store, "S2", both):
        if key.endswith(REFERENCE_MAP_SUFFIX):
            packed[position, OPTICAL_BYTES + SAR_BYTES :] = corine_extraction_mask(
                bands["Data"]
            ).ravel()
            filled[position, 2] = True
        else:
            missing = [band for band in OPTICAL_BAND_ORDER_4 if band not in bands]
            if missing:
                LOG.warning("%s missing %s, left unfilled", key, missing)
                continue
            stack = np.stack([bands[band] for band in OPTICAL_BAND_ORDER_4]).astype(np.float32)
            packed[position, :OPTICAL_BYTES] = stretch_to_uint8(stack).ravel()
            filled[position, 0] = True
        done += 1
        if done % 10000 == 0:
            LOG.info("  S2 %d/%d  %.0f keys/s", done, count * 2, done / (time.time() - mark))

    packed.flush()

    complete = filled.all(axis=1)
    if not complete.all():
        LOG.warning(
            "%d of %d patches were incomplete and are excluded from the index",
            int((~complete).sum()),
            count,
        )

    index = subset.loc[complete, ["patch_id", "s1_name", "labels", "country"]].copy()
    index["row"] = np.flatnonzero(complete)
    index.to_parquet(out_dir / "index.parquet")

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "split": split,
                "rows": int(complete.sum()),
                "patch_side": PATCH_SIDE,
                "optical_bands": list(OPTICAL_BAND_ORDER_4),
                "patch_bytes": PATCH_BYTES,
                "constants_fingerprint": constants_fingerprint(),
                "selection": "stride",
                "source_rows": len(frame),
            },
            indent=2,
        )
    )
    LOG.info(
        "packed %d patches in %.1f min -> %s",
        int(complete.sum()),
        (time.time() - started) / 60,
        out_dir,
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cap", type=int, default=-1, help="-1 packs the whole split.")
    parser.add_argument(
        "--out", default=None, help="Defaults to <artifact_root>/cache/fusion_packed/<split>."
    )
    parser.add_argument("--lmdb", default=None)
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)

    configure_logging()
    root = Path(args.root) if args.root else dataset_dir("bigearthnet_txt")
    lmdb_path = Path(args.lmdb) if args.lmdb else root.parent / "bigearthnet_lmdb"
    out = Path(args.out) if args.out else artifact_root() / "cache" / "fusion_packed" / args.split

    pack(args.split, args.cap, out, lmdb_path, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
