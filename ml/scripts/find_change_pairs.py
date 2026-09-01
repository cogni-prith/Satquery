#!/usr/bin/env python3
"""Find and export the BigEarthNet cells whose water extent changed most between dates.

Reads the reBEN LMDB you already have on disk. Downloads nothing.

BigEarthNet is not a change-detection dataset -- it ships single-date patches for
land-cover classification. But a patch id encodes tile, cell and acquisition datetime, and
the same 1.2 km cell is often imaged two to four times. That makes it the only source here
carrying near-infrared at two dates, which is what NDWI needs, so it is the only source
the deterministic tool can actually measure change on.

Reads are issued in sorted key order. LMDB keys are the patch ids, so sorted order is
close to physical order on disk: on this external drive that is the difference between
roughly a thousand keys a second and roughly one, and the scan goes from hours to minutes.

The change is real but seasonal -- reservoirs filling, wetlands flooding, crops greening.
It is not the urban-expansion before/after that SECOND provides, and a pair found here
should not be presented as one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.utils.logging import configure_logging, get_logger

LOG = get_logger("find-pairs")

LMDB = Path("/media/cyborg-prithwish/Expansion/satquery-data/bigearthnet_lmdb")
METADATA = Path("/media/cyborg-prithwish/Expansion/satquery-data/bigearthnet_txt/metadata.parquet")
PATCH_RE = r"^S2[AB]_MSIL2A_(?P<dt>\d{8}T\d{6})_\w+_\w+_(?P<tile>T\w+)_(?P<r>\d+)_(?P<c>\d+)$"

#: Labels that mean the cell plausibly contains water worth watching.
WATER_LABELS = {"Inland waters", "Inland wetlands", "Coastal wetlands", "Marine waters"}

#: Below this, a "change" is a handful of pixels flickering on a threshold, not an event.
MIN_FRACTION_CHANGE = 0.04

#: Both dates must hold at least this much water.
#:
#: This is the ice filter, and it is the difference between a useful pair and a wrong one.
#: NDWI reads a frozen lake as not-water: ice is bright in NIR where open water is dark.
#: A boreal winter/summer pair therefore shows a lake going from 0% to 95% "water", and
#: the honest description of that is "the lake thawed", not "the lake filled". The first
#: run of this script returned eight Finnish cells doing exactly that, and
#: `contains_seasonal_snow` does not flag them because the snow is on the water, not the
#: land. Requiring water at BOTH dates selects change in extent rather than a change in
#: phase, which is the thing the tool can actually speak about.
MIN_WATER_BOTH_DATES = 0.03

#: Months when ice is plausible at high latitude, used only to annotate the manifest.
FREEZE_MONTHS = {"11", "12", "01", "02", "03", "04"}


def water_fraction(store, patch_id: str) -> float:
    """Share of the patch above the NDWI water threshold. Scale-free, so no GSD needed."""
    stored = store.read(patch_id)
    green = np.asarray(stored["B03"], dtype=np.float64)
    nir = np.asarray(stored["B08"], dtype=np.float64)
    ndwi = (green - nir) / (green + nir + 1e-10)
    return float(np.count_nonzero(ndwi > 0.0) / ndwi.size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=int, default=600, help="candidate cells to measure")
    parser.add_argument("--top", type=int, default=6, help="pairs to export")
    parser.add_argument("--out", default="demo_inputs/pairs", help="output directory")
    parser.add_argument("--seed", type=int, default=1337, help="sampling seed")
    args = parser.parse_args(argv)

    configure_logging()
    if not (LMDB / "data.mdb").exists():
        LOG.error("reBEN LMDB not found at %s. Mount the drive and re-run.", LMDB)
        return 1

    import pandas as pd

    from satquery.io.reben import ReBENStore

    LOG.info("reading metadata")
    frame = pd.read_parquet(METADATA)
    frame = frame.join(frame.patch_id.str.extract(PATCH_RE))
    frame = frame[(~frame.contains_cloud_or_shadow) & (~frame.contains_seasonal_snow)]
    frame = frame[frame.labels.apply(lambda labels: bool(WATER_LABELS & set(labels)))]

    # Keep only cells seen at two or more dates, and take the widest separation available.
    candidates: list[tuple[tuple, str, str, str]] = []
    for key, group in frame.groupby(["tile", "r", "c"]):
        if group.dt.nunique() < 2:
            continue
        ordered = group.sort_values("dt")
        first, last = ordered.iloc[0], ordered.iloc[-1]
        candidates.append((key, first.patch_id, last.patch_id, str(first.country)))
    LOG.info("%d water cells imaged 2+ times; measuring %d", len(candidates), args.scan)

    # Sample across the whole set, not the head of it. Taking the first N walks group
    # order, which is tile order, which is geography -- the first run measured 600 cells
    # that were all neighbours in one stable region and found nothing that changed.
    rng = np.random.default_rng(args.seed)
    if len(candidates) > args.scan:
        picked = rng.choice(len(candidates), size=args.scan, replace=False)
        candidates = [candidates[i] for i in sorted(picked)]

    # Sorted key order: see the module docstring. This is the whole performance story.
    wanted = sorted({pid for _, a, b, _ in candidates for pid in (a, b)})
    # A scan of 5,000 patches costs about nine minutes on the external drive, and the
    # ranking rules are the part that gets iterated on. Cache the measurements so a
    # changed threshold is a free re-run rather than another nine minutes.
    cache_path = Path(__file__).resolve().parents[1] / "scratch" / "water_fractions.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fractions: dict[str, float] = {}
    if cache_path.exists():
        fractions = json.loads(cache_path.read_text())
        LOG.info("loaded %d cached measurements from %s", len(fractions), cache_path.name)

    store = ReBENStore(LMDB, readahead=True)
    todo = [pid for pid in wanted if pid not in fractions]
    for index, patch_id in enumerate(todo, start=1):
        try:
            fractions[patch_id] = water_fraction(store, patch_id)
        except Exception as exc:  # one unreadable patch must not end the scan
            LOG.warning("skipping %s: %s", patch_id, exc)
        if index % 100 == 0:
            LOG.info("  measured %d/%d patches", index, len(todo))
    cache_path.write_text(json.dumps(fractions))

    ranked = []
    for key, first, last, country in candidates:
        if first not in fractions or last not in fractions:
            continue
        change = fractions[last] - fractions[first]
        if abs(change) < MIN_FRACTION_CHANGE:
            continue
        if min(fractions[first], fractions[last]) < MIN_WATER_BOTH_DATES:
            continue  # water absent at one date: a phase change, not an extent change
        months = (first.split("_")[2][4:6], last.split("_")[2][4:6])
        ranked.append(
            {
                "cell": f"{key[0]}_{key[1]}_{key[2]}",
                "tile": key[0],
                "country": country,
                "t1": first,
                "t2": last,
                "months": list(months),
                # Flagged, not dropped: at high latitude a winter date can still carry ice
                # on part of the surface even when both dates hold open water.
                "possible_ice": any(month in FREEZE_MONTHS for month in months),
                "water_t1": round(fractions[first], 4),
                "water_t2": round(fractions[last], 4),
                "change": round(change, 4),
            }
        )
    ranked.sort(key=lambda row: abs(row["change"]), reverse=True)

    # One pair per tile. Neighbouring cells in one tile are the same lake on the same two
    # days, so an unfiltered top-N is one place reported eight times.
    seen_tiles: set[str] = set()
    diverse = []
    for row in ranked:
        if row["tile"] in seen_tiles:
            continue
        seen_tiles.add(row["tile"])
        diverse.append(row)
    LOG.info("%d cells cleared the floors; %d distinct tiles", len(ranked), len(diverse))
    ranked = diverse
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    from export_demo_inputs import read_patch, write_tif

    exported = []
    for row in ranked[: args.top]:
        direction = "filled" if row["change"] > 0 else "drained"
        stem = f"{row['cell']}_{row['country'].lower().replace(' ', '_')}_{direction}"
        for suffix, patch_id in (("date_1", row["t1"]), ("date_2", row["t2"])):
            date = patch_id.split("_")[2][:8]
            path = out / f"{stem}__{suffix}_{date[:4]}-{date[4:6]}-{date[6:]}.tif"
            write_tif(path, read_patch(store, patch_id))
        exported.append(row)
        LOG.info(
            "  %s  %s  water %.1f%% -> %.1f%%  (%+.1f pp)",
            row["cell"],
            row["country"],
            row["water_t1"] * 100,
            row["water_t2"] * 100,
            row["change"] * 100,
        )

    manifest = out / "pairs.json"
    manifest.write_text(json.dumps(exported, indent=2))
    print(f"\nexported {len(exported)} pairs to {out}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
