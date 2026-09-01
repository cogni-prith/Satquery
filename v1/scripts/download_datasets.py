#!/usr/bin/env python3
"""Print dataset acquisition instructions and verify checksums of files already present.

This script DOWNLOADS NOTHING. Several of these corpora are hundreds of gigabytes and
a few are gated behind an access request, so fetching them is a deliberate human act.
What this does is tell you exactly what to run, where to put it, and whether what you
already have is intact.

Data lives outside the repo under `$SATQUERY_DATA_ROOT` and is symlinked into `data/`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from omegaconf import OmegaConf

import _bootstrap  # noqa: F401  (import for its sys.path side effect)
from satquery.utils.logging import get_logger
from satquery.utils.paths import configs_dir, data_root

LOG = get_logger("download")

#: How to obtain each corpus. Single-line commands only, per the project conventions.
INSTRUCTIONS: dict[str, dict[str, str]] = {
    "bigearthnet_txt": {
        "url": "https://txt.bigearth.net",
        "note": "Built on BigEarthNet v2.0 (reBEN). Large; check the licence before use.",
        "command": "echo 'Follow the download instructions at https://txt.bigearth.net and extract into $SATQUERY_DATA_ROOT/bigearthnet_txt'",
    },
    "vrsbench": {
        "url": "https://huggingface.co/datasets/xiang709/VRSBench",
        "note": "29,614 aerial images at 512x512 plus captions, referring expressions and QA.",
        "command": "huggingface-cli download xiang709/VRSBench --repo-type dataset --local-dir $SATQUERY_DATA_ROOT/vrsbench",
    },
    "rsvqa": {
        "url": "https://rsvqa.sylvainlobry.com",
        "note": "Evaluation only. LR is Sentinel-2 at 10 m; HR is aerial at roughly 0.15 m.",
        "command": "echo 'Download the RSVQA LR and HR archives from https://rsvqa.sylvainlobry.com and extract into $SATQUERY_DATA_ROOT/rsvqa'",
    },
    "cdvqa": {
        "url": "https://github.com/YZHJessica/CDVQA",
        "note": "2,968 bi-temporal pairs from the SECOND subset. Two official test splits.",
        "command": "git clone https://github.com/YZHJessica/CDVQA $SATQUERY_DATA_ROOT/cdvqa",
    },
    "levir_cd": {
        "url": "https://chenhao.in/LEVIR/",
        "note": "Optional. Only needed for the change-mask segmentation head.",
        "command": "echo 'Download LEVIR-CD from https://chenhao.in/LEVIR/ and extract into $SATQUERY_DATA_ROOT/levir_cd'",
    },
    "second": {
        "url": "https://captain-whu.github.io/SCD/",
        "note": "Optional. Only needed for the change-mask segmentation head.",
        "command": "echo 'Download SECOND from https://captain-whu.github.io/SCD/ and extract into $SATQUERY_DATA_ROOT/second'",
    },
}


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 so a 40 GB archive does not have to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected_checksums(dataset: str) -> dict[str, str]:
    """Read the `checksums:` block from a dataset config, if the config exists."""
    config_path = configs_dir() / "data" / f"{dataset}.yaml"
    if not config_path.is_file():
        return {}
    config = OmegaConf.load(config_path)
    raw = config.get("checksums") or {}
    return {str(name): str(value) for name, value in dict(raw).items()}


def verify(dataset: str) -> tuple[int, int, int]:
    """Verify a dataset's recorded checksums against what is on disk.

    Returns:
        `(matched, mismatched, missing)`.
    """
    expected = load_expected_checksums(dataset)
    root = data_root() / dataset

    if not expected:
        LOG.info("  no checksums recorded in configs/data/%s.yaml; nothing to verify", dataset)
        return (0, 0, 0)
    if not root.exists():
        LOG.info("  %s does not exist; nothing to verify", root)
        return (0, 0, len(expected))

    matched = mismatched = missing = 0
    for relative, digest in sorted(expected.items()):
        target = root / relative
        if not target.is_file():
            LOG.warning("  MISSING   %s", target)
            missing += 1
            continue
        actual = sha256sum(target)
        if actual == digest:
            LOG.info("  ok        %s", relative)
            matched += 1
        else:
            LOG.error("  MISMATCH  %s", relative)
            LOG.error("            expected %s", digest)
            LOG.error("            actual   %s", actual)
            mismatched += 1
    return (matched, mismatched, missing)


def main(argv: list[str] | None = None) -> int:
    """Print instructions, then verify whatever is already on disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(INSTRUCTIONS), help="Limit to one dataset.")
    parser.add_argument(
        "--verify-only", action="store_true", help="Skip the instructions, just check checksums."
    )
    args = parser.parse_args(argv)

    datasets = [args.dataset] if args.dataset else sorted(INSTRUCTIONS)
    root = data_root()
    LOG.info("SATQUERY_DATA_ROOT resolves to %s", root)
    if not root.exists():
        LOG.warning("that directory does not exist yet; create it or set SATQUERY_DATA_ROOT")

    total_mismatched = 0
    for dataset in datasets:
        entry = INSTRUCTIONS[dataset]
        LOG.info("=" * 78)
        LOG.info("%s  --  %s", dataset, entry["url"])
        LOG.info("  %s", entry["note"])
        if not args.verify_only:
            LOG.info("  target:  %s", root / dataset)
            LOG.info("  run:     %s", entry["command"])
        _, mismatched, _ = verify(dataset)
        total_mismatched += mismatched

    LOG.info("=" * 78)
    if total_mismatched:
        LOG.error("%d file(s) failed checksum verification", total_mismatched)
        return 1
    LOG.info("nothing was downloaded. This script never downloads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
