#!/usr/bin/env python3
"""Download an EarthDial checkpoint and repair its missing remote code.

Two jobs:

1. Pull the weights for a model config's ``hf_repo_id`` from the HuggingFace Hub.
2. Copy in the custom modelling modules the published repo's ``auto_map`` references
   but does not actually ship. Without step 2, ``trust_remote_code=True`` fails on a
   missing file, which reads like a corrupted download but is a defect in the upstream
   repo. See ``satquery.models.vlm.backbone.ensure_remote_code``.

This is the only script in the repo that downloads model weights, and it only runs when
you ask it to. Datasets are still never downloaded automatically -- see
``scripts/download_datasets.py``, which prints commands and fetches nothing.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from satquery.models.vlm.backbone import (
    REMOTE_CODE_FILES,
    REMOTE_CODE_SOURCE,
    BackboneConfig,
    EarthDialBackbone,
    ensure_remote_code,
)
from satquery.utils.logging import configure_logging, get_logger

LOG = get_logger("fetch-model")


def main(argv: list[str] | None = None) -> int:
    """Resolve the config, download the snapshot, patch the remote code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/earthdial_4b_rgb.yaml")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Fail instead of downloading; use to check what is already cached.",
    )
    args = parser.parse_args(argv)

    configure_logging()
    cfg = BackboneConfig.from_yaml(args.config)
    LOG.info("config:  %s", args.config)
    LOG.info("repo:    %s (revision %s)", cfg.hf_repo_id, cfg.revision or "main")

    backbone = EarthDialBackbone(cfg)
    try:
        local_dir = backbone.snapshot(allow_download=not args.offline)
    except Exception as exc:
        LOG.error("could not resolve the checkpoint: %s", exc)
        return 1

    LOG.info("weights: %s", local_dir)

    copied = ensure_remote_code(local_dir)
    if copied:
        LOG.info("patched %d missing module(s) from %s", len(copied), REMOTE_CODE_SOURCE)
    else:
        LOG.info("remote code already complete (%d modules)", len(REMOTE_CODE_FILES))

    missing = [name for name in REMOTE_CODE_FILES if not (local_dir / name).exists()]
    if missing:
        LOG.error("still missing after patching: %s", missing)
        return 1

    LOG.info("ready. The checkpoint can now be loaded with trust_remote_code=True.")
    LOG.info("note: loading also needs a torchvision build matching your torch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
