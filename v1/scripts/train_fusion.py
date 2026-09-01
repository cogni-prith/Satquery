#!/usr/bin/env python3
"""Entry point for training the dual-encoder optical + SAR fusion segmenter.

Trains on the co-registered Sentinel-1 / Sentinel-2 pairs of reBEN (BigEarthNet v2.0),
targeting the built-up and water extraction masks derived from the CORINE reference
maps. This is the artifact for the optical-and-SAR row of the problem statement.

Uses `transformers.Trainer`. Selection is on mean IoU, never accuracy -- see
`satquery.train.fusion` for why accuracy is the wrong metric on these classes.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from satquery.preprocess.constants import FUSION_EXTRACTION_CLASSES, constants_fingerprint
from satquery.train.fusion import FusionTrainConfig, train
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("train-fusion")


def main(argv: list[str] | None = None) -> int:
    """Load the config, seed, and hand off to `satquery.train.fusion.train`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/fusion.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap the training split. Note the cap is a contiguous slice of a tile-ordered corpus, so a capped run is not representative.",
    )
    args = parser.parse_args(argv)

    configure_logging()
    cfg = FusionTrainConfig.from_yaml(args.config)
    if args.max_train_samples is not None:
        cfg.max_train_samples = args.max_train_samples
    seed = seed_everything(args.seed if args.seed is not None else cfg.seed)

    LOG.info("config: %s", args.config)
    LOG.info("seed: %s", seed)
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint())
    LOG.info(
        "extraction targets (%d, order is the output-channel order): %s",
        len(FUSION_EXTRACTION_CLASSES),
        list(FUSION_EXTRACTION_CLASSES),
    )

    checkpoint = train(cfg)
    LOG.info("checkpoint written to %s", checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
