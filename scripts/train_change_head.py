#!/usr/bin/env python3
"""Entry point for training the discriminative Siamese CDVQA change head.

CDVQA's answer set is closed over six classes, so a small classification head beats a
generative VLM on accuracy and runs in milliseconds. The VLM still handles free-form
change description; this head produces the scored answer.

Uses `transformers.Trainer`. The model and dataset builders are not implemented yet and
fail with a `NotImplementedError` naming what is missing.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from satquery.preprocess.constants import CDVQA_ANSWERS, constants_fingerprint
from satquery.train.change_head import ChangeHeadTrainConfig, train
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("train-change")


def main(argv: list[str] | None = None) -> int:
    """Load the config, seed, and hand off to `satquery.train.change_head.train`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/change_head.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    args = parser.parse_args(argv)

    configure_logging()
    cfg = ChangeHeadTrainConfig.from_yaml(args.config)
    seed = seed_everything(args.seed if args.seed is not None else cfg.seed)

    LOG.info("config: %s", args.config)
    LOG.info("seed: %s", seed)
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint())
    LOG.info(
        "closed answer set (%d classes, order is frozen): %s",
        len(CDVQA_ANSWERS),
        list(CDVQA_ANSWERS),
    )

    checkpoint = train(cfg)
    LOG.info("checkpoint written to %s", checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
