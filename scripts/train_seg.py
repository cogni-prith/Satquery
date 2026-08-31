#!/usr/bin/env python3
"""Entry point for the seg training run.

Uses `transformers.Trainer`. The model builder is not implemented yet and fails with a
`NotImplementedError` naming exactly what is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.preprocess.constants import constants_fingerprint
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("train-seg")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/segmentation.yaml")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    configure_logging()
    seed_everything(args.seed)
    LOG.info("config: %s", args.config)
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint())

    raise NotImplementedError(
        "seg training is not implemented. Missing: the model builder in "
        "satquery.models, its dataset view, and the Trainer assembly in satquery.train. "
        f"The config surface at {args.config} is complete, so only those pieces remain."
    )


if __name__ == "__main__":
    sys.exit(main())
