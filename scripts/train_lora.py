#!/usr/bin/env python3
"""Entry point for LoRA fine-tuning the VLM backbone on the data mix.

This is the artifact that proves "RS adaptation of a vision or VL component" to the
judges. It uses `transformers.Trainer` with `peft`; there is no hand-rolled loop.

The model and dataset builders are not implemented yet, so this currently fails with a
`NotImplementedError` naming exactly what is missing. It fails after the config has
been validated and the frozen-constants fingerprint logged, so a config mistake shows
up before the missing pieces do.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from satquery.preprocess.constants import constants_fingerprint
from satquery.train.lora import LoraTrainConfig, train
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("train-lora")


def main(argv: list[str] | None = None) -> int:
    """Load the config, seed, and hand off to `satquery.train.lora.train`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/lora_stage1.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    args = parser.parse_args(argv)

    configure_logging()
    cfg = LoraTrainConfig.from_yaml(args.config)
    seed = seed_everything(args.seed if args.seed is not None else cfg.seed)

    LOG.info("config: %s", args.config)
    LOG.info("seed: %s", seed)
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint())
    LOG.info("output: %s", cfg.resolved_output_dir)

    adapter_path = train(cfg)
    LOG.info("adapter written to %s", adapter_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
