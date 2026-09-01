#!/usr/bin/env python3
"""Entry point for the full evaluation suite. This is what `make eval` runs.

Runs every scored task, isolating per-task failure so one missing model does not abort
the suite, and writes a markdown scores table plus a JSON dump under the artifact root.

Any metric that did not actually run is reported as the literal string `TBD`. It is
never a zero and never an estimate: a fabricated score in a results table is worse than
no score at all.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from satquery.eval.harness import EvalSuiteConfig, run_suite
from satquery.eval.report import write_json, write_scores_table
from satquery.preprocess.constants import constants_fingerprint
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("run-eval")


def main(argv: list[str] | None = None) -> int:
    """Load the suite config, check for preprocessing drift, run, and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval/full_suite.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    args = parser.parse_args(argv)

    configure_logging()
    cfg = EvalSuiteConfig.from_yaml(args.config)
    seed = seed_everything(args.seed if args.seed is not None else cfg.seed)

    LOG.info("config: %s", args.config)
    LOG.info("seed: %s", seed)
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint())

    # run_suite calls assert_constants_match first, so preprocessing drift since the
    # checkpoint was trained fails loudly before a single sample is scored.
    results = run_suite(cfg)

    output_dir = cfg.output_dir
    table_path = write_scores_table(results, output_dir / "scores.md")
    json_path = write_json(results, output_dir / "scores.json")

    LOG.info("scores table: %s", table_path)
    LOG.info("scores json:  %s", json_path)
    LOG.info("")
    LOG.info("%s", table_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
