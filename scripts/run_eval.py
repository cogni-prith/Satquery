#!/usr/bin/env python3
"""Run every benchmark in one command and write one report.

`make eval` is the only entry point, so a score in the README is a claim someone else can
reproduce in a single step. Suites that cannot run report TBD; none of them report a guess.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.preprocess.constants import constants_fingerprint
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("eval")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval/full_suite.yaml")
    parser.add_argument("--suite", action="append", help="restrict to named suite(s)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    configure_logging()
    seed_everything(args.seed)
    LOG.info("config: %s", args.config)
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint())

    raise NotImplementedError(
        "eval.harness.run_suite is not implemented, so run_eval has nothing to drive. "
        "Missing: the per-task adapters that turn a suite entry into predictions, and the "
        f"models they call ({args.suite or 'all suites'}). The suite definitions, the "
        "metrics and the TBD-preserving report writer exist. Running a partial harness and "
        "reporting the suites that happened to work would publish a table that looks "
        "complete and is not."
    )


if __name__ == "__main__":
    sys.exit(main())
