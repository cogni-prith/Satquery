#!/usr/bin/env python3
"""Validate a training mix config and report what it would draw from.

Loading the mix is where the 40/40/20 weights are checked. A mix whose weights do not
sum to 1.0 trains on a silently different distribution from the one written down, and
that is invisible in the loss curve, so `load_mix_spec` treats it as a hard error and
this script surfaces it before a GPU-day is spent.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from satquery.data.mixer import load_mix_spec
from satquery.preprocess.constants import constants_fingerprint
from satquery.utils.logging import get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("build-mix")


def main(argv: list[str] | None = None) -> int:
    """Load, validate and describe a mix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/mix_v1.yaml", help="Mix YAML path.")
    args = parser.parse_args(argv)

    seed_everything()
    spec = load_mix_spec(args.config)

    LOG.info("mix %r from %s", spec.name, args.config)
    LOG.info("constants fingerprint: %s", constants_fingerprint())
    LOG.info("seed: %s", spec.seed)
    LOG.info("components:")
    for component in spec.components:
        LOG.info(
            "  %-18s weight %.2f  splits %s  config %s",
            component.name,
            component.weight,
            list(component.splits) or ["<default>"],
            component.config.name,
        )

    policy = spec.scale_policy
    LOG.info(
        "scale policy: enabled=%s apply_probability=%s", policy.enabled, policy.apply_probability
    )
    LOG.info("  scales (m):    %s", list(policy.scales_m))
    LOG.info("  probabilities: %s", list(policy.probabilities) or "<uniform>")
    LOG.info("mix validated. Component loaders are stubs, so no samples were materialised.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
