"""Session-wide test guards.

The suite must run offline, on CPU, in under a second. Two of the tools now have real
model-loading code behind them, and a test that accidentally reaches `.load()` would
try to pull an 8 GB checkpoint -- which is exactly what happened once, so it is fenced
off here rather than left to discipline.

`HF_HUB_OFFLINE` stops any Hub call from touching the network. It does not stop a
*cached* checkpoint from loading, so tests that must not load weights construct their
tools with `backbone=None` instead of going through the registry.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _offline_and_cpu_only() -> None:
    """Force every Hub client offline and hide any GPU for the duration of the suite."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
