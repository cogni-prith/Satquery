"""Package-wide logging. Nothing in `src/satquery` may call `print`.

TensorBoard is the default experiment tracker; Weights and Biases is used instead
when `WANDB_API_KEY` is present in the environment.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

__all__ = ["configure_logging", "get_logger", "tracker_backend"]

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO, log_file: Path | None = None) -> None:
    """Install the root handler once. Safe to call from every entry script."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring the root handler on first use."""
    configure_logging()
    return logging.getLogger(name)


def tracker_backend() -> str:
    """Return the experiment tracker to use: `"wandb"` if keyed, else `"tensorboard"`."""
    return "wandb" if os.environ.get("WANDB_API_KEY") else "tensorboard"
