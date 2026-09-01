"""Trainer callbacks: the bookkeeping half of a training run, GPU not required.

Two callbacks live here and both are fully implemented, because both are pure
bookkeeping over values `transformers.Trainer` already hands them:

- `ConstantsFingerprintCallback` stamps the frozen-preprocessing fingerprint into the
  run directory, so the eval harness can refuse to score a checkpoint that was
  trained under different constants.
- `ThroughputCallback` turns the batch geometry the Trainer was configured with into
  a samples-per-second line in the log.

`transformers` is an optional `gpu` extra, so this module must import on a CPU-only
laptop with nothing installed. A `TrainerCallback` subclass needs its base class at
class-creation time, which is import time, so the base is resolved once by
`_callback_base()`: the real `transformers.TrainerCallback` when transformers is
importable, `_TrainerCallbackShim` -- a structurally identical no-op carrying the
same hook signatures -- when it is not. Nothing branches on which one was chosen and
nothing needs to: when transformers is absent there is no Trainer to call these back.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from satquery.preprocess.constants import constants_fingerprint
from satquery.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

__all__ = [
    "CONSTANTS_FINGERPRINT_FILENAME",
    "ConstantsFingerprintCallback",
    "ThroughputCallback",
    "write_constants_fingerprint",
]

_LOG = get_logger(__name__)

#: Written next to every saved adapter or head. The eval harness reads it back.
CONSTANTS_FINGERPRINT_FILENAME = "constants_fingerprint.json"


def write_constants_fingerprint(directory: Path, **extra: Any) -> Path:
    """Record the frozen-constants fingerprint in `directory` and return the file written.

    Args:
        directory: Run or checkpoint directory. Created if it does not exist.
        extra: Additional JSON-serialisable run metadata to store alongside the
            fingerprint, for example the run name or the config file used.

    Returns:
        Path of the JSON file written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "constants_fingerprint": constants_fingerprint(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    path = directory / CONSTANTS_FINGERPRINT_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class _TrainerCallbackShim:
    """No-op stand-in for `transformers.TrainerCallback` on installs without transformers.

    It exists only so the subclasses below can be created at import time. Every hook
    mirrors the real signature and returns `None`, which is what the real callback
    handler treats as "leave the control flags alone".
    """

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called once before the first optimiser step."""

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called after each optimiser step, i.e. after gradient accumulation completes."""

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called once after the final optimiser step."""


def _callback_base() -> type:
    """Return the base class for the callbacks below: the real one, or the shim."""
    try:
        from transformers import TrainerCallback
    except ImportError:
        return _TrainerCallbackShim
    return TrainerCallback


_CallbackBase = _callback_base()


class ConstantsFingerprintCallback(_CallbackBase):  # type: ignore[misc,valid-type]
    """Stamp `constants_fingerprint()` into the run directory when training begins.

    Preprocessing drift between the training path and the inference path is silent
    and close to undebuggable, so every run leaves behind the hash of the frozen
    constants it was trained under. `eval/` compares that recorded value against the
    fingerprint of the process loading the checkpoint and fails loudly on a mismatch.
    """

    def __init__(self, run_dir: Path | None = None, **metadata: Any) -> None:
        """Args:
        run_dir: Where to write the record. Defaults to `TrainingArguments.output_dir`.
        metadata: Extra JSON-serialisable fields stored with the fingerprint.
        """
        self.run_dir = run_dir
        self.metadata = metadata
        self.record_path: Path | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Log the fingerprint and write it into the run directory."""
        fingerprint = constants_fingerprint()
        directory = self.run_dir or Path(args.output_dir)
        self.record_path = write_constants_fingerprint(directory, **self.metadata)
        _LOG.info(
            "frozen preprocessing constants fingerprint %s recorded at %s",
            fingerprint,
            self.record_path,
        )


class ThroughputCallback(_CallbackBase):  # type: ignore[misc,valid-type]
    """Log optimiser steps per second and samples per second over a rolling window.

    Samples per second is derived from the batch geometry the Trainer was configured
    with -- `per_device_train_batch_size * gradient_accumulation_steps * world_size`
    -- which is exact for a Trainer because that geometry is fixed for the whole run.
    Nothing here estimates or extrapolates: if a window has not elapsed yet, it logs
    nothing rather than a number it cannot stand behind.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        """Args:
        log_every_n_steps: Optimiser steps per reported window. Minimum 1.
        """
        self.log_every_n_steps = max(1, log_every_n_steps)
        self._window_started: float | None = None
        self._window_start_step: int = 0
        self._samples_per_step: int = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Capture the batch geometry and open the first timing window."""
        world_size = int(getattr(args, "world_size", 1) or 1)
        self._samples_per_step = (
            int(args.per_device_train_batch_size)
            * int(args.gradient_accumulation_steps)
            * world_size
        )
        self._window_start_step = int(getattr(state, "global_step", 0) or 0)
        self._window_started = time.perf_counter()
        _LOG.info(
            "throughput accounting: %d samples per optimiser step "
            "(per_device=%d x accum=%d x world_size=%d)",
            self._samples_per_step,
            args.per_device_train_batch_size,
            args.gradient_accumulation_steps,
            world_size,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Close and report a window once `log_every_n_steps` optimiser steps have passed."""
        if self._window_started is None:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        completed = step - self._window_start_step
        if completed < self.log_every_n_steps:
            return
        elapsed = time.perf_counter() - self._window_started
        if elapsed <= 0.0:
            return
        _LOG.info(
            "step %d | %.3f optimiser steps/s | %.1f samples/s",
            step,
            completed / elapsed,
            completed * self._samples_per_step / elapsed,
        )
        self._window_start_step = step
        self._window_started = time.perf_counter()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Close the timing window so a resumed run does not report across the gap."""
        self._window_started = None
