"""LoRA captioner on VRSBench. Last, and optional.

Exists because CIDEr punishes template phrasing, not because the system needs a VLM. It
is trained narrowly on VRSBench's 29,614 human-verified captions and used for the
captioning metric only. It never answers a quantitative question -- those are computed by
`symbolic/`, and routing them here would reintroduce exactly the guessing the pivot
removed.

Deliberately the last thing built. If it is never trained, every mandatory judging row is
still served.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["CaptionerConfig", "LoraCaptioner"]


@dataclass(frozen=True, slots=True)
class CaptionerConfig:
    """Hyperparameters of the captioner run."""

    base_model: str = "akshaydudhane/EarthDial_4B_RGB"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_task_type: str | None = None
    """None, not CAUSAL_LM: PeftModelForCausalLM injects inputs_embeds, which an
    InternVL-based chat model rejects."""
    max_new_tokens: int = 128


class LoraCaptioner:
    """A small VLM with a LoRA adapter, for captioning only."""

    def __init__(self, config: CaptionerConfig) -> None:
        self.config = config
        self._model: Any = None

    def build(self) -> Any:
        """Raises:
        NotImplementedError: always, until the backbone is wired.
        """
        raise NotImplementedError(
            "LoraCaptioner.build is not implemented. Missing: the "
            f"'{self.config.base_model}' backbone (~8 GB, not downloaded automatically) "
            "and the peft adapter attached to it."
        )

    def caption(self, pixel_values: Any) -> str:
        """One caption for one image.

        Raises:
            NotImplementedError: always, until a trained adapter exists.
        """
        raise NotImplementedError(
            "LoraCaptioner.caption is not implemented. Missing: LoraCaptioner.build and a "
            "trained adapter. A placeholder caption would be a fluent sentence about an "
            "image nothing looked at."
        )
