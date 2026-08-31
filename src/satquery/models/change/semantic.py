"""Semantic change detection on SECOND and LEVIR-CD.

Produces from-class to-class transitions, not merely a binary "something changed" mask.
That distinction is the whole reason this model exists: `symbolic/measures.change_matrix`
turns transitions into "this many square metres went from vegetation to built-up", which
is an answer. A binary mask cannot say what something became.

Siamese encoder with shared weights over both dates, a temporal-difference head, and two
outputs: a binary change mask and a pair of semantic masks. SECOND supplies exactly this
supervision; LEVIR-CD supplies the binary half only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["SemanticChangeConfig", "SemanticChangeNet"]


@dataclass(frozen=True, slots=True)
class SemanticChangeConfig:
    """Hyperparameters of the semantic change run."""

    encoder: str = "resnet18"
    shared_weights: bool = True
    """One encoder over both dates. Separate towers would let the model key on
    acquisition artefacts rather than on change."""
    num_classes: int = 6
    in_channels: int = 3
    image_size: int = 512


class SemanticChangeNet:
    """Weight-shared Siamese encoder producing transitions plus a binary mask."""

    def __init__(self, config: SemanticChangeConfig) -> None:
        self.config = config
        self._module: Any = None

    def build(self) -> Any:
        """Raises:
        NotImplementedError: always, until the module is written.
        """
        raise NotImplementedError(
            "SemanticChangeNet.build is not implemented. Missing: the weight-shared "
            f"Siamese torch.nn.Module over '{self.config.encoder}' with a binary change "
            f"head and two {self.config.num_classes}-class semantic heads."
        )

    def predict(self, pixel_values_t1: Any, pixel_values_t2: Any) -> dict[str, Any]:
        """Return `{'mask_t1', 'mask_t2', 'change_mask'}` for the symbolic layer.

        Raises:
            NotImplementedError: always, until trained weights exist.
        """
        raise NotImplementedError(
            "SemanticChangeNet.predict is not implemented. Missing: SemanticChangeNet.build "
            "and a checkpoint trained on SECOND. Returning placeholder masks would make "
            "symbolic.measures.change_matrix report transitions that never happened."
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        """Raises:
        NotImplementedError: always, until `build` exists.
        """
        raise NotImplementedError(
            f"SemanticChangeNet.save_checkpoint is not implemented (target {Path(path)})."
        )

    def load_checkpoint(self, path: str | Path) -> SemanticChangeNet:
        """Raises:
        NotImplementedError: always, until `save_checkpoint` exists.
        """
        raise NotImplementedError(
            f"SemanticChangeNet.load_checkpoint is not implemented (source {Path(path)})."
        )
