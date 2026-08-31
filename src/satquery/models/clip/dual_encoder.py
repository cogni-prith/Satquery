"""CLIP-style dual encoder on BigEarthNet.txt. The remote-sensing adaptation artifact.

Trained FIRST, because it is what satisfies the problem statement's mandate that "at
least one visual or vision-language component must be fine-tuned or otherwise adapted".
Contrastive, no generation, and cheap enough to finish before anything else is ready.

Follows RemoteCLIP (arXiv 2306.11029) and GeoRSCLIP (arXiv 2306.11300): a frozen or
lightly-tuned image tower and text tower pulled into one embedding space by InfoNCE over
image-caption pairs. On BigEarthNet.txt that gives open-vocabulary land cover matching
AND a shared Sentinel-1 / Sentinel-2 space, which is what the fusion path reads.

Zero-shot retrieval before and after adaptation is the proof artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["DualEncoderCLIP", "DualEncoderConfig"]


@dataclass(frozen=True, slots=True)
class DualEncoderConfig:
    """Hyperparameters of the contrastive adaptation run."""

    image_encoder: str = "timm/vit_base_patch16_224.openai_clip"
    text_encoder: str = "openai/clip-vit-base-patch16"
    embed_dim: int = 512
    temperature: float = 0.07
    """InfoNCE temperature. Learned in CLIP; fixed here so runs stay comparable."""
    freeze_text_tower: bool = True
    """The text side already speaks English. The image side is what has to learn
    remote sensing, and freezing text halves the memory on an 8 GB card."""
    image_size: int = 224
    in_channels: int = 3
    """Three, because the towers start from ImageNet/CLIP weights. A 10-band variant
    needs a re-initialised patch embedding, which is a separate decision."""


class DualEncoderCLIP:
    """Image and text towers projected into one embedding space."""

    def __init__(self, config: DualEncoderConfig) -> None:
        self.config = config
        self._module: Any = None

    def build(self) -> Any:
        """Instantiate both towers and the projection heads.

        Raises:
            NotImplementedError: always, until the towers are written.
        """
        raise NotImplementedError(
            "DualEncoderCLIP.build is not implemented. Missing: the torch.nn.Module "
            f"joining image tower '{self.config.image_encoder}' and text tower "
            f"'{self.config.text_encoder}' through {self.config.embed_dim}-d projections "
            "with an InfoNCE head, plus the optional 'gpu' extras torch and timm."
        )

    def encode_image(self, pixel_values: Any) -> Any:
        """L2-normalised image embeddings.

        Raises:
            NotImplementedError: always, until `build` exists.
        """
        raise NotImplementedError(
            "DualEncoderCLIP.encode_image is not implemented. Missing: "
            "DualEncoderCLIP.build and trained weights."
        )

    def encode_text(self, input_ids: Any) -> Any:
        """L2-normalised text embeddings.

        Raises:
            NotImplementedError: always, until `build` exists.
        """
        raise NotImplementedError(
            "DualEncoderCLIP.encode_text is not implemented. Missing: "
            "DualEncoderCLIP.build and trained weights."
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        """Write weights, config and the constants fingerprint.

        Raises:
            NotImplementedError: always, until `build` exists.
        """
        raise NotImplementedError(
            f"DualEncoderCLIP.save_checkpoint is not implemented (target {Path(path)}). "
            "Missing: a built module to take a state_dict from."
        )

    def load_checkpoint(self, path: str | Path) -> DualEncoderCLIP:
        """Restore weights, refusing on a constants-fingerprint mismatch.

        Raises:
            NotImplementedError: always, until `save_checkpoint` exists.
        """
        raise NotImplementedError(
            f"DualEncoderCLIP.load_checkpoint is not implemented (source {Path(path)}). "
            "Missing: the checkpoint format, the trained weights, and the fingerprint "
            "check that must refuse a checkpoint whose preprocessing has drifted."
        )
