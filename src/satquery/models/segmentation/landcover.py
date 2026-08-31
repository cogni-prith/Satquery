"""Land cover segmentation on the BigEarthNet.txt CLC reference maps.

Does most of the real work. Every quantitative answer -- areas, proportions, change
deltas, the built-up and water extractions -- is a pixel count over this model's output,
so its per-class IoU is the ceiling on the whole system's accuracy.

UNet or SegFormer (arXiv 2105.15203). SegFormer is preferred: its hierarchical encoder
handles the 120x120 reBEN patches and the 512x512 evaluation tiles without a change of
architecture, which matters given the 20x resolution gap to the hidden set.

Output is a class-id mask that `symbolic/measures.py` consumes directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["LandCoverConfig", "LandCoverSegmenter"]


@dataclass(frozen=True, slots=True)
class LandCoverConfig:
    """Hyperparameters of the segmentation run."""

    encoder: str = "nvidia/mit-b0"
    in_channels: int = 4
    """OPTICAL_BAND_ORDER_4: B02, B03, B04, B08. The frozen order, never positional."""
    num_classes: int = 6
    """CLC Level-1 plus background. The class ids are the output-layer ordering, so a
    permutation silently scrambles a reloaded checkpoint."""
    ignore_index: int = 255
    loss: str = "ce_dice"
    """Cross-entropy plus Dice. Built-up and water are rare and heavily zero-inflated in
    reBEN, so plain CE's minimiser is an empty mask at ~96% pixel accuracy."""


class LandCoverSegmenter:
    """Semantic segmentation producing the class-id masks the symbolic layer measures."""

    def __init__(self, config: LandCoverConfig) -> None:
        self.config = config
        self._module: Any = None

    def build(self) -> Any:
        """Instantiate the encoder and decode head.

        Raises:
            NotImplementedError: always, until the module is written.
        """
        raise NotImplementedError(
            "LandCoverSegmenter.build is not implemented. Missing: the torch.nn.Module "
            f"wrapping '{self.config.encoder}' for {self.config.in_channels}-channel input "
            f"and {self.config.num_classes} classes, plus the 'gpu' extras."
        )

    def predict_mask(self, pixel_values: Any) -> Any:
        """Return a class-id mask, channels-free, same spatial extent as the input.

        Raises:
            NotImplementedError: always, until trained weights exist. Returning a
                plausible mask here would put fabricated areas into every answer.
        """
        raise NotImplementedError(
            "LandCoverSegmenter.predict_mask is not implemented. Missing: "
            "LandCoverSegmenter.build and a trained checkpoint. Every area, proportion "
            "and change delta is a pixel count over this output, so a placeholder mask "
            "would fabricate every number downstream."
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        """Raises:
        NotImplementedError: always, until `build` exists.
        """
        raise NotImplementedError(
            f"LandCoverSegmenter.save_checkpoint is not implemented (target {Path(path)})."
        )

    def load_checkpoint(self, path: str | Path) -> LandCoverSegmenter:
        """Raises:
        NotImplementedError: always, until `save_checkpoint` exists.
        """
        raise NotImplementedError(
            f"LandCoverSegmenter.load_checkpoint is not implemented (source {Path(path)}). "
            "Missing: trained weights and the constants-fingerprint check."
        )
