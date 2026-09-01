"""Land cover segmentation on the reBEN CORINE reference maps.

Does most of the real work. Every quantitative answer -- areas, proportions, change
deltas, the built-up and water extractions -- is a pixel count over this model's output,
so its per-class IoU is the ceiling on the whole system's accuracy.

SegFormer (arXiv 2105.15203) rather than a UNet: its hierarchical encoder handles the
120x120 reBEN patches and 512x512 evaluation tiles without an architecture change, which
matters given the resolution gap to the hidden evaluation set.

The point of training this is the built-up figure. NDBI cannot separate impervious surface
from dry bare soil -- on real Alentejo imagery it reported 39.7% built-up on a cell CORINE
labels as agro-forestry with no urban class at all -- so the deterministic path reports
built-up as an explicit upper bound. A learned head is what turns that bound into a
measurement, and its agreement with NDWI/NDBI is what the confidence signal compares.

Output is a class-id mask that `symbolic/measures.py` consumes directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from satquery.preprocess.constants import LANDCOVER_CLASSES, LANDCOVER_IGNORE_INDEX
from satquery.utils.logging import get_logger

__all__ = ["LandCoverConfig", "LandCoverSegmenter"]

_LOG = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LandCoverConfig:
    """Hyperparameters of the segmentation run."""

    encoder: str = "nvidia/mit-b0"
    in_channels: int = 4
    """OPTICAL_BAND_ORDER_4: B02, B03, B04, B08. The frozen order, never positional."""
    num_classes: int = len(LANDCOVER_CLASSES)
    """CORINE Level-1 plus an ignored `unlabelled` class at index 0. This ordering IS the
    output layer's ordering, so permuting it silently scrambles a reloaded checkpoint."""
    ignore_index: int = LANDCOVER_IGNORE_INDEX
    dice_weight: float = 0.5
    """Weight on the Dice term. Built-up and water are rare and heavily zero-inflated in
    reBEN, so plain cross-entropy's minimiser is close to an empty mask at high pixel
    accuracy. Dice is computed per class and does not care how rare a class is."""


class LandCoverSegmenter:
    """Semantic segmentation producing the class-id masks the symbolic layer measures."""

    def __init__(self, config: LandCoverConfig | None = None) -> None:
        self.config = config or LandCoverConfig()
        self._module: Any = None

    # -- construction -----------------------------------------------------------------

    def build(self) -> Any:
        """Instantiate the encoder and decode head.

        The pretrained stem expects three channels and we feed four. Rather than discard
        the pretrained weights or drop a band, the existing RGB filters are kept and the
        fourth channel is initialised to their mean: near-infrared starts out behaving
        like an average visible band and the optimiser moves it from there. Random
        initialisation for that one channel would inject noise into an otherwise
        pretrained stem.
        """
        import torch
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                self.config.encoder,
                num_labels=self.config.num_classes,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:  # offline, or the hub is unreachable
            _LOG.warning(
                "could not load pretrained %s (%s); starting from scratch", self.config.encoder, exc
            )
            model = SegformerForSemanticSegmentation(
                SegformerConfig(num_labels=self.config.num_classes)
            )

        stem = model.segformer.encoder.patch_embeddings[0].proj
        if stem.in_channels != self.config.in_channels:
            new = torch.nn.Conv2d(
                self.config.in_channels,
                stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
            )
            with torch.no_grad():
                copied = min(stem.in_channels, self.config.in_channels)
                new.weight[:, :copied] = stem.weight[:, :copied]
                for extra in range(copied, self.config.in_channels):
                    new.weight[:, extra] = stem.weight.mean(dim=1)
                if stem.bias is not None and new.bias is not None:
                    new.bias.copy_(stem.bias)
            model.segformer.encoder.patch_embeddings[0].proj = new

        self._module = model
        return model

    # -- loss -------------------------------------------------------------------------

    def loss(self, logits: Any, targets: Any) -> Any:
        """Cross-entropy plus Dice, both ignoring the unlabelled class.

        Cross-entropy alone optimises pixel accuracy, which on reBEN is dominated by
        forest and farmland: a head that never predicts water still scores well. Dice is
        computed per class on the softmax, so a class covering 2% of the pixels carries
        the same weight in that term as one covering 60%.
        """
        import torch
        import torch.nn.functional as F

        # SegFormer emits logits at a quarter resolution; upsample to the label grid
        # rather than downsampling labels, which would quantise away thin features.
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )

        valid_pixels = targets != self.config.ignore_index
        if not bool(valid_pixels.any()):
            # Cross-entropy over zero valid pixels is nan, and a nan loss does not crash a
            # run -- it silently turns every weight to nan and the training continues
            # producing numbers. Return a real zero with a live graph instead.
            return (logits.sum() * 0.0).to(logits.dtype)

        ce = F.cross_entropy(logits, targets, ignore_index=self.config.ignore_index)

        probabilities = logits.softmax(dim=1)
        valid = valid_pixels.unsqueeze(1)
        one_hot = F.one_hot(targets.clamp(min=0), num_classes=self.config.num_classes).permute(
            0, 3, 1, 2
        )
        probabilities = probabilities * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        cardinality = probabilities.sum(dims) + one_hot.sum(dims)
        # Skip absent classes rather than scoring them a perfect 1.0, which would let a
        # batch containing no water report a Dice term as if water were solved.
        present = one_hot.sum(dims) > 0
        if present.any():
            dice = 1.0 - (2.0 * intersection[present] / (cardinality[present] + 1e-6)).mean()
        else:
            dice = torch.zeros((), device=logits.device)

        return ce + self.config.dice_weight * dice

    # -- inference --------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return an `(H, W)` class-id mask for a `(C, H, W)` reflectance stack.

        Raises:
            RuntimeError: No weights are loaded. Returning a plausible mask from an
                untrained head is exactly the fabrication the architecture refuses.
        """
        import torch
        import torch.nn.functional as F

        if self._module is None:
            raise RuntimeError(
                "LandCoverSegmenter has no weights loaded; call build() and load(), or "
                "train it. An untrained head must not return a mask."
            )

        self._module.eval()
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)
        device = next(self._module.parameters()).device
        with torch.no_grad():
            logits = self._module(pixel_values=tensor.to(device)).logits
            logits = F.interpolate(
                logits, size=tensor.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # -- persistence ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        if self._module is None:
            raise RuntimeError("nothing to save; the model has not been built")
        path.mkdir(parents=True, exist_ok=True)
        self._module.save_pretrained(path)
        return path

    def load(self, path: Path) -> Any:
        """Load trained weights.

        Raises:
            FileNotFoundError: The directory holds no model. Failing here is the point --
                the caller must not end up with a randomly initialised head that answers.
        """
        from transformers import SegformerForSemanticSegmentation

        if not path.is_dir():
            raise FileNotFoundError(f"no trained segmenter at {path}")
        self._module = SegformerForSemanticSegmentation.from_pretrained(path)
        return self._module
