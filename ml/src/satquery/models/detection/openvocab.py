"""Open-vocabulary detection for grounding, on the VRSBench referring split.

Grounding-DINO (arXiv 2303.05499) or LAE-DINO fine-tuned, or a simpler class-conditioned
detector paired with `symbolic/predicates.py`. The second option is genuinely competitive
here: many VRSBench referring phrases are "the largest building" or "the lake in the
north-east", which the predicates resolve exactly once boxes exist.

Scored acc@0.5 on horizontal boxes. Output boxes feed the symbolic layer, which is where
"largest" and "northernmost" are decided -- not by the detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["OpenVocabDetector", "OpenVocabDetectorConfig"]


@dataclass(frozen=True, slots=True)
class OpenVocabDetectorConfig:
    """Hyperparameters of the detection run."""

    backbone: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    max_boxes: int = 50
    image_size: int = 512


class OpenVocabDetector:
    """Phrase-conditioned box detection."""

    def __init__(self, config: OpenVocabDetectorConfig) -> None:
        self.config = config
        self._module: Any = None

    def build(self) -> Any:
        """Raises:
        NotImplementedError: always, until the detector is wired.
        """
        raise NotImplementedError(
            "OpenVocabDetector.build is not implemented. Missing: "
            f"'{self.config.backbone}' weights (~700 MB, not downloaded automatically) "
            "and the transformers wrapper around them."
        )

    def detect(self, pixel_values: Any, phrase: str) -> list[dict[str, Any]]:
        """Boxes in absolute pixel xyxy, with label and score.

        Absolute pixels, not normalised: `BoundingBox` in the contract is defined that way
        because that is how grounding is scored, and a normalised box passed through
        unconverted lands inside the top-left pixel while looking well formed.

        Raises:
            NotImplementedError: always, until weights exist.
        """
        raise NotImplementedError(
            "OpenVocabDetector.detect is not implemented. Missing: OpenVocabDetector.build "
            f"and fine-tuned weights (phrase was {phrase!r}). Returning invented boxes "
            "would give symbolic.predicates real-looking geometry to reason over."
        )
