"""Open-vocabulary detector: the second grounding entry in the registry.

Two grounding tools exist because the two failure modes are different. The fine-tuned
VLM box head (`models/vlm/tasks.py::GroundingTool`) handles descriptive and relational
referring expressions. This tool handles the other case: the query names a concrete
object class ("ships", "aircraft"), where a class-conditioned detector is both more
accurate and far cheaper than generation. The router's stage-one gate picks between
them from the declared specs; see `detector.openvocab` in `models/registry.py`.

Intended implementation, in preference order:

1. **LAE-DINO** -- "Locate Anything on Earth: Advancing Open-Vocabulary Object
   Detection for Remote Sensing Community" (Pan et al., AAAI 2025, arXiv:2408.09110).
   Remote-sensing-native, trained on LAE-1M, so it is the right first choice for
   aerial and satellite imagery.
2. **Grounding-DINO** -- "Grounding DINO: Marrying DINO with Grounded Pre-Training for
   Open-Set Object Detection" (Liu et al., ECCV 2024, arXiv:2303.05499),
   github.com/IDEA-Research/GroundingDINO. Natural-image pretrained fallback, used if
   LAE-DINO checkpoints are unavailable. Both take the same period-separated text
   prompt, which is why `build_text_prompt` below serves either.

No weights are downloaded by this repo. Torch is imported inside function bodies only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from satquery.serve.contracts import BoundingBox
from satquery.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np

__all__ = ["OpenVocabDetector", "OpenVocabDetectorConfig", "build_text_prompt"]

_LOG = get_logger(__name__)

#: Separator Grounding-DINO and LAE-DINO expect between class names in a text prompt.
_PROMPT_SEPARATOR = ". "
#: Terminator on the final class name. Both models tokenise the prompt on periods.
_PROMPT_TERMINATOR = "."


def build_text_prompt(class_names: list[str]) -> str:
    """Join class names into the period-separated prompt these detectors expect.

    Grounding-DINO tokenises its text prompt on periods and treats each segment as one
    open-vocabulary class, so `["ship", "aircraft"]` becomes `"ship. aircraft."`. Names
    are lowercased and stripped, surrounding periods are removed so a caller passing
    `"ship."` does not produce `"ship.."`, and duplicates are dropped while preserving
    the caller's order because the detector's output logits are aligned to prompt order.

    Args:
        class_names: Object class names, in the order the caller wants them scored.

    Returns:
        The prompt string, e.g. `"ship. aircraft."`. Empty when nothing usable remains.

    Raises:
        TypeError: `class_names` is a bare string rather than a list of names.
    """
    if isinstance(class_names, str):
        raise TypeError("class_names must be a list of names, not a single string")

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in class_names:
        name = " ".join(str(raw).strip().strip(".").split()).lower()
        if not name or name in seen:
            continue
        seen.add(name)
        cleaned.append(name)

    if not cleaned:
        return ""
    return _PROMPT_SEPARATOR.join(cleaned) + _PROMPT_TERMINATOR


@dataclass(slots=True)
class OpenVocabDetectorConfig:
    """Everything needed to instantiate one open-vocabulary detector checkpoint."""

    hf_repo_id: str = "IDEA-Research/grounding-dino-base"
    checkpoint_path: Path | None = None
    config_path: Path | None = None
    device: str = "cuda"
    dtype: str = "float32"
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    max_boxes: int = 100

    @property
    def uses_local_checkpoint(self) -> bool:
        """True when a local checkpoint is configured instead of a Hub repo id."""
        return self.checkpoint_path is not None


class OpenVocabDetector:
    """Lazily-loaded open-vocabulary detector behind the `detector.openvocab` spec.

    Construction is free; weights are touched only by :meth:`load`, so importing this
    module on a CPU-only machine costs nothing.
    """

    def __init__(self, config: OpenVocabDetectorConfig | None = None) -> None:
        self.config = config or OpenVocabDetectorConfig()
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """True once :meth:`load` has completed successfully."""
        return self._model is not None

    def load(self) -> None:
        """Materialise the detector weights.

        Intended implementation: LAE-DINO weights when available, otherwise
        `transformers.AutoModelForZeroShotObjectDetection` plus `AutoProcessor` on the
        Grounding-DINO repo id.
        """
        source = self.config.checkpoint_path or self.config.hf_repo_id
        raise NotImplementedError(
            "OpenVocabDetector.load is not implemented. Missing: open-vocabulary detector "
            f"weights at {source} -- LAE-DINO (arXiv:2408.09110) preferred, Grounding-DINO "
            "(arXiv:2303.05499) as fallback -- which this repo never downloads, plus the "
            "torch/transformers 'gpu' extra."
        )

    def detect(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[BoundingBox]:
        """Detect every prompted class in one preprocessed image.

        Args:
            image: Rendered uint8 HWC array, already through `preprocess/optical.py`
                (or `preprocess/sar.py`, though the spec accepts optical only).
            text_prompt: Period-separated class names, built by :func:`build_text_prompt`.
            box_threshold: Minimum box logit. Defaults to `config.box_threshold`.
            text_threshold: Minimum token-to-class score. Defaults to
                `config.text_threshold`.

        Returns:
            Boxes in absolute pixels, `xyxy`, with `image_index=0` -- the spec accepts
            only single-image input, so there is never another image to index. Detector
            output is normalised `cxcywh` and must be converted before a `BoundingBox`
            is constructed; `BoundingBox` rejects degenerate boxes.
        """
        raise NotImplementedError(
            "OpenVocabDetector.detect is not implemented. Missing: loaded detector weights "
            f"({self.config.checkpoint_path or self.config.hf_repo_id}) and the cxcywh-normalised "
            "to xyxy-pixel conversion. Returning fabricated boxes here would silently pass a "
            "smoke test and destroy an acc@0.5 grounding eval."
        )

    def __repr__(self) -> str:
        source = self.config.checkpoint_path or self.config.hf_repo_id
        state = "loaded" if self.is_loaded else "not loaded"
        return f"OpenVocabDetector(source={str(source)!r}, {state})"
