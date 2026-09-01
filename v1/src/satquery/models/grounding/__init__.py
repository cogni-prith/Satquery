"""Open-vocabulary detection, the second grounding entry in the tool registry."""

from satquery.models.grounding.open_vocab import (
    OpenVocabDetector,
    OpenVocabDetectorConfig,
    build_text_prompt,
)

__all__ = ["OpenVocabDetector", "OpenVocabDetectorConfig", "build_text_prompt"]
