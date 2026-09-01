"""Optical plus SAR fusion: the dual encoder cross-checked against the index layer."""

from satquery.models.fusion.dual_encoder import (
    DualEncoderFusion,
    FusionConfig,
    FusionExtractionTool,
)

__all__ = ["DualEncoderFusion", "FusionConfig", "FusionExtractionTool"]
