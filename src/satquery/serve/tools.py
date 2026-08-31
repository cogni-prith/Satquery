"""Tool implementations and their registry bindings.

Importing this module binds every available implementation to its spec in
`models/registry.py`. `ToolRegistry.get` imports it lazily on first use, so the
registry can be read without pulling in model code.

One tool here is fully implemented: `DeterministicIndexTool`. It has no weights, no
GPU dependency and no training distribution to fall outside of, which makes it the
safety net -- on unseen Cartosat-2S or RISAT data it still produces a defensible
answer with a visible map. Every other tool is an honest stub that raises
`NotImplementedError` naming the weights it is missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from satquery.io.raster import read_raster, write_raster
from satquery.models.base import BaseTool, infer_input_config_from_images
from satquery.models.registry import REGISTRY
from satquery.preprocess.constants import (
    NDBI_BUILTUP_THRESHOLD,
    NDVI_VEGETATION_THRESHOLD,
    NDWI_WATER_THRESHOLD,
    SAR_WATER_DB_THRESHOLD,
)
from satquery.preprocess.indices import (
    agreement_score,
    builtup_mask,
    compute_indices,
    sar_water_mask,
    vegetation_mask,
    water_mask,
)
from satquery.serve.contracts import (
    Evidence,
    ImageRef,
    Modality,
    ToolRequest,
    ToolResult,
    ToolSpec,
)
from satquery.utils.logging import get_logger
from satquery.utils.paths import artifact_dir

__all__ = [
    "ChangeDescriptionTool",
    "DeterministicIndexTool",
    "register_builtin_tools",
    "shared_backbone",
]

_LOG = get_logger(__name__)

#: Model config backing the three single-image VLM tools.
_VLM_CONFIG = "model/earthdial_4b_rgb.yaml"
_SHARED_BACKBONE: Any = None


def shared_backbone() -> Any:
    """Return the process-wide EarthDial handle, constructing it on first use.

    One handle, not three. `vlm.vqa`, `vlm.caption` and `vlm.grounding` are separate
    registry entries but they are the same 4B network, and loading it three times would
    need roughly 25 GB on a card that has 8.

    Constructing the handle is free and touches no weights -- `EarthDialBackbone.load`
    does that lazily, on the first real call. So importing this module still costs
    nothing on a machine with no GPU and no checkpoint.
    """
    global _SHARED_BACKBONE
    if _SHARED_BACKBONE is None:
        from satquery.models.vlm.backbone import BackboneConfig, EarthDialBackbone
        from satquery.utils.paths import configs_dir

        _SHARED_BACKBONE = EarthDialBackbone(BackboneConfig.from_yaml(configs_dir() / _VLM_CONFIG))
    return _SHARED_BACKBONE


#: Fraction of pixels above which a mask is described as "substantial" in prose.
_PROSE_THRESHOLD = 0.01


class DeterministicIndexTool(BaseTool):
    """Closed-form spectral and backscatter index extraction. Fully implemented.

    Handles three cases:

    - a single multispectral image: NDWI, NDBI and NDVI, plus their threshold masks;
    - a single SAR image: the backscatter water mask;
    - a co-registered optical and SAR pair: both of the above, with the agreement
      between the optical and SAR water masks reported as the confidence.

    That last number is the project's confidence signal. Two independent physical
    principles -- water absorbs NIR, and smooth water reflects radar away from the
    sensor -- agreeing on the same pixels is strong evidence. Disagreeing is a signal
    to a human that the scene is hard, and both maps are returned so they can judge.
    """

    def __init__(self, spec: ToolSpec | None = None) -> None:
        super().__init__(spec or REGISTRY.get_spec("indices.deterministic"))

    def _run(self, request: ToolRequest) -> ToolResult:
        params = self._effective_params(request)
        warnings: list[str] = []
        index_maps: dict[str, Any] = {}
        summaries: list[str] = []

        optical_water: np.ndarray | None = None
        sar_water: np.ndarray | None = None
        output_dir = artifact_dir("indices", request.request_id)

        for position, image in enumerate(request.images):
            array, ref = read_raster(image.path)
            warnings.extend(f"image {position}: {message}" for message in ref.warnings)

            if ref.modality is Modality.SAR:
                mask, sar_summary, sar_warnings = self._run_sar(array, ref)
                warnings.extend(sar_warnings)
                if mask is not None:
                    sar_water = mask
                    summaries.append(sar_summary)
                    if params["write_maps"]:
                        index_maps["sar_water"] = write_raster(
                            output_dir / f"image{position}_sar_water.tif",
                            mask.astype(np.uint8),
                            crs=ref.crs,
                            transform=ref.transform,
                            band_names=["sar_water"],
                        )
                continue

            maps, optical_summaries, optical_warnings = self._run_optical(
                array, ref, params["indices"]
            )
            warnings.extend(optical_warnings)
            summaries.extend(optical_summaries)
            if "ndwi" in maps:
                optical_water = water_mask(maps["ndwi"])
            if params["write_maps"]:
                for name, index_map in maps.items():
                    index_maps[name] = write_raster(
                        output_dir / f"image{position}_{name}.tif",
                        index_map.astype(np.float32),
                        crs=ref.crs,
                        transform=ref.transform,
                        band_names=[name],
                    )

        confidence: float | None = None
        if optical_water is not None and sar_water is not None:
            if optical_water.shape == sar_water.shape:
                confidence = agreement_score(optical_water, sar_water)
                summaries.append(
                    f"The optical NDWI water mask and the SAR backscatter water mask agree on "
                    f"{confidence:.1%} of the pixels either of them marks as water."
                )
            else:
                warnings.append(
                    f"optical mask {optical_water.shape} and SAR mask {sar_water.shape} differ "
                    "in size; the pair must be resampled onto a common grid before the two "
                    "water masks can be compared, so no agreement confidence is reported"
                )

        if not summaries:
            return ToolResult(
                request_id=request.request_id,
                tool_name=self.spec.name,
                tool_version=self.spec.version,
                params_used=params,
                warnings=warnings,
                error=(
                    "no index could be computed from these inputs. The optical indices need "
                    "named green, red, NIR and SWIR bands and the SAR mask needs a "
                    "co-polarised channel; see the warnings for which were missing."
                ),
            )

        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=" ".join(summaries),
            evidence=Evidence(index_maps=index_maps),
            confidence=confidence,
            params_used=params,
            warnings=warnings,
        )

    # -- helpers ------------------------------------------------------------------

    def _effective_params(self, request: ToolRequest) -> dict[str, Any]:
        """Merge the request's params over this spec's schema defaults."""
        properties = self.spec.param_schema.get("properties", {})
        params = {
            name: definition["default"]
            for name, definition in properties.items()
            if isinstance(definition, dict) and "default" in definition
        }
        params.update(request.params)
        return params

    def _run_optical(
        self,
        array: np.ndarray,
        ref: ImageRef,
        requested: list[str],
    ) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
        """Compute the optical indices available from this stack and describe them."""
        wanted = [name for name in requested if name != "sar_water"]
        maps, warnings = compute_indices(array, ref.band_names, wanted)

        summaries: list[str] = []
        described = {
            "ndwi": ("water", water_mask, NDWI_WATER_THRESHOLD),
            "ndbi": ("built-up land", builtup_mask, NDBI_BUILTUP_THRESHOLD),
            "ndvi": ("vegetation", vegetation_mask, NDVI_VEGETATION_THRESHOLD),
        }
        for name, index_map in maps.items():
            label, mask_fn, threshold = described[name]
            fraction = float(np.count_nonzero(mask_fn(index_map)) / index_map.size)
            summaries.append(
                f"{name.upper()} covers {fraction:.1%} of the scene as {label} "
                f"(threshold {threshold:+g})."
                if fraction >= _PROSE_THRESHOLD
                else f"{name.upper()} finds effectively no {label} ({fraction:.2%} of pixels)."
            )
        return maps, summaries, warnings

    def _run_sar(
        self, array: np.ndarray, ref: ImageRef
    ) -> tuple[np.ndarray | None, str, list[str]]:
        """Threshold the co-polarised SAR channel into a water mask."""
        from satquery.io.modality import polarisation_slots

        co_pol, _ = polarisation_slots(ref.band_names)
        if co_pol is None:
            return (
                None,
                "",
                [
                    f"no co-polarised SAR band found among {ref.band_names}; the backscatter "
                    "water mask needs VV or HH"
                ],
            )

        band = array[ref.band_names.index(co_pol)]
        mask = sar_water_mask(band)
        fraction = float(np.count_nonzero(mask) / mask.size)
        summary = (
            f"SAR {co_pol} backscatter below {SAR_WATER_DB_THRESHOLD:g} dB covers "
            f"{fraction:.1%} of the scene, indicating open water."
        )
        return mask, summary, []


class ChangeDescriptionTool(BaseTool):
    """Free-form bi-temporal change description. Stub.

    The generative half of the change doctrine: this describes what changed in prose,
    while `change.vqa_head` produces the scored closed-set CDVQA answer. The router
    calls both and reports both.
    """

    def __init__(self, spec: ToolSpec | None = None) -> None:
        super().__init__(spec or REGISTRY.get_spec("vlm.change_description"))

    def _run(self, request: ToolRequest) -> ToolResult:
        raise NotImplementedError(
            "vlm.change_description is not implemented. Missing: the EarthDial-4B "
            "non-optical checkpoint and the LoRA adapter fine-tuned on the bi-temporal "
            "split of the data mix. No adapter exists under the artifact root, so there is "
            "nothing to run; returning invented prose would silently corrupt the CDVQA "
            f"comparison for input config {infer_input_config_from_images(request).value}."
        )


def register_builtin_tools() -> None:
    """Bind every available implementation to its registered spec.

    Called for its side effects when `satquery.serve.tools` is imported. Binding is
    lazy: each factory is only invoked when a tool is actually requested, so importing
    this module never loads a model.
    """
    REGISTRY.bind("indices.deterministic", DeterministicIndexTool)
    REGISTRY.bind("vlm.change_description", ChangeDescriptionTool)

    from satquery.models.change.mask import ChangeMaskTool
    from satquery.models.change.siamese import ChangeVqaTool
    from satquery.models.fusion.dual_encoder import FusionExtractionTool
    from satquery.models.vlm.tasks import CaptionTool, GroundingTool, VqaTool

    # All three share one backbone handle; see `shared_backbone`.
    REGISTRY.bind("vlm.vqa", lambda: VqaTool(backbone=shared_backbone()))
    REGISTRY.bind("vlm.caption", lambda: CaptionTool(backbone=shared_backbone()))
    REGISTRY.bind("vlm.grounding", lambda: GroundingTool(backbone=shared_backbone()))
    REGISTRY.bind("change.vqa_head", ChangeVqaTool)  # loads its checkpoint lazily
    REGISTRY.bind("change.mask", ChangeMaskTool)
    REGISTRY.bind("fusion.extraction", FusionExtractionTool)

    # detector.openvocab has no BaseTool wrapper yet: OpenVocabDetector is a plain
    # detector class, not a tool. Left unbound deliberately, so REGISTRY.get for it
    # raises NotImplementedError naming the missing wrapper rather than half-working.
    _LOG.debug("registered %d tool specs, %d bound", len(REGISTRY), 8)


register_builtin_tools()
