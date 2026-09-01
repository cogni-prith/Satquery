"""The land-cover tool: learned masks, measured against the deterministic indices.

This is where the project's confidence signal stops being a placeholder. Every other tool
here produces one estimate and has nothing to check it against, so it reports no
agreement. This one produces two by construction -- a trained segmenter and a closed-form
spectral index, sharing no assumptions and no training data -- and their IoU is a real
measurement of how much the answer can be trusted on imagery neither has seen.

It also retires a caveat. NDBI cannot separate impervious surface from dry bare soil, so
the deterministic path reports built-up as an explicit upper bound. The segmenter scores
0.60 IoU on built-up over the reBEN validation split, so where it is available the
built-up figure becomes a measurement and the bound is no longer the honest ceiling.

Where the two disagree, that disagreement is reported rather than resolved. A low
agreement score is the most useful thing this system can say about an unfamiliar sensor.

One disagreement is systematic and is called out by name. The segmenter is trained on
CORINE, which labels a *designated* land cover: a reservoir is a water body whether or not
there is water in it today. NDWI measures the surface that is actually wet. On the
Alentejo demo patch during the 2017 drought, CORINE says 55.6% water, the segmenter
predicts 34.8% and NDWI observes 18.7% -- and the segmenter agrees with CORINE (IoU 0.61)
far better than NDWI does (0.34), because it learned the label it was given. For "how much
water is there now" the index is the right instrument, and the tool says so rather than
letting the reader assume the learned number supersedes it.
"""

from __future__ import annotations

import numpy as np

from satquery.models.base import BaseTool
from satquery.models.indices.render import (
    CLASS_RGB,
    render_mask_alpha,
    render_mask_overlay,
    write_png,
)
from satquery.preprocess.constants import LANDCOVER_CLASSES
from satquery.serve.contracts import Evidence, InputConfig, ToolRequest, ToolResult
from satquery.symbolic.record import build_record
from satquery.utils.logging import get_logger
from satquery.verbalize.templates import verbalize

__all__ = ["LandCoverTool"]

_LOG = get_logger(__name__)

#: Bands the segmenter was trained on, in the frozen order. Never positional.
BANDS = ("B02", "B03", "B04", "B08")

#: Learned class -> the index that provides its independent second opinion.
#:
#: Only these two have one. Agriculture and wetland have no closed-form index that means
#: the same thing, so no agreement is computed for them and none is claimed -- an
#: invented second opinion would make the confidence signal decorative.
INDEX_FOR_CLASS: dict[str, str] = {"water": "ndwi", "built_up": "ndbi"}


class LandCoverTool(BaseTool):
    """Segment land cover, and score the result against the spectral indices."""

    def __init__(self, spec, segmenter) -> None:
        super().__init__(spec)
        self.segmenter = segmenter

    def _run(self, request: ToolRequest) -> ToolResult:
        from satquery.models.base import infer_input_config_from_images

        config = infer_input_config_from_images(request)
        gsd_m = request.images[0].gsd_m
        warnings: list[str] = []

        first_masks, first_agreement, first_warn, base = self._analyse(request.images[0])
        warnings.extend(first_warn)

        if config is InputConfig.BI_TEMPORAL_PAIR:
            second_masks, _, second_warn, _ = self._analyse(request.images[1])
            warnings.extend(second_warn)
            shared = sorted(set(first_masks) & set(second_masks))
            if not shared:
                raise ValueError(
                    "no class was segmented at both dates, so no change can be computed"
                )
            outputs = {
                "masks_t1": {k: first_masks[k] for k in shared},
                "masks_t2": {k: second_masks[k] for k in shared},
                "agreement": first_agreement,
                "warnings": warnings,
            }
            intent = "change_trend"
        else:
            outputs = {"masks": first_masks, "agreement": first_agreement, "warnings": warnings}
            intent = "land_cover"

        record = build_record(intent, outputs, gsd_m)

        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=verbalize(record),
            evidence=self._render(request, base, outputs),
            # The worst agreement across classes, not the best: the confidence of an
            # answer is the confidence of its weakest supported claim.
            confidence=min(first_agreement.values()) if first_agreement else None,
            params_used={
                "classes": sorted(first_masks),
                "agreement_classes": sorted(first_agreement),
                "encoder": self.segmenter.config.encoder,
            },
            warnings=record.warnings,
            answer_record=record.model_dump(mode="json"),
        )

    # -- analysis ---------------------------------------------------------------------

    def _analyse(self, ref) -> tuple[dict, dict, list[str], np.ndarray]:
        """Segment one image and score each class against its index, where one exists."""
        from satquery.io.raster import read_raster
        from satquery.preprocess.indices import builtup_mask, compute_indices, water_mask
        from satquery.symbolic.measures import agreement_iou

        array, parsed = read_raster(ref.path)
        names = parsed.band_names
        missing = [band for band in BANDS if band not in names]
        if missing:
            raise ValueError(
                f"the segmenter needs {list(BANDS)} and this raster is missing {missing}. "
                f"Available: {names}."
            )

        stack = np.stack([array[names.index(band)] for band in BANDS]).astype(np.float32)
        # Sentinel-2 L2A is reflectance x 10000, which is how the model was trained.
        if stack.max() > 1.5:
            stack = stack / 10000.0

        predicted = self.segmenter.predict(stack)
        masks = {
            name: (predicted == index)
            for index, name in enumerate(LANDCOVER_CLASSES)
            if index != 0 and bool((predicted == index).any())
        }

        warnings = list(parsed.warnings)
        agreement: dict[str, float] = {}
        index_maps, index_warnings = compute_indices(array, names, ("ndwi", "ndbi"))
        warnings.extend(index_warnings)

        for klass, index_name in INDEX_FOR_CLASS.items():
            if klass not in masks or index_name not in index_maps:
                continue
            builder = water_mask if index_name == "ndwi" else builtup_mask
            score = agreement_iou(masks[klass], builder(index_maps[index_name]))
            if not np.isnan(score):
                agreement[klass] = score
                if score < 0.7 and klass == "water":
                    learned_share = float(masks[klass].mean())
                    index_share = float(builder(index_maps[index_name]).mean())
                    if learned_share > index_share * 1.3:
                        # Not noise. Measured on the Alentejo demo patch: CORINE labels
                        # 55.6% water, the segmenter predicts 34.8%, NDWI observes 18.7%.
                        warnings.append(
                            f"water: the learned mask ({learned_share * 100:.0f}%) exceeds "
                            f"NDWI ({index_share * 100:.0f}%). These measure different "
                            "things. The segmenter is trained on CORINE, which labels a "
                            "designated water body -- a reservoir stays a water body when "
                            "drawn down -- while NDWI measures the surface actually wet on "
                            "this date. For 'how much water is there now', trust NDWI; the "
                            "learned figure is closer to the mapped extent"
                        )
                    else:
                        warnings.append(
                            f"{klass}: the learned mask and {index_name.upper()} agree on "
                            f"only {score * 100:.0f}% of the union; check the map before "
                            "using the figure"
                        )
                elif score < 0.45:
                    warnings.append(
                        f"{klass}: the learned mask and {index_name.upper()} agree on only "
                        f"{score * 100:.0f}% of the union. Two independent estimates "
                        "disagreeing this much means the figure should be checked against "
                        "the map before it is used"
                    )

        return masks, agreement, warnings, stack

    # -- evidence ---------------------------------------------------------------------

    def _render(self, request: ToolRequest, base: np.ndarray, outputs: dict) -> Evidence:
        """Draw the segmentation. A failed render must never fail a good measurement."""
        from satquery.preprocess.optical import stretch_to_uint8
        from satquery.utils.paths import artifact_dir

        evidence = Evidence()
        try:
            out = artifact_dir("serve", "evidence")
            stem = request.request_id[:12]
            display = np.ascontiguousarray(stretch_to_uint8(base[[2, 1, 0]]).transpose(1, 2, 0))
            masks = outputs.get("masks") or outputs.get("masks_t1") or {}
            paintable = {k: v for k, v in masks.items() if k in CLASS_RGB}
            if paintable:
                evidence.overlay_path = write_png(
                    out / f"{stem}_overlay.png", render_mask_overlay(display, paintable)
                )
                headline = max(paintable, key=lambda k: int(paintable[k].sum()))
                evidence.highlight_path = write_png(
                    out / f"{stem}_highlight.png",
                    render_mask_alpha(paintable[headline], rgb=CLASS_RGB[headline]),
                )
        except Exception as exc:
            _LOG.warning("evidence rendering failed, measurement is unaffected: %s", exc)
        return evidence
