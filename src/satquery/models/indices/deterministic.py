"""The deterministic index tool. Closed-form arithmetic, no weights, no GPU.

The one tool in v2 that answers today, and the reason it is worth having at all: it has
no training distribution to fall outside of. NDWI on a Cartosat scene is the same
arithmetic as NDWI on the Sentinel tiles it was never trained on, because it was never
trained on anything.

It answers water, built-up and vegetation extent on a single image, and the change in
each between two dates. Every number it returns is a pixel count multiplied by the GSD
read from the affine transform, assembled into an `AnswerRecord` by `build_record` and
worded by the template verbalizer -- which never sees the image.

Confidence is only reported when there are genuinely two independent signals to compare:
an optical NDWI mask and a SAR backscatter mask of the same scene. On a single optical
image there is one signal, so no agreement exists and none is claimed.
"""

from __future__ import annotations

import numpy as np

from satquery.models.base import BaseTool, infer_input_config_from_images
from satquery.models.indices.render import (
    CLASS_RGB,
    render_change_alpha,
    render_change_map,
    render_mask_alpha,
    render_mask_overlay,
    write_png,
)
from satquery.preprocess.constants import SAR_WATER_DB_THRESHOLD
from satquery.serve.contracts import Evidence, InputConfig, Modality, ToolRequest, ToolResult
from satquery.symbolic.record import build_record
from satquery.utils.logging import get_logger
from satquery.verbalize.templates import verbalize

__all__ = ["DeterministicIndexTool"]

_LOG = get_logger(__name__)

#: Index name -> (class name in the record, mask builder). Order fixes the reporting order.
_CLASS_FOR_INDEX: dict[str, str] = {"ndwi": "water", "ndbi": "built_up", "ndvi": "vegetation"}

#: Words in a query that narrow the answer to one class. Absent, all computable classes
#: are reported -- an over-broad answer is recoverable, a silently narrowed one is not.
_CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "water": ("water", "lake", "river", "flood", "reservoir", "pond", "inundat"),
    "built_up": ("built", "urban", "building", "settlement", "construct", "city"),
    "vegetation": ("vegetation", "forest", "crop", "green", "tree", "farm"),
}


class DeterministicIndexTool(BaseTool):
    """Spectral and backscatter indices, thresholded into masks and measured."""

    def _run(self, request: ToolRequest) -> ToolResult:
        from satquery.io.raster import read_raster
        from satquery.preprocess.indices import (
            builtup_mask,
            compute_indices,
            sar_water_mask,
            vegetation_mask,
            water_mask,
        )

        wanted = self._requested_classes(request.query)
        which = [name for name, klass in _CLASS_FOR_INDEX.items() if klass in wanted]
        # NDBI is high over open water as well as over concrete -- both are bright in SWIR
        # relative to NIR -- so an unmasked NDBI mask counts a lake as built-up, and a
        # growing lake reads as a construction boom. Standard practice is to subtract the
        # water mask, which means NDWI must be computed whenever NDBI is, whether or not
        # the user asked about water.
        if "ndbi" in which:
            # NDWI to subtract open water, NDVI to subtract vegetation. Both are needed
            # whether or not the user asked about water or vegetation.
            for helper in ("ndwi", "ndvi"):
                if helper not in which:
                    which.insert(0, helper)
        warnings: list[str] = []
        config = infer_input_config_from_images(request)

        def masks_for(index: int) -> tuple[dict[str, np.ndarray], list[str]]:
            """Every computable class mask for one image, plus why any were skipped."""
            ref = request.images[index]
            array, parsed = read_raster(ref.path)
            local = list(parsed.warnings)

            if parsed.modality is Modality.SAR:
                # SAR carries no NIR, so only water is decidable -- from backscatter.
                from satquery.io.modality import polarisation_slots

                co_pol, _ = polarisation_slots(parsed.band_names)
                if co_pol is None:
                    local.append(
                        "SAR image carries no co-polarised channel (VV or HH), so the "
                        "backscatter water mask cannot be built"
                    )
                    return {}, local
                co = array[parsed.band_names.index(co_pol)]
                local.append(
                    f"water from SAR backscatter below {SAR_WATER_DB_THRESHOLD:.0f} dB on "
                    f"{co_pol}; built-up and vegetation need optical bands and are not reported"
                )
                return {"water": sar_water_mask(co)}, local

            maps, index_warnings = compute_indices(array, parsed.band_names, which)
            local.extend(index_warnings)
            built: dict[str, np.ndarray] = {}
            if "ndwi" in maps:
                built["water"] = water_mask(maps["ndwi"])
            if "ndbi" in maps:
                built_up = builtup_mask(maps["ndbi"])
                if "ndwi" in maps:
                    open_water = water_mask(maps["ndwi"])
                    overlap = int(np.count_nonzero(built_up & open_water))
                    built_up = built_up & ~open_water
                    if overlap:
                        local.append(
                            f"{overlap} pixel(s) exceeded both the NDBI and NDWI "
                            "thresholds; open water is bright in SWIR and is excluded "
                            "from built-up rather than double counted"
                        )
                if "ndvi" in maps:
                    built_up = built_up & ~vegetation_mask(maps["ndvi"])
                built["built_up"] = built_up
                if np.count_nonzero(built_up):
                    # The honest caveat, and the reason a learned segmenter is worth
                    # training. NDBI separates "bright in SWIR, dark in NIR" from
                    # everything else; dry bare soil and harvested fields look exactly
                    # like concrete under that test. Masking water and vegetation removes
                    # what is definitionally not built-up, but nothing in the arithmetic
                    # can tell a car park from a ploughed field.
                    local.append(
                        "built-up is an UPPER BOUND: NDBI cannot separate impervious "
                        "surface from dry bare soil or harvested cropland, which have "
                        "the same spectral signature. Treat it as 'built-up or bare "
                        "ground'."
                    )
                built["built_up"] = built_up
            if "ndvi" in maps:
                built["vegetation"] = vegetation_mask(maps["ndvi"])
            return built, local

        gsd_m = request.images[0].gsd_m

        if config is InputConfig.BI_TEMPORAL_PAIR:
            first, first_warnings = masks_for(0)
            second, second_warnings = masks_for(1)
            warnings.extend(first_warnings + second_warnings)
            # NDWI may have been computed only to mask NDBI. Drop what was not asked for.
            first = {name: mask for name, mask in first.items() if name in wanted}
            second = {name: mask for name, mask in second.items() if name in wanted}
            if not (set(first) & set(second)):
                raise ValueError(
                    "no class could be measured at both dates from these bands, so no "
                    f"change can be computed. Date 1 yielded {sorted(first)}, date 2 "
                    f"{sorted(second)}. NDWI needs green and NIR; NDBI needs SWIR and NIR."
                )
            self._require_same_grid(first, second)
            outputs = {"masks_t1": first, "masks_t2": second, "warnings": warnings}
            intent = "change_trend"
        else:
            built, local = masks_for(0)
            warnings.extend(local)
            built = {name: mask for name, mask in built.items() if name in wanted}
            if not built:
                raise ValueError(
                    "none of the requested indices could be computed from this image. "
                    "NDWI needs green and NIR, NDBI needs SWIR and NIR, NDVI needs NIR "
                    "and red. An RGB screenshot carries no NIR band, so no index applies."
                )
            if len(built) > 1:
                # NDWI, NDBI and NDVI are independent thresholds, not a partition of the
                # scene, so a pixel can be over the line for two of them and the
                # percentages can sum past 100. Reported without this note they read as
                # shares of one pie, which is a different and wrong claim.
                warnings.append(
                    "these are independent index thresholds, not a single land-cover "
                    "partition: a pixel can satisfy more than one, so the percentages "
                    "overlap and need not sum to 100"
                )
            outputs = {"masks": built, "warnings": warnings}
            intent = "land_cover"

            if config is InputConfig.CROSS_MODAL_PAIR:
                # Optical and SAR of the same scene: two independent estimates of water,
                # so agreement is real and can be reported as confidence.
                other, other_warnings = masks_for(1)
                warnings.extend(other_warnings)
                if "water" in built and "water" in other:
                    from satquery.symbolic.measures import agreement_iou

                    score = agreement_iou(built["water"], other["water"])
                    if not np.isnan(score):
                        outputs["agreement"] = {"water": score}

        record = build_record(intent, outputs, gsd_m)
        answer = verbalize(record)
        evidence = self._render_evidence(request, outputs)

        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=answer,
            evidence=evidence,
            confidence=self._confidence(record),
            params_used={"indices": which, "classes": sorted(wanted)},
            warnings=record.warnings,
            answer_record=record.model_dump(mode="json"),
        )

    # -- evidence ---------------------------------------------------------------------

    def _render_evidence(self, request: ToolRequest, outputs: dict) -> Evidence:
        """Write the maps behind the answer, so the number can be checked against pixels.

        Rendering failures are swallowed deliberately: a missing picture must never turn a
        correct measurement into a failed request. The answer stands on the record, and
        the maps are corroboration.
        """
        from satquery.models.base import load_model_input
        from satquery.utils.paths import artifact_dir

        evidence = Evidence()
        try:
            out = artifact_dir("serve", "evidence")
            stem = request.request_id[:12]
            base, _ = load_model_input(request.images[0])

            if "masks" in outputs:
                evidence.overlay_path = write_png(
                    out / f"{stem}_overlay.png",
                    render_mask_overlay(base, outputs["masks"]),
                )
                # The class the question was about, as a transparent layer for the canvas.
                first_class = next(iter(outputs["masks"]))
                evidence.highlight_path = write_png(
                    out / f"{stem}_highlight.png",
                    render_mask_alpha(
                        outputs["masks"][first_class],
                        rgb=CLASS_RGB.get(first_class, (56, 189, 248)),
                    ),
                )
            elif "masks_t1" in outputs:
                shared = sorted(set(outputs["masks_t1"]) & set(outputs["masks_t2"]))
                if shared:
                    name = shared[0]
                    evidence.mask_path = write_png(
                        out / f"{stem}_change.png",
                        render_change_map(
                            base, outputs["masks_t1"][name], outputs["masks_t2"][name]
                        ),
                    )
                    evidence.highlight_path = write_png(
                        out / f"{stem}_highlight.png",
                        render_change_alpha(outputs["masks_t1"][name], outputs["masks_t2"][name]),
                    )
                    evidence.index_maps = {
                        f"{name}_t1": write_png(
                            out / f"{stem}_t1.png",
                            render_mask_overlay(base, {name: outputs["masks_t1"][name]}),
                        ),
                        f"{name}_t2": write_png(
                            out / f"{stem}_t2.png",
                            render_mask_overlay(
                                load_model_input(request.images[1])[0],
                                {name: outputs["masks_t2"][name]},
                            ),
                        ),
                    }
        except Exception as exc:  # a missing map must not fail a good answer
            _LOG.warning("evidence rendering failed, answer is unaffected: %s", exc)

        return evidence

    # -- helpers ----------------------------------------------------------------------

    @staticmethod
    def _requested_classes(query: str) -> set[str]:
        """Classes the query asks about, or all three when it names none."""
        text = query.lower()
        named = {
            klass for klass, words in _CLASS_KEYWORDS.items() if any(word in text for word in words)
        }
        return named or set(_CLASS_KEYWORDS)

    @staticmethod
    def _require_same_grid(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
        """Refuse two dates that are not on one grid.

        `area_delta` would refuse anyway, but the message here can name the images rather
        than two anonymous shapes.
        """
        for name in sorted(set(first) & set(second)):
            if first[name].shape != second[name].shape:
                raise ValueError(
                    f"the two dates are not co-registered: {name} is "
                    f"{first[name].shape} at date 1 and {second[name].shape} at date 2. "
                    "Per-pixel change needs both images on one grid."
                )

    @staticmethod
    def _confidence(record) -> float | None:
        """The reported confidence, or None when nothing independent corroborated it.

        Returning a number here when there is only one signal would be exactly the
        softmax-maximum mistake this project refuses to make.
        """
        if not record.agreement_scores:
            return None
        return min(record.agreement_scores.values())
