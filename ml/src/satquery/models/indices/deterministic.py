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
        evidence = self._render_evidence(request, outputs, wanted)

        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=answer,
            evidence=evidence,
            confidence=self._confidence(record),
            params_used={
                "indices": which,
                "classes": sorted(wanted),
                # Names the class the highlight layer marks. The overlay is one class, not
                # all of them, and a legend that cannot say which one is guessing.
                "highlighted_class": self._headline_class(outputs, wanted)
                if (outputs.get("masks") or outputs.get("masks_t1"))
                else None,
            },
            warnings=record.warnings,
            answer_record=record.model_dump(mode="json"),
        )

    # -- evidence ---------------------------------------------------------------------

    def _render_evidence(
        self, request: ToolRequest, outputs: dict, wanted: set[str] | None = None
    ) -> Evidence:
        """Write the maps behind the answer, so the number can be checked against pixels.

        Each artifact is attempted independently. An earlier version wrapped the whole
        block in one try, so a raster with no blue band -- which only stops the *backdrop*
        from rendering -- silently discarded the change map and the highlight too, and the
        feature appeared to work only for files that happened to carry B02.

        The highlight needs no backdrop at all: it is a transparent layer over the live
        imagery, so it is written from the masks alone and survives a missing band.
        """
        from satquery.utils.paths import artifact_dir

        evidence = Evidence()
        out = artifact_dir("serve", "evidence")
        stem = request.request_id[:12]
        single = outputs.get("masks")
        first, second = outputs.get("masks_t1"), outputs.get("masks_t2")

        # -- the highlight, which depends on nothing but the masks ---------------------
        try:
            if single:
                name = self._headline_class(outputs, wanted)
                evidence.highlight_path = write_png(
                    out / f"{stem}_highlight.png",
                    render_mask_alpha(single[name], rgb=CLASS_RGB.get(name, (56, 189, 248))),
                )
            elif first and second:
                name = self._headline_class(outputs, wanted)
                evidence.highlight_path = write_png(
                    out / f"{stem}_highlight.png",
                    render_change_alpha(first[name], second[name]),
                )
        except Exception as exc:  # a missing map must not fail a good answer
            _LOG.warning("highlight rendering failed: %s", exc)

        # -- the tiles, which need a backdrop ------------------------------------------
        try:
            base = self._display_base(request.images[0])
        except Exception as exc:
            _LOG.warning("no displayable backdrop, tiles skipped: %s", exc)
            return evidence

        try:
            if single:
                evidence.overlay_path = write_png(
                    out / f"{stem}_overlay.png", render_mask_overlay(base, single)
                )
            elif first and second:
                name = self._headline_class(outputs, wanted)
                evidence.mask_path = write_png(
                    out / f"{stem}_change.png",
                    render_change_map(base, first[name], second[name]),
                )
                maps = {
                    f"{name}_t1": write_png(
                        out / f"{stem}_t1.png", render_mask_overlay(base, {name: first[name]})
                    )
                }
                try:
                    maps[f"{name}_t2"] = write_png(
                        out / f"{stem}_t2.png",
                        render_mask_overlay(
                            self._display_base(request.images[1]), {name: second[name]}
                        ),
                    )
                except Exception as exc:
                    _LOG.warning("second-date tile skipped: %s", exc)
                evidence.index_maps = maps
        except Exception as exc:
            _LOG.warning("tile rendering failed, answer is unaffected: %s", exc)

        return evidence

    @staticmethod
    def _display_base(ref) -> np.ndarray:
        """An RGB backdrop for the evidence tiles, true colour where that is possible.

        `load_model_input` is the frozen train/serve path and is left untouched -- it
        refuses a stack with no blue band, correctly, because a model must not be fed a
        silently different composite. A picture for a human has no such constraint, so
        when true colour is unavailable this falls back to the standard false-colour
        composite (NIR, red, green) that every remote sensing analyst reads, rather than
        producing nothing.
        """
        from satquery.io.raster import read_raster
        from satquery.models.base import load_model_input
        from satquery.preprocess.optical import stretch_to_uint8

        try:
            return load_model_input(ref)[0]
        except ValueError:
            pass

        array, parsed = read_raster(ref.path)
        names = parsed.band_names
        for combo in (("B08", "B04", "B03"), ("B11", "B08", "B04")):
            if all(band in names for band in combo):
                stack = np.stack([array[names.index(band)] for band in combo])
                return np.ascontiguousarray(stretch_to_uint8(stack).transpose(1, 2, 0))

        # Last resort: the first band, greyscale. Honest, and better than a blank tile.
        single = stretch_to_uint8(array[:1])[0]
        return np.repeat(single[:, :, np.newaxis], 3, axis=2)

    @staticmethod
    def _headline_class(outputs: dict, wanted: set[str] | None = None) -> str:
        """The class the highlight should mark.

        Three rules, in order.

        The query wins. If it named exactly one class, that is what the person asked to
        see, whatever else moved.

        Otherwise the largest change, weighted by how far the index can be trusted. NDWI
        is a reliable water detector and NDVI is decent, but NDBI cannot separate
        impervious surface from dry bare soil and is reported as an upper bound everywhere
        else in this system, so it ranks last here for consistency. On the Alentejo demo
        pair built-up has *more* changed pixels than water -- 3910 against 3187 -- yet it
        is the same physical event read through the weaker index: the reservoir bed is
        bare in October and submerged in March. Marking it as the headline would point at
        the one number the caveats tell the reader not to lean on.

        Alphabetical order, which the first version used, highlighted `built_up` on a
        scene where only the water moved and produced an empty overlay.
        """
        reliability = {"water": 1.0, "vegetation": 0.9, "built_up": 0.35}

        first, second = outputs.get("masks_t1"), outputs.get("masks_t2")
        available = (
            sorted(set(first) & set(second)) if first and second else sorted(outputs["masks"])
        )

        if wanted and len(wanted) == 1:
            only = next(iter(wanted))
            if only in available:
                return only

        def moved(name: str) -> float:
            if first and second:
                count = np.count_nonzero(
                    np.asarray(first[name], dtype=bool) ^ np.asarray(second[name], dtype=bool)
                )
            else:
                count = np.count_nonzero(np.asarray(outputs["masks"][name], dtype=bool))
            return float(count) * reliability.get(name, 0.5)

        return max(available, key=moved)

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
