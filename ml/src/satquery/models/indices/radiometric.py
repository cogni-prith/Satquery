"""Radiometric change detection: the tool that answers when no index can.

Every spectral tool here needs near-infrared, so a plain RGB pair -- a screenshot, a JPEG,
an aerial tile -- is refused by all of them. That refusal is correct and leaves a person
holding two pictures of the same place with nothing.

This tool differences those pictures. It reports the share of the scene whose colour
moved, in ground area when a transform is present, and draws the changed region on the
imagery. It never names a class, because it cannot: new concrete, bare soil, a wet road
and a different sun angle are the same event to it.

The weakness is the contract. Every answer carries it, the tool description carries it,
and the record's warnings carry it. A caller that strips them turns a screening result
into a land-cover claim, which is the fabrication the whole architecture is built to
avoid.
"""

from __future__ import annotations

import numpy as np

from satquery.models.base import BaseTool
from satquery.models.indices.render import render_mask_alpha, render_mask_overlay, write_png
from satquery.preprocess.constants import RGB_CHANGE_MIN_COMPONENT_PX
from satquery.preprocess.rgb_change import excess_green, rgb_change_mask
from satquery.serve.contracts import Evidence, ToolRequest, ToolResult
from satquery.symbolic.record import AnswerRecord, Fact
from satquery.utils.logging import get_logger
from satquery.verbalize.templates import verbalize

__all__ = ["RadiometricChangeTool"]

_LOG = get_logger(__name__)
#: The changed region is drawn in the same red the spectral path uses for "gained".
CHANGE_RGB = (255, 62, 78)


class RadiometricChangeTool(BaseTool):
    """Difference two co-registered images and measure where they disagree."""

    @staticmethod
    def _read_comparable(ref) -> tuple[np.ndarray, list[str]]:
        """Read an image WITHOUT a per-image contrast stretch.

        `load_model_input` percentile-stretches each raster into its own range, which is
        right for feeding a model and wrong here: it is a different nonlinear map for each
        image, applied before they are compared, and it clips each one's outliers at a
        different place. Routing this tool through it reported 96% of a scene as changed
        where reading the raw bands gives 30%.

        The two images must meet the difference operator having been through identical
        arithmetic, so this reads raw values and leaves all normalisation to
        `rgb_change`, which applies the same treatment to both.
        """
        from satquery.io.raster import read_raster

        array, parsed = read_raster(ref.path)
        if array.shape[0] < 3:
            band = array[0].astype(np.float64)
            return np.repeat(band[:, :, np.newaxis], 3, axis=2), list(parsed.warnings)
        return (
            np.ascontiguousarray(array[:3].astype(np.float64).transpose(1, 2, 0)),
            list(parsed.warnings),
        )

    def _run(self, request: ToolRequest) -> ToolResult:
        if len(request.images) != 2:
            raise ValueError(
                f"radiometric change needs exactly two images, got {len(request.images)}"
            )

        # None means "find it from this pair", which is the default and the right
        # behaviour. Defaulting to the constant here silently disabled the adaptive
        # threshold and put every scene back at 90% changed.
        raw_threshold = request.params.get("threshold")
        threshold = float(raw_threshold) if raw_threshold is not None else None
        min_px = int(request.params.get("min_component_px", RGB_CHANGE_MIN_COMPONENT_PX))

        first, warn_a = self._read_comparable(request.images[0])
        second, warn_b = self._read_comparable(request.images[1])
        second, warn_fit = self._fit_to_grid(second, first, request.images)

        mask, change_warnings = rgb_change_mask(
            first, second, threshold=threshold, min_component_px=min_px
        )
        warnings = list(dict.fromkeys([*warn_a, *warn_b, *warn_fit, *change_warnings]))

        share = float(mask.mean())
        gsd_m = request.images[0].gsd_m
        facts = [
            Fact(
                key="changed_share",
                value=share,
                unit="fraction",
                provenance="preprocess.rgb_change.rgb_change_mask",
            )
        ]
        if gsd_m and gsd_m > 0:
            facts.append(
                Fact(
                    key="changed_area",
                    value=float(np.count_nonzero(mask)) * gsd_m**2,
                    unit="m2",
                    provenance="preprocess.rgb_change.rgb_change_mask",
                )
            )

        record = AnswerRecord(
            intent="radiometric_change",
            facts=facts,
            class_proportions={"changed": share},
            warnings=[*warnings, *self._direction_note(first, second, mask)],
        )
        record.validate_provenance()

        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=verbalize(record),
            evidence=self._render(request, first, mask),
            # No second, independent estimate exists here, so no agreement can be
            # computed and none is claimed.
            confidence=None,
            params_used={
                "threshold": threshold if threshold is not None else "otsu (per pair)",
                "min_component_px": min_px,
            },
            warnings=record.warnings,
            answer_record=record.model_dump(mode="json"),
        )

    # -- helpers ----------------------------------------------------------------------

    @staticmethod
    def _fit_to_grid(
        moving: np.ndarray, reference: np.ndarray, images
    ) -> tuple[np.ndarray, list[str]]:
        """Put the second date on the first's grid, or refuse to guess.

        Two screenshots of one place, cropped by hand, arrive at 505x527 and 547x552 and
        the difference operator cannot touch them. Refusing was correct but useless: that
        is the imagery people actually have, and "crop them yourself first" hands the job
        back to the person who came here to avoid it.

        Where neither raster is georeferenced there is no measured alignment to preserve,
        so scaling the second onto the first's frame assumes the two cover the same
        extent. That assumption is a guess. It is a *stated* guess -- every answer carries
        the warning -- and residual misregistration lands in the mask as change, which is
        why the note names that consequence rather than just the resize.

        Where either raster does carry a CRS, the misalignment is measurable and a
        frame-fit would silently discard real georeferencing. That still refuses.
        """
        if moving.shape == reference.shape:
            return moving, []

        if any(getattr(ref, "crs", None) for ref in images):
            raise ValueError(
                f"the two images are not on one grid: {reference.shape[:2]} and "
                f"{moving.shape[:2]}, and at least one carries a CRS. Reprojecting them "
                "onto a common grid is a georeferencing job this tool will not guess at; "
                "warp them to matching bounds and resolution first."
            )

        from PIL import Image

        height, width = reference.shape[:2]
        # Bilinear, not nearest: nearest resampling puts aliasing noise into the very
        # difference this tool thresholds, and reports it as changed scene.
        fitted = np.stack(
            [
                np.asarray(
                    Image.fromarray(moving[:, :, band].astype(np.float32), mode="F").resize(
                        (width, height), Image.BILINEAR
                    ),
                    dtype=np.float64,
                )
                for band in range(moving.shape[2])
            ],
            axis=-1,
        )
        return fitted, [
            f"the two images are not on one grid ({reference.shape[:2]} and "
            f"{moving.shape[:2]}) and neither is georeferenced, so date 2 was scaled onto "
            "date 1's frame. This assumes both cover the same extent; where they do not, "
            "the offset appears in the result as changed scene"
        ]

    @staticmethod
    def _direction_note(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> list[str]:
        """One weak hint at the direction of change, clearly labelled as weak.

        Excess Green finds green things, which is not the same as finding vegetation, so
        this is offered as a hint and never as a measurement.
        """
        if not mask.any():
            return []
        delta = float((excess_green(second) - excess_green(first))[mask].mean())
        if abs(delta) < 0.01:
            return []
        direction = "greener" if delta > 0 else "less green"
        return [
            f"hint only: the changed area became {direction} on the Excess Green index, "
            "which finds green things rather than live vegetation and is not a "
            "measurement of vegetation"
        ]

    def _render(self, request: ToolRequest, base: np.ndarray, mask: np.ndarray) -> Evidence:
        """Draw the changed region. A failure here must not fail a good measurement."""
        from satquery.utils.paths import artifact_dir

        evidence = Evidence()
        try:
            out = artifact_dir("serve", "evidence")
            stem = request.request_id[:12]
            evidence.highlight_path = write_png(
                out / f"{stem}_highlight.png", render_mask_alpha(mask, rgb=CHANGE_RGB)
            )
            from satquery.preprocess.optical import stretch_to_uint8

            # The backdrop is for looking at, so it may be stretched -- unlike the arrays
            # the measurement ran on.
            display = stretch_to_uint8(base.transpose(2, 0, 1)).transpose(1, 2, 0)
            evidence.mask_path = write_png(
                out / f"{stem}_change.png",
                render_mask_overlay(np.ascontiguousarray(display), {"changed": mask}),
            )
        except Exception as exc:
            _LOG.warning("evidence rendering failed, measurement is unaffected: %s", exc)
        return evidence
