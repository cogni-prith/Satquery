"""Semantic change masks, composed from the trained land-cover segmenter.

Composed, not separately trained, and the distinction is worth stating plainly.

SECOND ships per-pixel semantic change labels and is what a Siamese change network would
be trained on. Those label maps are not on this machine -- CDVQA ships only the image
pairs -- so there is nothing here to fit a change network to, and inventing one would mean
training on labels that do not exist.

What does exist is a segmenter measured at 0.95 IoU on water and 0.60 on built-up. Run it
on both dates and the per-pixel class transition IS the semantic change mask: the same
quantity a change network predicts, arrived at by measuring each date rather than by
learning the difference directly.

Two honest consequences of composing rather than training:

It cannot exploit correlations between the dates. A Siamese network sees both images at
once and can use a shadow moving, or one date's texture, to disambiguate the other. This
looks at each date in isolation, so registration error and per-date segmentation noise
both land in the change mask unattenuated.

Its errors are, on the other hand, exactly the segmenter's errors, and those are measured.
A separately trained change head would have its own error profile that nothing here has
scored, which is worse to ship than a known one.
"""

from __future__ import annotations

import numpy as np

from satquery.models.base import BaseTool
from satquery.serve.contracts import Evidence, ToolRequest, ToolResult
from satquery.symbolic.record import build_record
from satquery.utils.logging import get_logger
from satquery.verbalize.templates import verbalize

__all__ = ["SemanticChangeTool"]

_LOG = get_logger(__name__)


class SemanticChangeTool(BaseTool):
    """Per-pixel class transitions between two dates of one scene."""

    def __init__(self, spec, landcover_tool) -> None:
        super().__init__(spec)
        self.landcover = landcover_tool

    def _run(self, request: ToolRequest) -> ToolResult:
        if len(request.images) != 2:
            raise ValueError(f"semantic change needs exactly two images, got {len(request.images)}")

        first, first_agreement, warnings, base = self.landcover._analyse(request.images[0])
        second, _, second_warnings, _ = self.landcover._analyse(request.images[1])
        warnings = list(dict.fromkeys([*warnings, *second_warnings]))

        shared = sorted(set(first) & set(second))
        if not shared:
            raise ValueError(
                "no class was segmented at both dates, so no transition can be measured"
            )

        for name in shared:
            if first[name].shape != second[name].shape:
                raise ValueError(
                    "the two dates are not on one grid, so a per-pixel transition cannot "
                    f"be computed: {first[name].shape} and {second[name].shape}"
                )

        warnings.append(
            "this change mask is composed by segmenting each date independently, not "
            "predicted by a change network: registration error and per-date segmentation "
            "noise both appear in it unattenuated"
        )

        gsd_m = request.images[0].gsd_m
        # The class that moved most. Picked here rather than inside the renderer because
        # the legend drawn over the highlight has to name what the red pixels are, and a
        # class chosen privately by the renderer cannot be reported with its own areas.
        highlighted = max(shared, key=lambda k: int(np.count_nonzero(first[k] ^ second[k])))
        record = build_record(
            "change_trend",
            {
                "masks_t1": {k: first[k] for k in shared},
                "masks_t2": {k: second[k] for k in shared},
                "agreement": first_agreement,
                "warnings": warnings,
            },
            gsd_m,
        )

        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=verbalize(record),
            evidence=self._render(request, base, first, second, highlighted),
            confidence=min(first_agreement.values()) if first_agreement else None,
            params_used={
                "classes": shared,
                "highlighted_class": highlighted,
                "composed_from": "seg.landcover",
                "transitions": self._transition_summary(first, second, shared, gsd_m),
            },
            warnings=record.warnings,
            answer_record=record.model_dump(mode="json"),
        )

    @staticmethod
    def _transition_summary(first, second, shared, gsd_m) -> dict[str, float]:
        """From-class to-class areas, which is what makes this *semantic* change.

        "Something changed here" is what a binary mask gives. Naming both endpoints --
        vegetation became built-up -- is the part that answers a question.
        """
        scale = float(gsd_m) ** 2 if gsd_m and gsd_m > 0 else 1.0
        unit = "m2" if gsd_m and gsd_m > 0 else "px"
        out: dict[str, float] = {}
        for source in shared:
            for target in shared:
                if source == target:
                    continue
                count = int(np.count_nonzero(first[source] & second[target]))
                if count:
                    out[f"{source}->{target}_{unit}"] = round(count * scale, 1)
        return out

    def _render(self, request, base, first, second, name) -> Evidence:
        """Draw the transition. A failed render must not fail a good measurement."""
        from satquery.models.indices.render import render_change_alpha, render_change_map, write_png
        from satquery.preprocess.optical import stretch_to_uint8
        from satquery.utils.paths import artifact_dir

        evidence = Evidence()
        try:
            out = artifact_dir("serve", "evidence")
            stem = request.request_id[:12]
            display = np.ascontiguousarray(stretch_to_uint8(base[[2, 1, 0]]).transpose(1, 2, 0))
            evidence.mask_path = write_png(
                out / f"{stem}_change.png",
                render_change_map(display, first[name], second[name]),
            )
            evidence.highlight_path = write_png(
                out / f"{stem}_highlight.png",
                render_change_alpha(first[name], second[name]),
            )
        except Exception as exc:
            _LOG.warning("evidence rendering failed, measurement is unaffected: %s", exc)
        return evidence
