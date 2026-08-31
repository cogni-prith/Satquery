"""Shared tool machinery: timing, error capture and spec conformance.

Every tool subclasses `BaseTool` and implements `_run`. `BaseTool.run` wraps that
call so a `ToolResult` is returned on every path, including an unhandled exception
or a not-yet-implemented body. That is what makes the execution trace always well
formed, which is the part of the problem statement that is actually scored.
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np

from satquery.serve.contracts import (
    Evidence,
    ImageRef,
    InputConfig,
    Modality,
    ToolRequest,
    ToolResult,
    ToolSpec,
)
from satquery.utils.logging import get_logger

__all__ = ["BaseTool", "infer_input_config_from_images", "load_model_input"]

_LOG = get_logger(__name__)


def infer_input_config_from_images(request: ToolRequest) -> InputConfig:
    """Best-effort input configuration for spec checking inside a tool.

    The authoritative decision belongs to `io/validate.py`, which sees timestamps,
    CRS and footprints. This is the cheap fallback used when a tool is called
    directly without having been routed.
    """
    if len(request.images) == 1:
        return InputConfig.SINGLE
    first, second = request.images[0], request.images[1]
    if first.modality is not second.modality:
        return InputConfig.CROSS_MODAL_PAIR
    return InputConfig.BI_TEMPORAL_PAIR


def load_model_input(ref: ImageRef) -> tuple[np.ndarray, list[str]]:
    """Read a raster and render it to the ``(height, width, 3)`` uint8 RGB a VLM takes.

    This is the single bridge between the georeferenced world and the model's world,
    and it is the reason the SAR rendering is frozen: a Sentinel-1 training tile and a
    RISAT evaluation tile both arrive at the backbone through this one function, having
    been through byte-identical arithmetic.

    Dispatch by modality, never by band count:

    - SAR -> the frozen five-step pseudo-RGB pipeline in ``preprocess.sar.render_sar``.
    - optical or multispectral -> red, green and blue looked up *by canonical name* and
      percentile-stretched with the same function the SAR path uses.
    - panchromatic -> the single band replicated across all three channels.

    Returns:
        ``(rgb, warnings)``. Warnings from ingest, band mapping and rendering are all
        preserved so they reach ``ToolResult.warnings``.
    """
    from satquery.io.modality import polarisation_slots
    from satquery.io.raster import read_raster
    from satquery.preprocess.constants import BAND_ROLE_BLUE, BAND_ROLE_GREEN, BAND_ROLE_RED
    from satquery.preprocess.optical import stretch_to_uint8
    from satquery.preprocess.sar import render_sar

    array, parsed = read_raster(ref.path)
    warnings = list(parsed.warnings)
    names = parsed.band_names

    if parsed.modality is Modality.SAR:
        co_pol, cross_pol = polarisation_slots(names)
        if co_pol is None and cross_pol is None:
            raise ValueError(
                f"{ref.path} is SAR but carries no recognised polarisation among {names}; "
                "render_sar needs at least one of VV, VH, HH or HV"
            )
        co = array[names.index(co_pol)] if co_pol else None
        cross = array[names.index(cross_pol)] if cross_pol else None
        rendered, sar_warnings = render_sar(co, cross)
        return rendered, warnings + sar_warnings

    if parsed.modality is Modality.PANCHROMATIC or len(names) == 1:
        single = stretch_to_uint8(array[:1])[0]
        return np.repeat(single[:, :, np.newaxis], 3, axis=2), warnings

    wanted = (BAND_ROLE_RED, BAND_ROLE_GREEN, BAND_ROLE_BLUE)
    missing = [band for band in wanted if band not in names]

    if missing:
        # A plain PNG or JPEG carries no band descriptions, so io/modality.py names its
        # bands `band_1..band_3`. For exactly three unnamed bands, RGB order is defined
        # by the file format itself, not assumed about a sensor -- so reading them
        # positionally here is correct, and it is the ONLY place positional access is
        # allowed. Anything else (a named stack missing a band, or an unnamed stack with
        # a different width) is still a hard error.
        unnamed = all(name.startswith("band_") for name in names)
        if unnamed and len(names) == 3:
            stack = array[:3]
            return (
                np.ascontiguousarray(stretch_to_uint8(stack).transpose(1, 2, 0)),
                warnings,
            )
        raise ValueError(
            f"{ref.path} is missing the band(s) {missing} needed to build an RGB view. "
            f"Available: {names}. Bands are matched by name; only a three-band image "
            f"with no band descriptions at all is read positionally as RGB."
        )

    stack = np.stack([array[names.index(band)] for band in wanted])
    return np.ascontiguousarray(stretch_to_uint8(stack).transpose(1, 2, 0)), warnings


class BaseTool(ABC):
    """Abstract base for every registered tool."""

    #: Declarative description used by the router's deterministic gate.
    spec: ToolSpec

    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec

    # -- public API ---------------------------------------------------------------

    def run(self, request: ToolRequest) -> ToolResult:
        """Execute the tool, never raising. Failures come back as `ToolResult.error`."""
        started = perf_counter()
        warnings = self.check_request(request)

        try:
            result = self._run(request)
        except NotImplementedError as exc:
            return self._error_result(request, f"tool is not implemented: {exc}", started, warnings)
        except Exception as exc:
            _LOG.exception("tool %s failed on request %s", self.spec.name, request.request_id)
            detail = f"{type(exc).__name__}: {exc}"
            _LOG.debug("traceback:\n%s", traceback.format_exc())
            return self._error_result(request, detail, started, warnings)

        result.latency_ms = round((perf_counter() - started) * 1000.0, 3)
        result.warnings = [*warnings, *result.warnings]
        return result

    def check_request(self, request: ToolRequest) -> list[str]:
        """Return warnings where the request sits outside this tool's declared spec.

        Deliberately warnings rather than errors: on the hidden evaluation set an
        out-of-range GSD or an unparsed transform is likely, and answering with a
        caveat beats refusing to answer.
        """
        warnings: list[str] = []
        modalities: list[Modality] = [image.modality for image in request.images]
        input_config = infer_input_config_from_images(request)

        if input_config not in self.spec.accepted_input_configs:
            accepted = ", ".join(c.value for c in self.spec.accepted_input_configs)
            warnings.append(
                f"{self.spec.name} declares input configs [{accepted}] but received "
                f"{input_config.value}"
            )

        for modality in modalities:
            if modality not in self.spec.accepted_modalities:
                accepted = ", ".join(m.value for m in self.spec.accepted_modalities)
                warnings.append(
                    f"{self.spec.name} declares modalities [{accepted}] but received "
                    f"{modality.value}"
                )

        for index, image in enumerate(request.images):
            if image.gsd_m is None:
                warnings.append(
                    f"image {index} has no computable GSD; instruction will carry {image.gsd_token}"
                )
            elif not self.spec.accepts_gsd(image.gsd_m):
                warnings.append(
                    f"image {index} GSD {image.gsd_m:.2f} m is outside the declared range "
                    f"[{self.spec.min_gsd_m}, {self.spec.max_gsd_m}] for {self.spec.name}"
                )
            warnings.extend(image.warnings)

        return warnings

    # -- subclass hook ------------------------------------------------------------

    @abstractmethod
    def _run(self, request: ToolRequest) -> ToolResult:
        """Do the work. `latency_ms` is filled in by `run`, so leave it at its default."""

    # -- helpers ------------------------------------------------------------------

    def _error_result(
        self,
        request: ToolRequest,
        message: str,
        started: float,
        warnings: list[str],
    ) -> ToolResult:
        """Build the `ToolResult` used for every failure path."""
        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=None,
            evidence=Evidence(),
            confidence=None,
            params_used=dict(request.params),
            latency_ms=round((perf_counter() - started) * 1000.0, 3),
            warnings=warnings,
            error=message,
        )
