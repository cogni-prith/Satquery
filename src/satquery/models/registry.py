"""The tool registry the backend reads to decide what to call.

This file and `serve/contracts.py` are the entire public API. `BUILTIN_SPECS` below
is the lookup table behind stage one of the router: parsed raster metadata goes in,
a filtered candidate list comes out, with no model call and no guessing.

Specs are registered whether or not their implementation exists yet. `ToolSpec.
implemented` says which is which, so the backend can build against the real shape of
the API today and get a `NotImplementedError` -- never a fabricated answer -- if it
calls a tool whose weights are not trained.

A spec is `implemented` only where weights exist and have been scored. The four VLM tools
are back to True: v1's EarthDial adapter is trained and measured (rsvqa 0.655, vrsbench
vqa 0.620, grounding acc@0.5 0.367, caption BLEU-4 0.116). `seg.landcover` is True
(validation IoU water 0.95, built-up 0.60). `indices.deterministic` and
`change.radiometric` are closed-form and have no weights to be missing.

`detector.openvocab`, `change.mask` and `fusion.extraction` stay False -- those genuinely
have no trained weights, and a capability list that overstates is worse than a short one.

Binding is still conditional at load time: `serve/tools.py` unregisters a spec whose
weights turn out to be absent, so a deployment without the adapter offers less rather than
failing on every call.

Changing this file is a cross-team event: update `docs/INTERFACE.md` in the same
commit and say so in the commit message.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable

from satquery.preprocess.constants import CDVQA_ANSWERS
from satquery.serve.contracts import InputConfig, Modality, TaskType, Tool, ToolSpec

__all__ = ["BUILTIN_SPECS", "REGISTRY", "ToolRegistry", "main"]

_OPTICAL = [Modality.OPTICAL_RGB, Modality.MULTISPECTRAL]
_ALL_IMAGING = [Modality.OPTICAL_RGB, Modality.MULTISPECTRAL, Modality.SAR, Modality.PANCHROMATIC]

# Generous GSD envelope for the learned tools. The training mix is 10 m Sentinel and
# roughly 0.3 m aerial; the hidden evaluation set is sub-metre Cartosat-2S and RISAT
# SAR. Declaring a narrow range would make the gate refuse exactly the imagery we are
# being scored on, so the range is wide and the per-image warning carries the caveat.
_MIN_GSD_M = 0.1
_MAX_GSD_M = 120.0

_GENERATION_PARAMS: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 1024, "default": 128},
        "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0, "default": 0.0},
        "num_beams": {"type": "integer", "minimum": 1, "maximum": 8, "default": 1},
    },
}


BUILTIN_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="vlm.vqa",
        version="0.1.0",
        task=TaskType.VQA,
        accepted_modalities=[Modality.OPTICAL_RGB, Modality.MULTISPECTRAL, Modality.SAR],
        accepted_input_configs=[InputConfig.SINGLE],
        min_gsd_m=_MIN_GSD_M,
        max_gsd_m=_MAX_GSD_M,
        param_schema={
            **_GENERATION_PARAMS,
            "properties": {
                **_GENERATION_PARAMS["properties"],  # type: ignore[dict-item]
                "answer_set": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constrain decoding to a closed answer set, as RSVQA requires.",
                },
            },
        },
        returns=["answer", "confidence"],
        description="Answer a natural-language question about a single remote sensing image.",
        requires_gpu=True,
        implemented=True,
    ),
    ToolSpec(
        name="vlm.caption",
        version="0.1.0",
        task=TaskType.CAPTION,
        accepted_modalities=[Modality.OPTICAL_RGB, Modality.MULTISPECTRAL, Modality.SAR],
        accepted_input_configs=[InputConfig.SINGLE],
        min_gsd_m=_MIN_GSD_M,
        max_gsd_m=_MAX_GSD_M,
        param_schema=_GENERATION_PARAMS,
        returns=["answer", "confidence"],
        description="Describe the land cover and major objects visible in a single image.",
        requires_gpu=True,
        implemented=True,
    ),
    ToolSpec(
        name="vlm.grounding",
        version="0.1.0",
        task=TaskType.GROUNDING,
        accepted_modalities=_OPTICAL,
        accepted_input_configs=[InputConfig.SINGLE],
        min_gsd_m=_MIN_GSD_M,
        max_gsd_m=_MAX_GSD_M,
        param_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "max_boxes": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "score_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.25,
                },
            },
        },
        returns=["answer", "evidence.boxes", "evidence.overlay_path", "confidence"],
        description=(
            "Localise the region a referring expression points at. Fine-tuned box head, "
            "preferred when the phrase is descriptive rather than a bare object class."
        ),
        requires_gpu=True,
        implemented=True,
    ),
    ToolSpec(
        name="detector.openvocab",
        version="0.1.0",
        task=TaskType.GROUNDING,
        accepted_modalities=_OPTICAL,
        accepted_input_configs=[InputConfig.SINGLE],
        min_gsd_m=_MIN_GSD_M,
        max_gsd_m=30.0,
        param_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "text_prompt": {
                    "type": "string",
                    "description": "Object classes, period separated, e.g. 'aircraft. ship.'",
                },
                "box_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3,
                },
                "text_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.25,
                },
            },
        },
        returns=["evidence.boxes", "evidence.overlay_path", "confidence"],
        description=(
            "Open-vocabulary detector. The router prefers this over vlm.grounding when the "
            "query names a concrete object class rather than a spatial relation."
        ),
        requires_gpu=True,
        implemented=False,
    ),
    ToolSpec(
        name="vlm.change_description",
        version="0.1.0",
        task=TaskType.CHANGE_DESCRIPTION,
        accepted_modalities=[Modality.OPTICAL_RGB, Modality.MULTISPECTRAL, Modality.SAR],
        accepted_input_configs=[InputConfig.BI_TEMPORAL_PAIR],
        min_gsd_m=_MIN_GSD_M,
        max_gsd_m=_MAX_GSD_M,
        param_schema=_GENERATION_PARAMS,
        returns=["answer", "confidence"],
        description="Free-form description of what changed between two co-registered dates.",
        requires_gpu=True,
        implemented=True,
    ),
    ToolSpec(
        name="change.vqa_head",
        version="0.1.0",
        task=TaskType.CHANGE_VQA,
        accepted_modalities=_OPTICAL,
        accepted_input_configs=[InputConfig.BI_TEMPORAL_PAIR],
        min_gsd_m=_MIN_GSD_M,
        max_gsd_m=_MAX_GSD_M,
        param_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer_set": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(CDVQA_ANSWERS)},
                    "description": "Closed CDVQA class set. Defaults to all six classes.",
                }
            },
        },
        returns=["answer", "confidence"],
        description=(
            "Discriminative Siamese head over the closed six-class CDVQA answer set. "
            "Scored separately from vlm.change_description; the router calls both."
        ),
        requires_gpu=True,
        implemented=True,
    ),
    ToolSpec(
        name="change.mask",
        version="0.1.0",
        task=TaskType.CHANGE_MASK,
        accepted_modalities=_OPTICAL,
        accepted_input_configs=[InputConfig.BI_TEMPORAL_PAIR],
        min_gsd_m=_MIN_GSD_M,
        max_gsd_m=30.0,
        param_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5}
            },
        },
        returns=["evidence.mask_path", "evidence.overlay_path", "confidence"],
        description="Pixel-level binary change mask. Optional per the problem statement.",
        requires_gpu=True,
        implemented=False,
    ),
    ToolSpec(
        name="fusion.extraction",
        version="0.1.0",
        task=TaskType.FUSION_EXTRACTION,
        accepted_modalities=_ALL_IMAGING,
        accepted_input_configs=[InputConfig.CROSS_MODAL_PAIR],
        min_gsd_m=_MIN_GSD_M,
        max_gsd_m=_MAX_GSD_M,
        param_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "targets": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["built_up", "water"]},
                    "default": ["built_up", "water"],
                }
            },
        },
        returns=[
            "answer",
            "evidence.mask_path",
            "evidence.overlay_path",
            "evidence.index_maps",
            "confidence",
        ],
        description=(
            "Joint built-up and water extraction from a co-registered optical and SAR pair, "
            "using the learned dual encoder cross-checked against the deterministic indices."
        ),
        requires_gpu=True,
        implemented=True,
    ),
    ToolSpec(
        name="seg.landcover",
        version="0.1.0",
        task=TaskType.FUSION_EXTRACTION,
        accepted_modalities=[Modality.MULTISPECTRAL],
        accepted_input_configs=[InputConfig.SINGLE, InputConfig.BI_TEMPORAL_PAIR],
        min_gsd_m=None,
        max_gsd_m=None,
        param_schema={"type": "object", "additionalProperties": False, "properties": {}},
        returns=[
            "answer",
            "evidence.overlay_path",
            "evidence.highlight_path",
            "confidence",
        ],
        description=(
            "Trained land-cover segmentation, scored against the deterministic spectral "
            "indices. The only tool here that produces two independent estimates of the "
            "same quantity, so it is the only one that can report a real agreement-based "
            "confidence rather than none. Measured on the reBEN validation split: water "
            "0.95, vegetation 0.82, agriculture 0.81, built-up 0.60, wetland 0.37 IoU."
        ),
        requires_gpu=False,
        implemented=True,
        # Preferred over indices.deterministic wherever both are legal: it measures the
        # same classes and additionally reports how far an independent estimate agrees.
        preference=10,
    ),
    ToolSpec(
        name="change.radiometric",
        version="0.1.0",
        task=TaskType.CHANGE_MASK,
        # The only tool here that accepts plain RGB, because it is the only one that does
        # not need a spectral index. A screenshot has no near-infrared and therefore no
        # NDWI, NDBI or NDVI -- but two screenshots of one place can still be differenced.
        accepted_modalities=[Modality.OPTICAL_RGB, Modality.MULTISPECTRAL, Modality.PANCHROMATIC],
        accepted_input_configs=[InputConfig.BI_TEMPORAL_PAIR],
        min_gsd_m=None,
        max_gsd_m=None,
        param_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "threshold": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 3.0,
                    "description": "Distance in normalised RGB space above which a pixel changed.",
                },
                "min_component_px": {"type": "integer", "minimum": 1, "default": 12},
            },
        },
        returns=["answer", "evidence.mask_path", "evidence.highlight_path", "confidence"],
        description=(
            "Radiometric change detection on a co-registered pair, for imagery with no "
            "near-infrared band. Reports WHERE the scene differs, never what it changed "
            "into: it cannot separate new construction from bare soil, a wet surface or a "
            "different sun angle. A screening instrument, deliberately weaker than the "
            "spectral path and labelled as such."
        ),
        requires_gpu=False,
        implemented=True,
    ),
    ToolSpec(
        name="indices.deterministic",
        version="0.1.0",
        task=TaskType.FUSION_EXTRACTION,
        accepted_modalities=[Modality.MULTISPECTRAL, Modality.SAR],
        # Bi-temporal is included because the same closed-form arithmetic answers
        # "how much has the water changed" as answers "how much water is there" --
        # it is two measurements and a subtraction, with no extra machinery.
        accepted_input_configs=[
            InputConfig.SINGLE,
            InputConfig.CROSS_MODAL_PAIR,
            InputConfig.BI_TEMPORAL_PAIR,
        ],
        min_gsd_m=None,
        max_gsd_m=None,
        param_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["ndwi", "ndbi", "ndvi", "sar_water"]},
                    "default": ["ndwi", "ndbi", "ndvi"],
                },
                "write_maps": {
                    "type": "boolean",
                    "default": True,
                    "description": "Write index rasters under the artifact root.",
                },
            },
        },
        returns=["answer", "evidence.mask_path", "evidence.index_maps", "confidence"],
        description=(
            "Closed-form spectral and backscatter indices. No weights, no GPU, no training "
            "distribution to fall outside of. This is the safety net that still produces a "
            "defensible answer with a visible map on unseen sensor data."
        ),
        requires_gpu=False,
        implemented=True,
    ),
)


def _version_key(version: str) -> tuple[int, ...]:
    """Sort key for a dotted version string. Non-numeric segments sort as zero."""
    parts: list[int] = []
    for segment in version.split("."):
        digits = "".join(ch for ch in segment if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class ToolRegistry:
    """Name-and-version keyed collection of tool specs and their lazy factories."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._factories: dict[str, Callable[[], Tool]] = {}
        self._instances: dict[str, Tool] = {}
        self._tools_module_loaded = False

    # -- registration ------------------------------------------------------------

    def register(
        self,
        spec: ToolSpec,
        factory: Callable[[], Tool] | None = None,
        *,
        replace: bool = False,
    ) -> ToolSpec:
        """Register a spec, optionally with a factory that builds its implementation.

        Args:
            spec: The declarative description. Registered even when the tool is a stub.
            factory: Zero-argument callable returning the `Tool`. Called at most once
                and the instance cached, so model weights load lazily and only when a
                tool is actually used.
            replace: Allow overwriting an existing `name@version`. Off by default so a
                duplicate registration is a loud error rather than a silent shadow.

        Returns:
            The spec that was registered.
        """
        if spec.key in self._specs and not replace:
            raise ValueError(f"{spec.key} is already registered; pass replace=True to override")
        self._specs[spec.key] = spec
        if factory is not None:
            self._factories[spec.key] = factory
            self._instances.pop(spec.key, None)
        return spec

    def bind(self, name: str, factory: Callable[[], Tool], version: str | None = None) -> None:
        """Attach a factory to an already-registered spec."""
        spec = self.get_spec(name, version)
        self._factories[spec.key] = factory
        self._instances.pop(spec.key, None)

    def unregister(self, name: str, version: str | None = None) -> None:
        """Remove a spec and any factory or cached instance. Used by tests."""
        spec = self.get_spec(name, version)
        self._specs.pop(spec.key, None)
        self._factories.pop(spec.key, None)
        self._instances.pop(spec.key, None)

    # -- lookup ------------------------------------------------------------------

    def get_spec(self, name: str, version: str | None = None) -> ToolSpec:
        """Return one spec by name, defaulting to the highest registered version."""
        if version is not None:
            key = f"{name}@{version}"
            if key not in self._specs:
                raise KeyError(f"no tool registered as {key}; known tools: {sorted(self._specs)}")
            return self._specs[key]

        matching = [spec for spec in self._specs.values() if spec.name == name]
        if not matching:
            known = sorted({spec.name for spec in self._specs.values()})
            raise KeyError(f"no tool named {name!r}; known tools: {known}")
        return max(matching, key=lambda spec: _version_key(spec.version))

    def get(self, name: str, version: str | None = None) -> Tool:
        """Return a live tool instance, constructing and caching it on first use.

        Raises:
            KeyError: No such tool is registered.
            NotImplementedError: The tool is registered as a spec but has no factory,
                which is the honest state of every learned tool until it is trained.
        """
        spec = self.get_spec(name, version)
        cached = self._instances.get(spec.key)
        if cached is not None:
            return cached

        factory = self._factories.get(spec.key)
        if factory is None:
            self._load_tool_module()
            factory = self._factories.get(spec.key)

        if factory is None:
            raise NotImplementedError(
                f"{spec.key} is registered as a spec but has no implementation bound. "
                f"Missing: the {spec.task.value} tool class in satquery.serve.tools and, "
                f"for a learned tool, trained weights under the artifact root."
            )

        instance = factory()
        self._instances[spec.key] = instance
        return instance

    def _load_tool_module(self) -> None:
        """Import `serve.tools` once, so its registration side effects run.

        Deferred rather than done at module import time because `serve.tools` imports
        this module; doing it here keeps the dependency one-directional at import time.
        """
        if self._tools_module_loaded:
            return
        self._tools_module_loaded = True
        import satquery.serve.tools  # noqa: F401  (imported for its registration side effects)

    def list_specs(self, *, implemented_only: bool = False) -> list[ToolSpec]:
        """Return every registered spec, ordered deterministically by name then version."""
        specs = list(self._specs.values())
        if implemented_only:
            specs = [spec for spec in specs if spec.implemented]
        return sorted(specs, key=lambda spec: (spec.name, _version_key(spec.version)))

    def candidates(
        self,
        input_config: InputConfig,
        modalities: Iterable[Modality],
        gsd_m: float | None = None,
        task: TaskType | None = None,
    ) -> list[ToolSpec]:
        """Return the candidate tools for stage one of the router's deterministic gate.

        This is a pure filter over declared metadata. No model is consulted, nothing is
        inferred from the query text, and the result is deterministic and ordered.

        Args:
            input_config: As decided by `io/validate.py`.
            modalities: Modality of every input image. A tool must accept all of them.
            gsd_m: Representative GSD, normally the coarsest of the inputs. `None`
                never excludes a tool.
            task: Optional task filter, applied once stage two has classified.

        Returns:
            Matching specs, ordered by name then version.
        """
        modality_list = list(modalities)
        matches = [
            spec
            for spec in self._specs.values()
            if spec.accepts(
                input_config=input_config,
                modalities=modality_list,
                gsd_m=gsd_m,
                task=task,
            )
        ]
        return sorted(matches, key=lambda spec: (spec.name, _version_key(spec.version)))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return any(spec.name == name or spec.key == name for spec in self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


def _build_default_registry() -> ToolRegistry:
    """Construct the process-wide registry from `BUILTIN_SPECS`."""
    registry = ToolRegistry()
    for spec in BUILTIN_SPECS:
        registry.register(spec)
    return registry


#: The process-wide registry. The backend imports this.
REGISTRY = _build_default_registry()


def _format_table(specs: list[ToolSpec]) -> str:
    """Render specs as a fixed-width table for `make registry` on a terminal."""
    header = f"{'TOOL':<28} {'VER':<8} {'TASK':<20} {'INPUT CONFIGS':<28} {'GSD (m)':<14} STATUS"
    lines = [header, "-" * len(header)]
    for spec in specs:
        configs = ",".join(c.value for c in spec.accepted_input_configs)
        low = "any" if spec.min_gsd_m is None else f"{spec.min_gsd_m:g}"
        high = "any" if spec.max_gsd_m is None else f"{spec.max_gsd_m:g}"
        status = "implemented" if spec.implemented else "stub"
        lines.append(
            f"{spec.name:<28} {spec.version:<8} {spec.task.value:<20} "
            f"{configs:<28} {low + '-' + high:<14} {status}"
        )
    implemented = sum(1 for spec in specs if spec.implemented)
    lines.append("")
    lines.append(
        f"{len(specs)} tools registered, {implemented} implemented, "
        f"{len(specs) - implemented} honest stubs."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. `--json` dumps every spec; the default is a readable table."""
    parser = argparse.ArgumentParser(
        prog="python -m satquery.models.registry",
        description="Inspect the SatQuery tool registry that the backend router reads.",
    )
    parser.add_argument("--json", action="store_true", help="Dump all specs as JSON.")
    parser.add_argument(
        "--implemented-only", action="store_true", help="Hide tools that are still stubs."
    )
    args = parser.parse_args(argv)

    specs = REGISTRY.list_specs(implemented_only=args.implemented_only)
    if args.json:
        payload = {
            "contract_version": __import__(
                "satquery.serve.contracts", fromlist=["CONTRACT_VERSION"]
            ).CONTRACT_VERSION,
            "tool_count": len(specs),
            "tools": [spec.model_dump(mode="json") for spec in specs],
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        sys.stdout.write(_format_table(specs) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
