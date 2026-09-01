"""Task wrappers that turn one EarthDial backbone into three registered tools.

`VqaTool`, `CaptionTool` and `GroundingTool` are the single-image half of the
problem statement: VQA is mandatory, and the architecture commits to doing both of the
"one more single-image task" options, captioning and grounding.

Each class is thin on purpose. It owns:

- its `ToolSpec`, looked up from the registry rather than redeclared, so the spec the
  router reads and the spec the tool enforces cannot disagree;
- a pure prompt builder, implemented for real and unit-testable with no GPU;
- parameter defaulting from `ToolSpec.param_schema`, so `ToolResult.params_used`
  reports what was actually applied rather than what was requested.

`_run` needs weights: it raises `NotImplementedError` when the tool has no backbone
configured, and otherwise runs real generation.

Binding these classes into the registry is `serve/tools.py`'s job, owned by the lead.
This module only makes them importable.

Grounding output contract
-------------------------
`GroundingTool` returns boxes as `satquery.serve.contracts.BoundingBox`: absolute
pixel coordinates in `xyxy` order against `ToolRequest.images[0]`, hence
`image_index=0` -- the grounding spec accepts only `InputConfig.SINGLE`, so there is
never a second image to point at. Coordinates are not normalised and not geographic;
the backend reprojects with the raster's affine transform when it needs map
coordinates. VRSBench scores grounding as `acc@tau` on horizontal boxes, so a box
must be axis-aligned in the source raster's own pixel grid. Any model that emits
normalised or `[0, 1000]`-scaled coordinates (InternVL-style `<box>` tokens do) must
be rescaled by `image.width` / `image.height` before a `BoundingBox` is constructed;
`BoundingBox` rejects degenerate boxes, which is the intended failure mode for a
malformed generation.
"""

from __future__ import annotations

import re
from typing import Any

from satquery.models.base import BaseTool, load_model_input
from satquery.models.registry import REGISTRY
from satquery.models.vlm.backbone import BackboneConfig, EarthDialBackbone
from satquery.preprocess.constants import (
    BOX_COORDINATE_SCALE,
    INSTRUCTION_CAPTION,
    INSTRUCTION_CHANGE_DESCRIPTION,
    INSTRUCTION_CHANGE_QUESTION_TEMPLATE,
    INSTRUCTION_REFER_TEMPLATE,
    INSTRUCTION_VQA_TEMPLATE,
)
from satquery.preprocess.gsd import prefix_instruction
from satquery.serve.contracts import BoundingBox, Evidence, ToolRequest, ToolResult, ToolSpec
from satquery.utils.logging import get_logger

__all__ = [
    "CaptionTool",
    "ChangeDescriptionTool",
    "GroundingTool",
    "VlmTaskTool",
    "VqaTool",
    "default_params",
    "parse_boxes",
    "parse_internvl_boxes",
]

_LOG = get_logger(__name__)

#: Instruction templates come from `preprocess/constants.py`, where they are frozen
#: verbatim from the VRSBench training split. They are NOT redefined here: a paraphrase
#: at inference time silently prompts the adapter off-distribution.
CAPTION_INSTRUCTION = INSTRUCTION_CAPTION
GROUNDING_INSTRUCTION_TEMPLATE = INSTRUCTION_REFER_TEMPLATE


def default_params(spec: ToolSpec, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge requested params over the defaults declared in `spec.param_schema`.

    Pure and GPU-free. The result is what belongs in `ToolResult.params_used`: the
    parameters actually applied, including defaults the caller never mentioned.

    Args:
        spec: The tool spec whose JSON Schema carries the defaults.
        params: Requested parameters, normally `ToolRequest.params`.

    Returns:
        A new mapping of effective parameters. Keys the schema does not declare are
        passed through unchanged; validating them is the backend router's job.
    """
    properties = spec.param_schema.get("properties", {})
    effective: dict[str, Any] = {}
    if isinstance(properties, dict):
        for key, schema in properties.items():
            if isinstance(schema, dict) and "default" in schema:
                effective[str(key)] = schema["default"]
    effective.update(params or {})
    return effective


#: Three box formats show up in practice, on two coordinate scales. All were confirmed
#: against real EarthDial output on VRSBench validation imagery, not assumed:
#:
#: - ``[[10, 51, 26, 55, 90]]`` is what EarthDial actually emits: ``(x1, y1, x2, y2,
#:   angle)`` normalised to 0-100. The angle is DROPPED -- VRSBench scores ``acc@tau``
#:   on horizontal boxes, so the axis-aligned extent is what counts.
#: - ``{<45><45><59><59>}`` is the VRSBench annotation format, also 0-100.
#: - ``[[x1, y1, x2, y2]]`` is InternVL's stock format, normalised to 0-1000.
#:
#: The scale is inferred from the format rather than configured, because getting it
#: wrong is a silent 10x error: the box still parses, still renders, and still scores
#: near zero. The five-value pattern is tried first; the four-value pattern cannot
#: match a five-value list anyway, since it requires a closing bracket after the
#: fourth number, but ordering makes that independent of regex subtleties.
INTERNVL_BOX_SCALE = 1000.0

_ORIENTED_BOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*-?\d+(?:\.\d+)?\s*\]"
)
_ANGLE_BOX_RE = re.compile(r"<(-?\d+(?:\.\d+)?)>\s*" * 4)
_BRACKET_BOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def parse_boxes(
    text: str,
    *,
    width: int | None,
    height: int | None,
    label: str,
    max_boxes: int = 10,
) -> list[BoundingBox]:
    """Extract bounding boxes from a generation, in absolute pixel ``xyxy``.

    Coordinates arrive normalised, so every one is rescaled by the source raster's own
    width and height. VRSBench scores ``acc@tau`` against absolute pixels, so a wrong
    scale produces a plausible-looking box and a near-zero score.

    Values are clamped to the image: a minority of VRSBench targets exceed the nominal
    range (154 observed against a nominal 100), and a real generation can wander
    further. Clamping keeps a slightly-overflowing box usable; it does not invent one.

    Malformed or degenerate generations are skipped rather than repaired, because
    repairing a degenerate box turns a miss into a fabricated hit.

    Args:
        text: Raw model output.
        width: Source raster width in pixels. Without it no rescale is possible and the
            function returns nothing rather than guessing.
        height: Source raster height in pixels.
        label: Phrase the box answers.
        max_boxes: Cap on how many boxes to return.
    """
    if width is None or height is None:
        return []

    # Most specific first. Each pattern carries the scale its format implies.
    matches = [(m.groups(), BOX_COORDINATE_SCALE) for m in _ORIENTED_BOX_RE.finditer(text)]
    if not matches:
        matches = [(m.groups(), BOX_COORDINATE_SCALE) for m in _ANGLE_BOX_RE.finditer(text)]
    if not matches:
        matches = [(m.groups(), INTERNVL_BOX_SCALE) for m in _BRACKET_BOX_RE.finditer(text)]

    boxes: list[BoundingBox] = []
    for groups, scale in matches:
        x_min, y_min, x_max, y_max = (float(value) for value in groups)
        try:
            boxes.append(
                BoundingBox(
                    x_min=min(max(x_min / scale * width, 0.0), width),
                    y_min=min(max(y_min / scale * height, 0.0), height),
                    x_max=min(max(x_max / scale * width, 0.0), width),
                    y_max=min(max(y_max / scale * height, 0.0), height),
                    label=label,
                    image_index=0,
                )
            )
        except ValueError:
            _LOG.debug("skipping degenerate generated box %s", groups)
        if len(boxes) >= max_boxes:
            break
    return boxes


#: Kept as the old name so nothing downstream breaks on the rename.
parse_internvl_boxes = parse_boxes


class VlmTaskTool(BaseTool):
    """Shared base for the three single-image VLM tools.

    Holds the backbone handle without loading it. The registry caches one instance per
    tool, so the weights are materialised on first real call and not at import.
    """

    #: Registry name of the spec this tool implements. Set by each subclass.
    tool_name: str = ""

    def __init__(
        self,
        spec: ToolSpec | None = None,
        backbone: EarthDialBackbone | None = None,
        *,
        backbone_config: BackboneConfig | None = None,
    ) -> None:
        """Construct the tool.

        Args:
            spec: Override the registry spec. Defaults to `REGISTRY.get_spec(tool_name)`
                at the highest registered version.
            backbone: An already-constructed backbone handle, shared between tools so
                three tools do not hold three copies of a 4B model.
            backbone_config: Used to build a backbone when one is not supplied. When
                both are None the tool has no backbone and `_run` says so.
        """
        if not self.tool_name:
            raise TypeError(f"{type(self).__name__} must set a class-level `tool_name`")
        super().__init__(spec or REGISTRY.get_spec(self.tool_name))
        if backbone is None and backbone_config is not None:
            backbone = EarthDialBackbone(backbone_config)
        self.backbone = backbone

    # -- pure helpers -------------------------------------------------------------

    def effective_params(self, request: ToolRequest) -> dict[str, Any]:
        """Requested params merged over this spec's schema defaults."""
        return default_params(self.spec, request.params)

    def build_instruction(self, request: ToolRequest) -> str:
        """Build the GSD-prefixed instruction string for this request.

        The token comes from `images[0].gsd_m` via
        `satquery.preprocess.gsd.prefix_instruction`, which is idempotent and emits
        `<gsd:unknown>` rather than a guess. The backbone must never add its own.
        """
        return prefix_instruction(self._instruction_body(request), request.images[0].gsd_m)

    def _instruction_body(self, request: ToolRequest) -> str:
        """Token-free instruction text. Overridden per task."""
        return request.query.strip()

    def _generate(self, request: ToolRequest, params: dict[str, Any]) -> tuple[str, list[str]]:
        """Render every input image and run one generation pass.

        Returns:
            `(answer, warnings)`. Warnings come from raster ingest and SAR rendering and
            are carried through to `ToolResult.warnings` rather than swallowed.
        """
        if self.backbone is None:
            raise self._missing(
                "no backbone is configured on this tool. Construct it with a "
                "`backbone_config` built from configs/model/earthdial_4b_rgb.yaml, or "
                "pass a shared EarthDialBackbone."
            )

        images: list[Any] = []
        warnings: list[str] = []
        for ref in request.images:
            rendered, image_warnings = load_model_input(ref)
            images.append(rendered)
            warnings.extend(image_warnings)

        generation = {
            key: params[key]
            for key in ("max_new_tokens", "temperature", "num_beams")
            if key in params
        }
        answer = self.backbone.generate(images, self.build_instruction(request), **generation)
        return answer, warnings

    def _result(
        self,
        request: ToolRequest,
        params: dict[str, Any],
        answer: str,
        warnings: list[str],
        evidence: Evidence | None = None,
    ) -> ToolResult:
        """Assemble the ToolResult. `latency_ms` is filled in by `BaseTool.run`."""
        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=answer,
            evidence=evidence or Evidence(),
            params_used=params,
            warnings=warnings,
        )

    def _missing(self, detail: str) -> NotImplementedError:
        """Build the honest failure for a tool whose weights do not exist yet."""
        repo = self.backbone.config.hf_repo_id if self.backbone is not None else "<no backbone>"
        return NotImplementedError(
            f"{self.spec.name} is not implemented. Missing: {detail} The backbone is {repo}; "
            "the fine-tuned LoRA adapter from the BigEarthNet.txt and VRSBench mix has not been "
            "trained, so there is no weight to run and a fabricated answer would corrupt eval."
        )


class VqaTool(VlmTaskTool):
    """`vlm.vqa` -- answer a natural-language question about one image.

    Mandatory per the problem statement. Scored on RSVQA (closed set, so decoding is
    constrained by the `answer_set` param and normalised through
    `satquery.eval.answer_norm`) and on VRSBench VQA (open set, LLM judge).
    """

    tool_name = "vlm.vqa"

    def _instruction_body(self, request: ToolRequest) -> str:
        """Wrap the question in the frozen VQA template the model was tuned on."""
        return INSTRUCTION_VQA_TEMPLATE.format(question=request.query.strip().rstrip("?."))

    def _run(self, request: ToolRequest) -> ToolResult:
        """Generate an answer. Raises if no backbone is configured -- never invents one."""
        params = self.effective_params(request)
        answer, warnings = self._generate(request, params)
        return self._result(request, params, answer, warnings)


class CaptionTool(VlmTaskTool):
    """`vlm.caption` -- describe the land cover and major objects in one image.

    Scored with the VRSBench caption metrics. The user query is ignored unless it
    carries a steer, because VRSBench captioning is an unconditional description task.
    """

    tool_name = "vlm.caption"

    def _instruction_body(self, request: ToolRequest) -> str:
        """Always the frozen caption instruction. The user's phrasing is deliberately unused.

        VRSBench captioning is an unconditional description task: every reference caption
        was written against this exact prompt. Passing the user's wording through instead
        would prompt the adapter off-distribution and make the caption metrics measure
        something other than what they claim to.

        The user's query is still on `ToolRequest` and in the trace, so nothing is lost
        for auditing -- it just does not reach the model.
        """
        return CAPTION_INSTRUCTION

    def _run(self, request: ToolRequest) -> ToolResult:
        """Generate a description of the scene."""
        params = self.effective_params(request)
        answer, warnings = self._generate(request, params)
        return self._result(request, params, answer, warnings)


class GroundingTool(VlmTaskTool):
    """`vlm.grounding` -- localise the region a referring expression points at.

    Fine-tuned box head on the VRSBench referring split (52,472 referring
    expressions), scored `acc@tau` on horizontal boxes. See the module docstring for
    how generated coordinates must be converted into `BoundingBox`.

    The router prefers `detector.openvocab` when the query names a bare object class;
    this tool is for descriptive and relational phrases.
    """

    tool_name = "vlm.grounding"

    def _instruction_body(self, request: ToolRequest) -> str:
        """Wrap the referring phrase in the frozen grounding template."""
        return GROUNDING_INSTRUCTION_TEMPLATE.format(phrase=request.query.strip().rstrip("?."))

    def _run(self, request: ToolRequest) -> ToolResult:
        """Generate a box and rescale it into the source raster's pixel grid."""
        params = self.effective_params(request)
        answer, warnings = self._generate(request, params)

        image = request.images[0]
        boxes = parse_boxes(
            answer,
            width=image.width,
            height=image.height,
            label=request.query.strip(),
            max_boxes=int(params.get("max_boxes", 10)),
        )
        if not boxes:
            warnings.append(
                "no bounding box could be parsed from the generation; the answer text is "
                "returned but carries no evidence"
            )
        return self._result(request, params, answer, warnings, Evidence(boxes=boxes))


class ChangeDescriptionTool(VlmTaskTool):
    """`vlm.change_description` -- the generative half of the change doctrine.

    `the architecture` is explicit that CDVQA must not be routed through the VLM alone: the answer
    set is closed over nineteen values, so a discriminative head beats a generative model on
    the scored metric and answers in milliseconds. But it is equally explicit that the VLM
    keeps the free-form job and that the router should "route both, report both".

    This is that second half. `change.vqa_head` answers *which* label; this answers *what
    happened*, in prose a person can read. Neither substitutes for the other -- shipping only
    the classifier leaves a user staring at a bare token like `30_to_40`, and shipping only
    the prose loses the metric the benchmark actually scores.

    Unlike the head, this tool has no closed vocabulary to fall back on, so it inherits the
    backbone's fluency *and* its willingness to narrate confidently about imagery far from
    anything it was trained on. That is a property of the tool, not a defect to hide: the
    warning below travels with every answer.
    """

    tool_name = "vlm.change_description"

    def _instruction_body(self, request: ToolRequest) -> str:
        """Frozen bi-temporal framing, with the caller's question when there is one.

        A bare query is passed through the template rather than used directly, because the
        template is what tells the model which image is earlier. Dropping it for a
        user-supplied question would let "what changed?" reach the backbone with no
        ordering cue at all.
        """
        query = request.query.strip()
        generic = {"", "what changed", "what changed?", "describe the change", "describe changes"}
        if query.lower().rstrip("?") in {g.rstrip("?") for g in generic}:
            return INSTRUCTION_CHANGE_DESCRIPTION
        return INSTRUCTION_CHANGE_QUESTION_TEMPLATE.format(question=query)

    def _run(self, request: ToolRequest) -> ToolResult:
        """Describe the change in prose.

        Raises:
            ValueError: Fewer than two images. A change description from a single image
                would be pure invention, and the model would produce one fluently.
        """
        if len(request.images) != 2:
            raise ValueError(
                f"vlm.change_description needs exactly two images, got {len(request.images)}. "
                "A single image cannot show change, and the backbone would narrate one anyway."
            )

        params = self.effective_params(request)
        answer, warnings = self._generate(request, params)
        warnings.append(
            "free-form change description is not scored against the closed CDVQA answer "
            "set; use change.vqa_head for the measured answer and treat this prose as "
            "explanation rather than evidence"
        )
        return self._result(request, params, answer, warnings)
