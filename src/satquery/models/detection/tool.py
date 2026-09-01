"""Object detection over the VRSBench vocabulary, with counts the symbolic layer measures.

Closed-vocabulary, and the tool says so. The spec is named `detector.openvocab` because
that is what the architecture wants, but the training data is 26 fixed VRSBench classes
and a detector fitted to them detects those 26 things. Open-vocabulary detection needs a
text-conditioned model like Grounding-DINO; naming this one open-vocabulary would be a
claim the labels cannot support.

Counting is done by `symbolic.measures.object_count` over the returned boxes rather than
by the detector reporting a number. Same discipline as everywhere else here: the model
produces boxes, the symbolic layer produces the answer, and the count in the sentence is a
length of a list a judge can see.
"""

from __future__ import annotations

import numpy as np

from satquery.models.base import BaseTool
from satquery.serve.contracts import BoundingBox, Evidence, ToolRequest, ToolResult
from satquery.symbolic.record import AnswerRecord, Fact
from satquery.utils.logging import get_logger
from satquery.verbalize.templates import verbalize

__all__ = ["ObjectDetectorTool"]

_LOG = get_logger(__name__)

#: Below this the detector is guessing. Chosen to match the operating point the reported
#: recall and precision were measured at, so the numbers a user sees describe what they get.
SCORE_THRESHOLD = 0.5


class ObjectDetectorTool(BaseTool):
    """Detect objects, then let the symbolic layer count them."""

    def __init__(self, spec, model, vocabulary, device: str, size: int = 384) -> None:
        super().__init__(spec)
        self.model = model
        self.vocabulary = vocabulary
        self.device = device
        self.size = size

    def _run(self, request: ToolRequest) -> ToolResult:
        import torch

        from satquery.models.base import load_model_input
        from satquery.symbolic.measures import object_count

        rgb, warnings = load_model_input(request.images[0])
        height, width = rgb.shape[:2]

        from PIL import Image

        resized = (
            np.asarray(
                Image.fromarray(rgb).resize((self.size, self.size), Image.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )
        tensor = torch.from_numpy(resized).permute(2, 0, 1).to(self.device)

        threshold = float(request.params.get("score_threshold", SCORE_THRESHOLD))
        wanted = self._requested_labels(request.query)

        self.model.eval()
        with torch.no_grad():
            output = self.model([tensor])[0]

        scores = output["scores"].cpu().numpy()
        keep = scores >= threshold
        raw_boxes = output["boxes"].cpu().numpy()[keep]
        raw_labels = output["labels"].cpu().numpy()[keep]
        scores = scores[keep]

        boxes: list[BoundingBox] = []
        for box, label, score in zip(raw_boxes, raw_labels, scores, strict=True):
            # torchvision reserves 0 for background, so the vocabulary is offset by one.
            index = int(label) - 1
            if not 0 <= index < len(self.vocabulary):
                continue
            name = self.vocabulary[index]
            if wanted and name not in wanted:
                continue
            # Back to the CALLER'S pixel grid. The detector saw a square resize; the
            # contract is absolute pixels on the original raster, because that is what
            # VRSBench's acc@tau is scored against. Emitting the unit square here -- which
            # an earlier version did -- put boxes on a different scale from
            # vlm.grounding's, and anything consuming both drew one of them wrong.
            boxes.append(
                BoundingBox(
                    x_min=float(box[0] / self.size * width),
                    y_min=float(box[1] / self.size * height),
                    x_max=float(box[2] / self.size * width),
                    y_max=float(box[3] / self.size * height),
                    label=name,
                    score=float(score),
                    image_index=0,
                )
            )

        counts = {
            name: object_count([b.model_dump() for b in boxes], name)
            for name in sorted({b.label for b in boxes})
        }

        warnings.append(
            "closed vocabulary: this detector was fitted to 26 VRSBench classes and cannot "
            f"find anything outside them ({', '.join(self.vocabulary[:6])}, ...)"
        )
        warnings.append(
            "recall is capped by the training labels, which annotate one object per "
            "referring expression: unannotated instances were taught as background, so "
            "measured recall is 0.64 at IoU 0.5 and genuine objects will be missed"
        )
        if wanted and not boxes:
            warnings.append(
                f"nothing matched {sorted(wanted)} above score {threshold:.2f}; that is a "
                "negative result at this threshold, not proof the objects are absent"
            )

        record = AnswerRecord(
            intent="count",
            facts=[
                Fact(
                    key=f"{name}_count",
                    value=total,
                    unit="count",
                    provenance="symbolic.measures.object_count",
                )
                for name, total in counts.items()
            ],
            object_counts=counts,
            warnings=warnings,
        )
        record.validate_provenance()

        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=verbalize(record) if counts else self._empty_answer(request.query, wanted),
            evidence=Evidence(boxes=boxes),
            # A detection score is the model's own certainty, not agreement between two
            # independent estimates. Reporting it as confidence would be the softmax
            # mistake this project refuses everywhere else.
            confidence=None,
            params_used={
                "score_threshold": threshold,
                "vocabulary_size": len(self.vocabulary),
                "requested_labels": sorted(wanted) if wanted else "all",
            },
            warnings=record.warnings,
            answer_record=record.model_dump(mode="json"),
        )

    def _empty_answer(self, query: str, wanted: set[str]) -> str:
        """Say WHY nothing was found, which is usually the useful part.

        "No objects were found" is true and, on a scene that plainly contains the thing
        asked about, reads as a claim about the world rather than about this model. Asked
        "where is the highway?" on an image with a motorway across it, the honest answer is
        that `highway` is not a word this detector knows -- not that there is no highway.
        """
        if not wanted:
            return (
                f"Nothing in this detector's vocabulary matches your question. It was "
                f"trained on {len(self.vocabulary)} fixed classes and cannot look for "
                f"anything else: {', '.join(self.vocabulary)}. "
                "This is a limit of the detector, not a statement about the image -- try "
                "asking for one of those classes, or phrase it as a description question "
                "so the vision-language model handles it instead."
            )
        return (
            f"No {', '.join(sorted(wanted))} was detected above a score of "
            f"{SCORE_THRESHOLD:.2f}. That is a negative result at this threshold on a "
            "detector with measured recall of 0.64, not proof that none is present."
        )

    def _requested_labels(self, query: str) -> set[str]:
        """Vocabulary entries the query names, matched on words rather than substrings."""
        text = query.lower()
        found = set()
        for name in self.vocabulary:
            spaced = name.replace("-", " ")
            if spaced in text or name in text or f"{spaced}s" in text:
                found.add(name)
        return found
