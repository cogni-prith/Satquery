"""Discriminative Siamese change head for CDVQA, plus the tool that serves it.

Project doctrine, from CLAUDE.md: **do not route CDVQA through the VLM alone.**
CDVQA's answer set is closed over exactly six land-cover classes (`CDVQA_ANSWERS` in
`preprocess/constants.py`). A shared-weight Siamese encoder over the two dates,
followed by a small MLP classifier, beats a generative VLM on that closed set and
runs in milliseconds instead of seconds. The VLM keeps the free-form job
(`vlm.change_description`); this head produces the answer that is actually scored on
CDVQA test-1 and test-2. The router calls both and the trace reports both.

Architecture follows the standard Siamese change-detection recipe used by CDVQA's own
baselines (Yuan et al., "Change Detection Meets Visual Question Answering",
IEEE TGRS 2022, github.com/YZHJessica/CDVQA): one weight-shared image encoder applied
to t1 and t2, a temporal fusion of the two embeddings, then a classification head over
the closed answer set. The encoder is any `timm` backbone so the same code can be
re-run at Sentinel scale and at Cartosat scale without a structural change.

The frozen ordering of `CDVQA_ANSWERS` is this head's output-layer ordering. Permuting
it would silently scramble a reloaded checkpoint, so the index mapping lives in the
real, tested `answer_index` / `answer_name` pair below and nowhere else.

`torch`, `timm` and `peft` are optional `gpu` extras: this module must import on a
CPU-only laptop with none of them installed, so they are imported inside function
bodies only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from satquery.eval.answer_norm import normalize_cdvqa_class
from satquery.models.base import BaseTool
from satquery.models.registry import REGISTRY
from satquery.preprocess.constants import CDVQA_ANSWERS
from satquery.serve.contracts import ToolRequest, ToolResult, ToolSpec
from satquery.utils.logging import get_logger
from satquery.utils.paths import artifact_root, configs_dir

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    pass

__all__ = [
    "ChangeHeadConfig",
    "ChangeVqaTool",
    "answer_index",
    "answer_name",
    "build_change_head",
    "build_vocabulary",
    "encode_question",
    "tokenize_question",
]

_LOG = get_logger(__name__)

#: Temporal fusion strategies accepted by `ChangeHeadConfig.fusion`.
FUSION_STRATEGIES: tuple[str, ...] = ("concat", "diff", "abs_diff", "concat_diff")


# --------------------------------------------------------------------------------------
# Closed answer set mapping -- real, pure, and the only place the ordering is used
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class ChangeHeadConfig:
    """Hyperparameters of the two-tower CDVQA head.

    CDVQA is visual *question* answering: the answer to "Have the areas of buildings
    changed?" depends on which class is asked about, so an image-only Siamese network
    cannot express the task. This head therefore has two towers -- a weight-shared image
    encoder over the two dates, and a text encoder over the question -- fused before
    classification.

    The question vocabulary is tiny and templated (297 distinct questions, 47 words), so
    a learned embedding plus GRU is sufficient and needs no pretrained language model.
    """

    encoder_name: str = "resnet18"
    """`timm` backbone, weight-shared across the two dates."""

    pretrained: bool = True
    """ImageNet initialisation before remote sensing adaptation."""

    image_size: int = 256
    """Side length each date is resized to. SECOND ships 512; 256 halves memory."""

    text_embed_dim: int = 64
    text_hidden_dim: int = 128
    """Question tower widths. Small on purpose: 47 words, 297 distinct questions."""

    hidden_dim: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout={self.dropout} must lie in [0.0, 1.0)")
        for name in ("image_size", "hidden_dim", "text_embed_dim", "text_hidden_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def num_classes(self) -> int:
        """Output width: the full closed answer set, frozen in `preprocess/constants.py`."""
        return len(CDVQA_ANSWERS)

    @classmethod
    def from_config(cls, cfg: DictConfig | dict[str, Any]) -> ChangeHeadConfig:
        """Build from an OmegaConf node or mapping, ignoring unknown keys."""
        raw = (
            OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        )
        if not isinstance(raw, dict):
            raise TypeError(f"change head config must be a mapping, got {type(raw)!r}")
        known = {f.name for f in fields(cls)}
        return cls(**{str(k): v for k, v in raw.items() if str(k) in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChangeHeadConfig:
        """Load `configs/model/change_head.yaml`."""
        return cls.from_config(OmegaConf.load(Path(path)))


def build_vocabulary(questions: Iterable[str]) -> dict[str, int]:
    """Build the question token vocabulary, deterministically.

    Sorted so the mapping is reproducible across runs; it is saved beside the weights
    because a different token ordering would silently scramble a reloaded checkpoint in
    the same way a permuted answer set would.

    Index 0 is reserved for padding and unknown tokens.
    """
    tokens = {token for question in questions for token in tokenize_question(question)}
    return {token: index for index, token in enumerate(sorted(tokens), start=1)}


def tokenize_question(question: str) -> list[str]:
    """Lowercase word tokens. The questions are templated, so this is sufficient."""
    return re.findall(r"[a-z0-9_]+", question.lower())


def encode_question(question: str, vocabulary: dict[str, int], max_length: int = 24) -> list[int]:
    """Map a question to a fixed-length id sequence, right-padded with 0."""
    ids = [vocabulary.get(token, 0) for token in tokenize_question(question)][:max_length]
    return ids + [0] * (max_length - len(ids))


def answer_index(answer: str) -> int:
    """Position of `answer` in the frozen `CDVQA_ANSWERS` tuple."""
    normalised = normalize_cdvqa_class(answer)
    for index, candidate in enumerate(CDVQA_ANSWERS):
        if candidate == normalised:
            return index
    raise ValueError(
        f"{answer!r} (normalised {normalised!r}) is not in the closed CDVQA answer set "
        f"{list(CDVQA_ANSWERS)}"
    )


def answer_name(index: int) -> str:
    """Answer string at `index` in the frozen tuple."""
    if not 0 <= index < len(CDVQA_ANSWERS):
        raise ValueError(f"answer index {index} outside [0, {len(CDVQA_ANSWERS)})")
    return CDVQA_ANSWERS[index]


def build_change_head(config: ChangeHeadConfig, vocab_size: int) -> Any:
    """Construct the two-tower module.

    Defined inside a function because it subclasses `torch.nn.Module`, and torch is an
    optional `gpu` extra that must not be imported when the package is merely imported.

    Image features from both dates are combined as ``[f1, f2, |f1 - f2|]``. The absolute
    difference is what actually carries "what changed"; keeping f1 and f2 as well lets
    the head answer questions about *what is there*, not only what moved.
    """
    import timm
    import torch
    from torch import nn

    class SiameseChangeHead(nn.Module):
        """Weight-shared image encoder over two dates, plus a question encoder."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = timm.create_model(
                config.encoder_name, pretrained=config.pretrained, num_classes=0
            )
            image_dim = self.encoder.num_features

            self.embedding = nn.Embedding(vocab_size + 1, config.text_embed_dim, padding_idx=0)
            self.question = nn.GRU(config.text_embed_dim, config.text_hidden_dim, batch_first=True)

            fused = image_dim * 3 + config.text_hidden_dim
            self.classifier = nn.Sequential(
                nn.Linear(fused, config.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.num_classes),
            )

        def forward(
            self,
            pixel_values_t1: Any,
            pixel_values_t2: Any,
            question_ids: Any,
            labels: Any = None,
        ) -> dict[str, Any]:
            first = self.encoder(pixel_values_t1)
            second = self.encoder(pixel_values_t2)
            visual = torch.cat([first, second, torch.abs(first - second)], dim=1)

            _, hidden = self.question(self.embedding(question_ids))
            logits = self.classifier(torch.cat([visual, hidden[-1]], dim=1))

            output: dict[str, Any] = {"logits": logits}
            if labels is not None:
                output["loss"] = nn.functional.cross_entropy(logits, labels)
            return output

    return SiameseChangeHead()


class ChangeVqaTool(BaseTool):
    """`change.vqa_head` -- the scored, discriminative CDVQA answer.

    Deliberately not the VLM. The answer set is closed over 19 values, so a classifier
    beats a generative model on accuracy and runs in milliseconds. `vlm.change_description`
    supplies the prose; the router calls both and reports both.

    The checkpoint and its question vocabulary are loaded lazily on first call, so
    importing this module costs nothing.
    """

    def __init__(self, spec: ToolSpec | None = None, checkpoint: str | Path | None = None) -> None:
        super().__init__(spec or REGISTRY.get_spec("change.vqa_head"))
        self._checkpoint = Path(checkpoint) if checkpoint else None
        self._model: Any = None
        self._vocabulary: dict[str, int] | None = None
        self._config: ChangeHeadConfig | None = None

    def _resolve_checkpoint(self) -> Path:
        """Locate the trained head: the configured path, else the newest checkpoint."""
        if self._checkpoint is not None:
            return self._checkpoint
        root = artifact_root() / "change_head"
        base = root if root.is_dir() else artifact_root() / "train" / "change_head"
        candidates = sorted(
            base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True
        )
        if not candidates:
            raise NotImplementedError(
                f"change.vqa_head has no trained checkpoint under {base}. Train one with "
                "`make train-change`; a fabricated class label would corrupt the CDVQA score."
            )
        return candidates[0]

    def load(self) -> None:
        """Materialise the head and its vocabulary."""
        if self._model is not None:
            return
        import torch
        from safetensors.torch import load_file

        checkpoint = self._resolve_checkpoint()
        vocabulary_path = checkpoint.parent / "question_vocabulary.json"
        if not vocabulary_path.is_file():
            raise NotImplementedError(
                f"question vocabulary not found at {vocabulary_path}. It is written beside "
                "the weights at train time; without it the token ids are meaningless and "
                "predictions would be silently wrong rather than absent."
            )

        self._vocabulary = json.loads(vocabulary_path.read_text())
        self._config = ChangeHeadConfig.from_yaml(configs_dir() / "model" / "change_head.yaml")
        model = build_change_head(self._config, len(self._vocabulary))

        weights = checkpoint / "model.safetensors"
        state = (
            load_file(str(weights))
            if weights.is_file()
            else torch.load(checkpoint / "pytorch_model.bin", map_location="cpu")
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"{checkpoint} is missing {len(missing)} parameter tensors "
                f"(first: {missing[:3]}). Loading it would run a partly random model and "
                "report a plausible but meaningless score."
            )
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        self._model = model
        _LOG.info("change head loaded from %s (unexpected keys: %d)", checkpoint, len(unexpected))

    def _run(self, request: ToolRequest) -> ToolResult:
        """Classify the question over the closed answer set."""
        import torch

        self.load()
        assert self._vocabulary is not None and self._config is not None

        from satquery.data.collate import ChangeHeadCollator

        collator = ChangeHeadCollator(
            vocabulary=self._vocabulary, image_size=self._config.image_size
        )
        sample = _request_to_sample(request)
        batch = collator.encode(sample)
        device = next(self._model.parameters()).device

        with torch.inference_mode():
            output = self._model(
                pixel_values_t1=batch["pixel_values_t1"].unsqueeze(0).to(device),
                pixel_values_t2=batch["pixel_values_t2"].unsqueeze(0).to(device),
                question_ids=batch["question_ids"].unsqueeze(0).to(device),
            )
        probabilities = torch.softmax(output["logits"][0].float(), dim=-1)
        index = int(probabilities.argmax())

        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=answer_name(index),
            confidence=float(probabilities[index]),
            params_used={"answer_set": list(CDVQA_ANSWERS)},
        )


def _request_to_sample(request: ToolRequest) -> Any:
    """Adapt a `ToolRequest` into the `Sample` the collator expects.

    The collator is shared with training on purpose: the exact same tiling, resizing and
    normalisation runs at inference as ran at fit time.
    """
    from satquery.data.schema import AnswerType, Sample
    from satquery.serve.contracts import TaskType

    if len(request.images) != 2:
        raise ValueError(
            f"change.vqa_head needs two dates, got {len(request.images)}. The router's "
            "gate should only route bi-temporal pairs here."
        )
    return Sample(
        sample_id=request.request_id,
        task=TaskType.CHANGE_VQA,
        answer_type=AnswerType.CLOSED_SET,
        images=list(request.images),
        instruction=request.query.strip(),
        # A placeholder label: the collator needs one, and it is never read at inference.
        answer_text=CDVQA_ANSWERS[0],
        source="request",
    )
