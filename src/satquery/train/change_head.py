"""Training surface for the discriminative CDVQA change head.

CDVQA answers are closed over the six land-cover classes frozen in
`preprocess/constants.CDVQA_ANSWERS`, so the scored answer comes from a Siamese
encoder plus a small classification head rather than from generative decoding: it is
more accurate on a closed set and runs in milliseconds. The VLM still produces the
free-form change description; the router calls both and the trace reports both.

Architecture follows the standard Siamese bi-temporal setup used by CDVQA
(Yuan et al., github.com/YZHJessica/CDVQA): one shared image encoder applied to both
dates, the two embeddings combined (concat plus absolute difference), a question
encoding fused in, and a linear classifier over the six classes.

Training runs through `transformers.Trainer` with a custom `compute_loss` -- a plain
cross-entropy over the closed answer set. There is no hand-rolled loop here either.

Real in this module: `ChangeHeadTrainConfig.from_yaml`, `build_training_arguments`,
the fingerprint bookkeeping in `train`. Stubbed with `NotImplementedError`:
`build_model` and `build_datasets`. `torch` and `transformers` are imported inside
function bodies only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from satquery.preprocess.constants import CDVQA_ANSWERS, constants_fingerprint
from satquery.train.callbacks import (
    ConstantsFingerprintCallback,
    ThroughputCallback,
    write_constants_fingerprint,
)
from satquery.utils.logging import get_logger, tracker_backend
from satquery.utils.paths import artifact_root, configs_dir, repo_root
from satquery.utils.seed import DEFAULT_SEED

if TYPE_CHECKING:
    from transformers import Trainer, TrainingArguments

__all__ = [
    "ChangeHeadTrainConfig",
    "build_datasets",
    "build_model",
    "build_trainer",
    "build_training_arguments",
    "train",
]

_LOG = get_logger(__name__)

#: Subdirectory of the run directory the trained head is saved into.
HEAD_SUBDIR = "change_head"

_GPU_EXTRA_HINT = (
    "install the GPU extra to run training: `uv sync --extra gpu` (or `pip install -e '.[gpu]'`)"
)


@dataclass
class ChangeHeadTrainConfig:
    """Every knob for one CDVQA change-head run, loaded from `configs/train/*.yaml`.

    The number of output classes is not a field: it is `len(CDVQA_ANSWERS)`, frozen in
    `preprocess/constants.py`, because the class order is the output-layer order and a
    permutation would silently scramble a reloaded checkpoint.
    """

    # -- what to train on --------------------------------------------------------------
    model_config: str = "configs/model/change_head.yaml"
    """Architecture config for the two-tower head."""

    data_config: str = "configs/data/cdvqa.yaml"

    # -- run identity --------------------------------------------------------------------
    run_name: str = "change_head"
    output_dir: str = "train/change_head"
    seed: int = DEFAULT_SEED

    # -- architecture ---------------------------------------------------------------------
    image_encoder: str = "timm/resnet50.a1_in1k"
    freeze_image_encoder: bool = False
    text_encoder: str = "distilbert-base-uncased"
    freeze_text_encoder: bool = True
    fusion_hidden_dim: int = 512
    classifier_dropout: float = 0.1
    image_size: int = 512
    label_smoothing: float = 0.0

    # -- transformers.TrainingArguments -------------------------------------------------
    num_train_epochs: float = 20.0
    max_steps: int = -1
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    logging_steps: int = 25
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    max_eval_samples: int = 2000
    """Cap on the val slice used for periodic eval, to keep steps cheap."""
    report_to: str | None = None

    # -- callbacks -------------------------------------------------------------------------
    throughput_log_every_n_steps: int = 50

    @property
    def num_labels(self) -> int:
        """Size of the closed CDVQA answer set. Never configurable."""
        return len(CDVQA_ANSWERS)

    def resolved_output_dir(self) -> Path:
        """Return `output_dir` absolute, relative values going under the artifact root."""
        candidate = Path(self.output_dir).expanduser()
        if candidate.is_absolute():
            return candidate
        return artifact_root() / candidate

    def resolved_model_config(self) -> Path:
        """Absolute path of the head architecture config."""
        return _resolve_config_path(self.model_config)

    def resolved_data_config(self) -> Path:
        """Absolute path of the data YAML, relative values resolving against the repo root."""
        return _resolve_config_path(self.data_config)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain mapping, for logging and run metadata."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChangeHeadTrainConfig:
        """Load a config from YAML with OmegaConf.

        Args:
            path: YAML file. Relative paths resolve against the repo root, then
                against `configs/train/`.

        Returns:
            A populated `ChangeHeadTrainConfig`.

        Raises:
            FileNotFoundError: The YAML does not exist.
            ValueError: The YAML is not a mapping, or sets a key this dataclass does
                not declare.
        """
        resolved = _resolve_config_path(path, extra_dir=configs_dir() / "train")
        if not resolved.is_file():
            raise FileNotFoundError(f"change-head training config not found: {resolved}")

        raw = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
        if not isinstance(raw, dict):
            raise ValueError(f"{resolved} must contain a YAML mapping, got {type(raw).__name__}")

        known = {f_.name for f_ in cls.__dataclass_fields__.values()}
        unknown = sorted(set(map(str, raw)) - known)
        if unknown:
            raise ValueError(
                f"{resolved} sets unknown key(s) {unknown}; known keys are {sorted(known)}"
            )
        return cls(**{str(key): value for key, value in raw.items()})


def _resolve_config_path(path: str | Path, extra_dir: Path | None = None) -> Path:
    """Resolve a config path: absolute as given, else repo-relative, else under `extra_dir`."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    repo_relative = repo_root() / candidate
    if repo_relative.exists() or extra_dir is None:
        return repo_relative
    in_extra = extra_dir / candidate
    return in_extra if in_extra.exists() else repo_relative


def build_training_arguments(cfg: ChangeHeadTrainConfig) -> TrainingArguments:
    """Build `transformers.TrainingArguments` for the change-head run.

    `report_to` falls back to `satquery.utils.logging.tracker_backend()`.

    Raises:
        ImportError: `transformers` is not installed.
    """
    try:
        from transformers import TrainingArguments
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(
            f"transformers is required to build TrainingArguments; {_GPU_EXTRA_HINT}"
        ) from exc

    output_dir = cfg.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=cfg.run_name,
        seed=cfg.seed,
        data_seed=cfg.seed,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=cfg.remove_unused_columns,
        report_to=[cfg.report_to or tracker_backend()],
    )


def build_vocabulary_for(cfg: ChangeHeadTrainConfig) -> dict[str, int]:
    """Build the question vocabulary from the TRAINING split only.

    Training-split only, deliberately: building it from test questions would leak the
    evaluation distribution into the model's input encoding.
    """
    from satquery.data.datasets.cdvqa import CDVQADataset
    from satquery.models.change.siamese import build_vocabulary

    train = CDVQADataset(split="train")
    return build_vocabulary(train[i].instruction for i in range(len(train)))


def build_model(cfg: ChangeHeadTrainConfig, vocab_size: int | None = None) -> Any:
    """Construct the two-tower head.

    Args:
        cfg: Run configuration.
        vocab_size: Question vocabulary size. Derived from the training split when not
            supplied.
    """
    from satquery.models.change.siamese import ChangeHeadConfig, build_change_head

    head_config = ChangeHeadConfig.from_yaml(cfg.resolved_model_config())
    size = vocab_size if vocab_size is not None else len(build_vocabulary_for(cfg))
    model = build_change_head(head_config, size)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _LOG.info(
        "change head: %s encoder, %d answers, vocab %d, %.2fM trainable",
        head_config.encoder_name,
        head_config.num_classes,
        size,
        trainable / 1e6,
    )
    return model


def build_datasets(cfg: ChangeHeadTrainConfig) -> tuple[Any, Any | None]:
    """Return `(train, eval)` CDVQA views.

    The eval view is the `val` split, never `test_1` or `test_2`: those are the scored
    splits and must stay unseen until the final measurement.
    """
    from satquery.data.datasets.cdvqa import CDVQADataset
    from satquery.data.mixer import SubsetSampleDataset

    train = CDVQADataset(split="train")
    validation = CDVQADataset(split="val")
    len(train), len(validation)

    limit = min(cfg.max_eval_samples, len(validation))
    _LOG.info("CDVQA train=%d val=%d (eval capped at %d)", len(train), len(validation), limit)
    return train, SubsetSampleDataset(validation, range(limit))


def _compute_metrics(prediction: Any) -> dict[str, float]:
    """Top-1 accuracy over the closed answer set."""
    import numpy as np

    logits, labels = prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    return {"accuracy": float((np.asarray(logits).argmax(-1) == np.asarray(labels)).mean())}


def build_trainer(cfg: ChangeHeadTrainConfig) -> Trainer:
    """Assemble the `Trainer` for the change head.

    Calls `build_model` and `build_datasets` and therefore fails honestly until both
    are written.

    Raises:
        ImportError: `transformers` is not installed.
        NotImplementedError: Propagated from `build_model` or `build_datasets`.
    """
    try:
        from transformers import Trainer
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(
            f"transformers is required to build a Trainer; {_GPU_EXTRA_HINT}"
        ) from exc

    from satquery.data.collate import ChangeHeadCollator
    from satquery.models.change.siamese import ChangeHeadConfig

    args = build_training_arguments(cfg)
    vocabulary = build_vocabulary_for(cfg)
    model = build_model(cfg, vocab_size=len(vocabulary))
    train_dataset, eval_dataset = build_datasets(cfg)

    head_config = ChangeHeadConfig.from_yaml(cfg.resolved_model_config())
    collator = ChangeHeadCollator(vocabulary=vocabulary, image_size=head_config.image_size)

    # Saved beside the weights: a different token ordering scrambles a reloaded
    # checkpoint exactly as a permuted answer set would.
    output = cfg.resolved_output_dir()
    output.mkdir(parents=True, exist_ok=True)
    (output / "question_vocabulary.json").write_text(json.dumps(vocabulary, indent=2))

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        callbacks=[
            ConstantsFingerprintCallback(
                run_dir=cfg.resolved_output_dir(),
                run_name=cfg.run_name,
                data_config=str(cfg.resolved_data_config()),
            ),
            ThroughputCallback(log_every_n_steps=cfg.throughput_log_every_n_steps),
        ],
    )


def train(cfg: ChangeHeadTrainConfig) -> Path:
    """Run change-head training and return the directory the head was saved to.

    Raises:
        NotImplementedError: Propagated from `build_trainer` while the head module
            and the CDVQA loader are unwritten.
    """
    from satquery.utils.seed import seed_everything

    seed_everything(cfg.seed)

    output_dir = cfg.resolved_output_dir()
    fingerprint = constants_fingerprint()
    record = write_constants_fingerprint(
        output_dir,
        run_name=cfg.run_name,
        stage="change_head",
        cdvqa_classes=list(CDVQA_ANSWERS),
        config=cfg.to_dict(),
    )
    _LOG.info("run %s -> %s", cfg.run_name, output_dir)
    _LOG.info("frozen preprocessing constants fingerprint %s (recorded at %s)", fingerprint, record)
    _LOG.info("seed %d, tracker %s", cfg.seed, cfg.report_to or tracker_backend())

    trainer = build_trainer(cfg)
    trainer.train()

    head_dir = output_dir / HEAD_SUBDIR
    trainer.save_model(str(head_dir))
    write_constants_fingerprint(
        head_dir, run_name=cfg.run_name, stage="change_head", cdvqa_classes=list(CDVQA_ANSWERS)
    )
    _LOG.info("change head saved to %s", head_dir)
    return head_dir
