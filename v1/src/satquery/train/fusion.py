"""Training for the dual-encoder optical + SAR fusion segmenter.

Runs through `transformers.Trainer`, the same as every other trainer in this repo --
no hand-written loop. The model returns a `{"loss", "logits"}` mapping, which is the
only contract `Trainer` needs, so the custom part here is entirely in what gets
*measured*, not in how the steps are taken.

**Accuracy is deliberately not reported.** The extraction classes are heavily
zero-inflated: over 400 sampled reBEN patches, built-up is absent from 84% of them and
water from 73%, and the median patch contains neither. A model that predicts background
everywhere therefore scores about 96% pixel accuracy while extracting nothing at all.
Per-class IoU is the metric that cannot be gamed that way -- an empty prediction scores
zero on it -- so IoU is what `metric_for_best_model` selects on, and accuracy never
appears in the log.

Metrics are reduced *before* accumulation. `Trainer` otherwise gathers every logit of
every eval sample, which for a segmenter is `(N, T, H, W)` floats and runs into
gigabytes. `preprocess_logits_for_metrics` collapses each sample to per-class
intersection and union counts first, so evaluation memory stays flat in the number of
samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from satquery.preprocess.constants import FUSION_EXTRACTION_CLASSES, constants_fingerprint
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
    "FusionTrainConfig",
    "build_datasets",
    "build_model",
    "build_trainer",
    "build_training_arguments",
    "train",
]

_LOG = get_logger(__name__)

#: Subdirectory of the run directory the trained model is saved into.
MODEL_SUBDIR = "fusion"

#: Filename of the portable checkpoint `DualEncoderFusion.load_checkpoint` reads.
CHECKPOINT_FILENAME = "model.pt"

_GPU_EXTRA_HINT = (
    "install the GPU extra to run training: `uv sync --extra gpu` (or `pip install -e '.[gpu]'`)"
)


def _resolve_config_path(path: str | Path, extra_dir: Path | None = None) -> Path:
    """Resolve a config path against the repo root, then against `extra_dir`."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    from_root = repo_root() / candidate
    if from_root.is_file() or extra_dir is None:
        return from_root
    return extra_dir / candidate


@dataclass
class FusionTrainConfig:
    """Every knob for one fusion run, loaded from `configs/train/fusion.yaml`."""

    # -- what to train on ----------------------------------------------------------------
    model_config: str = "configs/model/fusion.yaml"
    data_root: str | None = None
    """reBEN corpus directory. `None` uses `dataset_dir('bigearthnet_txt')`."""
    lmdb_path: str | None = None
    """Converted LMDB. `None` uses `<data_root>/../bigearthnet_lmdb`."""
    packed_cache: str | None = None
    """Directory of a packed cache from `scripts/pack_fusion_cache.py`, holding one
    subdirectory per split. When set, training reads from it instead of the LMDB.

    This is normally required in practice, not an optimisation. The LMDB sits on a
    spinning USB drive where random reads run about a thousand times slower than
    sequential ones, so a shuffled epoch against it takes days. The packer streams the
    corpus once; this reads the result at local-disk speed."""

    train_split: str = "train"
    eval_split: str = "validation"
    max_train_samples: int = -1
    """Cap on the training split, for quick runs. -1 uses all of it."""
    max_eval_samples: int = 2000
    """Cap on the eval slice, to keep periodic evaluation cheap."""

    # -- run identity --------------------------------------------------------------------
    run_name: str = "fusion_stage1"
    output_dir: str = "train/fusion_stage1"
    seed: int = DEFAULT_SEED

    # -- transformers.TrainingArguments ---------------------------------------------------
    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 50
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_mean_iou"
    greater_is_better: bool = True
    dataloader_num_workers: int = 8
    remove_unused_columns: bool = False
    report_to: str | None = None

    # -- callbacks -------------------------------------------------------------------------
    throughput_log_every_n_steps: int = 50

    def resolved_output_dir(self) -> Path:
        """Return `output_dir` absolute, relative values going under the artifact root."""
        candidate = Path(self.output_dir).expanduser()
        return candidate if candidate.is_absolute() else artifact_root() / candidate

    def resolved_model_config(self) -> Path:
        """Absolute path of the architecture config."""
        return _resolve_config_path(self.model_config)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain mapping, for logging and run metadata."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FusionTrainConfig:
        """Load a config from YAML with OmegaConf.

        Raises:
            FileNotFoundError: The YAML does not exist.
            ValueError: The YAML is not a mapping, or sets an undeclared key.
        """
        resolved = _resolve_config_path(path, extra_dir=configs_dir() / "train")
        if not resolved.is_file():
            raise FileNotFoundError(f"fusion training config not found: {resolved}")

        raw = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
        if not isinstance(raw, dict):
            raise ValueError(f"{resolved} must contain a YAML mapping, got {type(raw).__name__}")

        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(map(str, raw)) - known)
        if unknown:
            raise ValueError(
                f"{resolved} sets unknown key(s) {unknown}; known keys are {sorted(known)}"
            )
        return cls(**{str(key): value for key, value in raw.items()})


def build_training_arguments(cfg: FusionTrainConfig) -> TrainingArguments:
    """Build `transformers.TrainingArguments` for the fusion run.

    Raises:
        ImportError: `transformers` is not installed.
    """
    try:
        from transformers import TrainingArguments
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(
            f"transformers is required to build TrainingArguments; {_GPU_EXTRA_HINT}"
        ) from exc

    return TrainingArguments(
        output_dir=str(cfg.resolved_output_dir()),
        run_name=cfg.run_name,
        seed=cfg.seed,
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
        report_to=cfg.report_to or tracker_backend(),
        label_names=["labels"],
    )


def build_model(cfg: FusionTrainConfig) -> Any:
    """Instantiate the dual encoder from the architecture config."""
    from satquery.models.fusion.dual_encoder import FusionConfig, build_fusion_model

    fusion_config = FusionConfig.from_yaml(cfg.resolved_model_config())
    _LOG.info(
        "fusion architecture: optical=%s sar=%s strategy=%s targets=%s",
        fusion_config.optical_encoder,
        fusion_config.sar_encoder,
        fusion_config.fusion,
        list(fusion_config.targets),
    )
    return build_fusion_model(fusion_config)


def build_datasets(cfg: FusionTrainConfig) -> tuple[Any, Any]:
    """Return `(train_dataset, eval_dataset)` of co-registered reBEN pairs."""
    from torch.utils.data import Subset

    from satquery.data.datasets.bigearthnet_fusion import (
        BigEarthNetFusionDataset,
        PackedFusionDataset,
    )

    if cfg.packed_cache:
        root = Path(cfg.packed_cache).expanduser()
        train = PackedFusionDataset(root / cfg.train_split)
        evaluation = PackedFusionDataset(root / cfg.eval_split)
        _LOG.info(
            "packed cache: %d train / %d eval patches from %s",
            len(train),
            len(evaluation),
            root,
        )
        if 0 < cfg.max_train_samples < len(train):
            train = Subset(train, list(range(cfg.max_train_samples)))
        if 0 < cfg.max_eval_samples < len(evaluation):
            evaluation = Subset(evaluation, list(range(cfg.max_eval_samples)))
        return train, evaluation

    def _load(split: str, cap: int) -> Any:
        dataset = BigEarthNetFusionDataset(
            root=Path(cfg.data_root) if cfg.data_root else None,
            split=split,
            lmdb_path=Path(cfg.lmdb_path) if cfg.lmdb_path else None,
        )
        if cap > 0 and cap < len(dataset):
            # A contiguous head slice, not a random sample: reBEN rows are grouped by
            # Sentinel tile, so a slice is a geographic subset and a capped run is
            # explicitly *not* representative. Saying so here beats a quiet random
            # subsample that looks representative and is not.
            _LOG.warning(
                "capping %s to the first %d of %d rows; reBEN is ordered by tile, so this "
                "slice covers only a few scenes and its scores do not generalise",
                split,
                cap,
                len(dataset),
            )
            return Subset(dataset, list(range(cap)))
        return dataset

    return _load(cfg.train_split, cfg.max_train_samples), _load(
        cfg.eval_split, cfg.max_eval_samples
    )


def _reduce_logits_for_metrics(logits: Any, labels: Any) -> Any:
    """Collapse a batch of logit maps to per-class intersection and union counts.

    Called by `Trainer` before eval predictions are accumulated. Without it the whole
    `(N, targets, H, W)` logit tensor is gathered into memory, which for a segmenter is
    gigabytes; with it, evaluation costs two integers per class per sample.

    Returns:
        Tensor of shape `(batch, targets, 2)` holding `[intersection, union]`.
    """
    import torch

    predicted = (torch.sigmoid(logits) > 0.5).to(torch.bool)
    truth = labels.to(torch.bool)
    intersection = (predicted & truth).sum(dim=(-2, -1))
    union = (predicted | truth).sum(dim=(-2, -1))
    return torch.stack([intersection, union], dim=-1).to(torch.float32)


def _compute_metrics(prediction: Any) -> dict[str, float]:
    """Per-class IoU and their mean, from the reduced counts.

    IoU is aggregated over the whole eval set rather than averaged per image. A
    per-image mean is dominated by the many patches that contain no positives at all,
    where IoU is either undefined or trivially perfect, and it drifts with batch
    composition. Summing intersections and unions first gives a dataset-level figure
    that is stable and comparable between runs.

    A class with no positives anywhere in the eval set reports `nan`, never `0.0` or
    `1.0`: both would be a fabricated score for something that was never measured.
    """
    import numpy as np

    counts = prediction.predictions
    if isinstance(counts, tuple):
        counts = counts[0]
    counts = np.asarray(counts)

    intersection = counts[..., 0].sum(axis=0)
    union = counts[..., 1].sum(axis=0)

    metrics: dict[str, float] = {}
    observed: list[float] = []
    for index, name in enumerate(FUSION_EXTRACTION_CLASSES[: len(union)]):
        if union[index] == 0:
            metrics[f"iou_{name}"] = float("nan")
            continue
        value = float(intersection[index] / union[index])
        metrics[f"iou_{name}"] = value
        observed.append(value)

    metrics["mean_iou"] = float(np.mean(observed)) if observed else float("nan")
    return metrics


def build_trainer(cfg: FusionTrainConfig) -> Trainer:
    """Assemble the `Trainer` for the fusion segmenter.

    Raises:
        ImportError: `transformers` is not installed.
    """
    try:
        from transformers import Trainer
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(
            f"transformers is required to build a Trainer; {_GPU_EXTRA_HINT}"
        ) from exc

    from satquery.data.collate import FusionCollator, PackedFusionCollator
    from satquery.models.fusion.dual_encoder import FusionConfig

    fusion_config = FusionConfig.from_yaml(cfg.resolved_model_config())
    args = build_training_arguments(cfg)
    model = build_model(cfg)
    train_dataset, eval_dataset = build_datasets(cfg)

    if cfg.packed_cache:

        def _backing(dataset: Any) -> Any:
            return dataset.dataset if hasattr(dataset, "dataset") else dataset

        registered = {
            str(_backing(dataset).cache_dir): _backing(dataset)
            for dataset in (train_dataset, eval_dataset)
        }
        collator: Any = PackedFusionCollator(datasets=registered, targets=fusion_config.targets)
    else:
        collator = FusionCollator(
            targets=fusion_config.targets,
            optical_bands=fusion_config.optical_in_channels,
        )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        preprocess_logits_for_metrics=_reduce_logits_for_metrics,
        callbacks=[
            ConstantsFingerprintCallback(
                run_dir=cfg.resolved_output_dir(),
                run_name=cfg.run_name,
                data_config="bigearthnet_fusion",
            ),
            ThroughputCallback(log_every_n_steps=cfg.throughput_log_every_n_steps),
        ],
    )


def train(cfg: FusionTrainConfig) -> Path:
    """Run fusion training and return the path of the portable checkpoint.

    Two artifacts are written. `Trainer` saves its own checkpoint directory for
    resuming, and `DualEncoderFusion.save_checkpoint` writes the single `model.pt` the
    serving tool loads -- carrying the architecture config and the constants
    fingerprint, so inference can refuse a checkpoint whose preprocessing has drifted.
    """
    from satquery.models.fusion.dual_encoder import DualEncoderFusion, FusionConfig
    from satquery.utils.seed import seed_everything

    seed_everything(cfg.seed)

    output_dir = cfg.resolved_output_dir()
    fingerprint = constants_fingerprint()
    record = write_constants_fingerprint(
        output_dir,
        run_name=cfg.run_name,
        stage="fusion",
        targets=list(FUSION_EXTRACTION_CLASSES),
        config=cfg.to_dict(),
    )
    _LOG.info("run %s -> %s", cfg.run_name, output_dir)
    _LOG.info("frozen preprocessing constants fingerprint %s (recorded at %s)", fingerprint, record)
    _LOG.info("seed %d, tracker %s", cfg.seed, cfg.report_to or tracker_backend())

    trainer = build_trainer(cfg)
    trainer.train()

    model_dir = output_dir / MODEL_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)

    fusion_config = FusionConfig.from_yaml(cfg.resolved_model_config())
    wrapper = DualEncoderFusion.from_module(fusion_config, trainer.model)
    checkpoint = wrapper.save_checkpoint(model_dir / CHECKPOINT_FILENAME)

    write_constants_fingerprint(
        model_dir,
        run_name=cfg.run_name,
        stage="fusion",
        targets=list(FUSION_EXTRACTION_CLASSES),
    )
    _LOG.info("fusion checkpoint saved to %s", checkpoint)
    return checkpoint
