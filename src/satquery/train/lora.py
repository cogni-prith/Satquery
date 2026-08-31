"""LoRA adaptation of the EarthDial-4B backbone on the BigEarthNet.txt-led mix.

This is the run that produces the "RS adaptation of a vision or VL component"
artifact the problem statement makes mandatory: a training log, adapter weights and
a before/after table. A generic VLM with no remote-sensing adaptation fails SIH26167
outright, so the configuration surface here is deliberately complete and fully
loggable even though two pieces of the assembly are not written yet.

Method follows LoRA (Hu et al., 2021, arXiv:2106.09685) applied through
`peft.LoraConfig`, driven by `transformers.Trainer`. There is no hand-rolled
training loop in this repo and there must never be one.

Backbone: `akshaydudhane/EarthDial_4B_RGB` (EarthDial, InternVL-based). The
non-optical checkpoints are separate registry entries and get their own config file
rather than a branch in this module.

What is real here and what is not:

- `LoraTrainConfig.from_yaml`, `build_peft_config`, `build_training_arguments` and the
  fingerprint bookkeeping in `train` are implemented.
- `build_model` and `build_datasets` raise `NotImplementedError` naming exactly what
  is missing. `build_trainer` and `train` call them, so an attempt to train fails
  loudly at the missing piece instead of quietly training on nothing.

`torch`, `transformers` and `peft` are optional `gpu` extras and are imported inside
function bodies only, so this module imports on a CPU-only laptop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from satquery.preprocess.constants import constants_fingerprint
from satquery.train.callbacks import (
    ConstantsFingerprintCallback,
    ThroughputCallback,
    write_constants_fingerprint,
)
from satquery.utils.logging import get_logger, tracker_backend
from satquery.utils.paths import artifact_root, configs_dir, repo_root
from satquery.utils.seed import DEFAULT_SEED

if TYPE_CHECKING:
    from peft import LoraConfig
    from transformers import Trainer, TrainingArguments

__all__ = [
    "LoraTrainConfig",
    "build_datasets",
    "build_model",
    "build_peft_config",
    "build_trainer",
    "build_training_arguments",
    "train",
]

_LOG = get_logger(__name__)

#: Name the adapter directory is saved under, inside the run output directory.
ADAPTER_SUBDIR = "adapter"

_GPU_EXTRA_HINT = (
    "install the GPU extra to run training: `uv sync --extra gpu` (or `pip install -e '.[gpu]'`)"
)


@dataclass
class LoraTrainConfig:
    """Every knob for one LoRA adaptation run, loaded from `configs/train/*.yaml`.

    Single responsibility: hold the run definition and nothing else. It builds no
    objects; the `build_*` functions read it. Keys map one-for-one onto the YAML so
    a diff of two config files is a diff of two runs.
    """

    # -- what to train on ------------------------------------------------------------
    model_config: str = "configs/model/earthdial_4b_rgb.yaml"
    data_config: str = "configs/data/mix_v1.yaml"

    # -- run identity ----------------------------------------------------------------
    run_name: str = "lora_stage1"
    output_dir: str = "train/lora_stage1"
    seed: int = DEFAULT_SEED

    # -- LoRA hyperparameters (peft.LoraConfig) ---------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    # None, not "CAUSAL_LM". InternVLChatModel is not a standard causal LM: its forward
    # takes pixel_values and image_flags and does NOT accept inputs_embeds, which
    # PeftModelForCausalLM passes unconditionally. A task_type of None gives the generic
    # PeftModel, whose forward hands every keyword through untouched.
    lora_task_type: str | None = None
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_modules_to_save: list[str] = field(default_factory=list)

    # -- transformers.TrainingArguments ------------------------------------------------
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    max_sequence_length: int = 2048

    resume_from_checkpoint: str | None = None
    """Checkpoint directory to continue from, relative to the run output dir or absolute.

    `"auto"` picks the highest-numbered `checkpoint-N` already in the output directory.
    Resuming restores the optimizer and LR-scheduler state, not just the weights: a
    weights-only restart would re-run the warmup and discard the momentum estimates,
    which is a different training trajectory from the one being continued.
    """
    report_to: str | None = None

    # -- callbacks ---------------------------------------------------------------------
    throughput_log_every_n_steps: int = 50

    # -- derived ------------------------------------------------------------------------

    def resolved_output_dir(self) -> Path:
        """Return `output_dir` as an absolute path, relative paths going under the artifact root.

        Checkpoints and logs are gitignored build products, so a relative value in
        YAML means "somewhere under `$SATQUERY_ARTIFACT_ROOT`", never the repo.
        """
        candidate = Path(self.output_dir).expanduser()
        if candidate.is_absolute():
            return candidate
        return artifact_root() / candidate

    def resolved_model_config(self) -> Path:
        """Absolute path of the model YAML, relative values resolving against the repo root."""
        return _resolve_config_path(self.model_config)

    def resolved_data_config(self) -> Path:
        """Absolute path of the data-mix YAML, relative values resolving against the repo root."""
        return _resolve_config_path(self.data_config)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain mapping, for logging and run metadata."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> LoraTrainConfig:
        """Load a config from YAML with OmegaConf.

        Args:
            path: YAML file. A relative path resolves against the repo root, then
                against `configs/train/`.

        Returns:
            A populated `LoraTrainConfig`.

        Raises:
            FileNotFoundError: The YAML does not exist.
            ValueError: The YAML contains a key this dataclass does not declare. A
                silently ignored typo in a hyperparameter is a wasted GPU day.
        """
        resolved = _resolve_config_path(path, extra_dir=configs_dir() / "train")
        if not resolved.is_file():
            raise FileNotFoundError(f"LoRA training config not found: {resolved}")

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


def build_peft_config(cfg: LoraTrainConfig) -> LoraConfig:
    """Build the `peft.LoraConfig` for this run.

    Args:
        cfg: Run configuration.

    Returns:
        A `peft.LoraConfig` ready to hand to `peft.get_peft_model`.

    Raises:
        ImportError: `peft` is not installed.
    """
    try:
        from peft import LoraConfig
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(f"peft is required to build a LoRA config; {_GPU_EXTRA_HINT}") from exc

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type=cfg.lora_task_type or None,
        target_modules=list(cfg.lora_target_modules),
        modules_to_save=list(cfg.lora_modules_to_save) or None,
    )


def build_training_arguments(cfg: LoraTrainConfig) -> TrainingArguments:
    """Build `transformers.TrainingArguments` from the run configuration.

    `report_to` falls back to `satquery.utils.logging.tracker_backend()`, which is
    Weights and Biases when `WANDB_API_KEY` is set and TensorBoard otherwise.

    Args:
        cfg: Run configuration.

    Returns:
        Populated `TrainingArguments` whose `output_dir` is the resolved run directory.

    Raises:
        ImportError: `transformers` is not installed.
    """
    try:
        from transformers import TrainingArguments
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(
            f"transformers is required to build TrainingArguments; {_GPU_EXTRA_HINT}"
        ) from exc

    report_to = cfg.report_to or tracker_backend()
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
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=cfg.remove_unused_columns,
        report_to=[report_to],
    )


def build_model(cfg: LoraTrainConfig) -> Any:
    """Load the EarthDial backbone in 4-bit and attach LoRA adapters to it.

    The base weights are frozen and quantised to NF4; only the adapters train. On an
    8 GB card that is not an optimisation, it is the difference between running and not:
    the bf16 weights alone are 8.29 GB.

    `task_type` is deliberately left unset on the LoRA config. InternVLChatModel is not a
    standard `CausalLM` -- its `forward` takes `pixel_values` and `image_flags` alongside
    `input_ids` -- so a generic `PeftModel`, which forwards every keyword argument
    untouched, is the safe wrapper.
    """
    from peft import get_peft_model, prepare_model_for_kbit_training

    from satquery.models.vlm.backbone import BackboneConfig, EarthDialBackbone

    backbone_config = BackboneConfig.from_yaml(cfg.resolved_model_config())

    # Training always starts from the BASE weights. The model config may name an adapter
    # so that inference and `make eval` load it, but attaching it here would wrap a
    # PeftModel in a second PeftModel and train a fresh LoRA on top of the old one --
    # a different thing entirely from continuing the existing adaptation. Continuation is
    # `resume_from_checkpoint`, which restores adapter, optimizer and scheduler together.
    if backbone_config.is_adapted:
        _LOG.info(
            "ignoring adapter_path=%s for training; continuation is handled by "
            "resume_from_checkpoint",
            backbone_config.adapter_path,
        )
        backbone_config = replace(backbone_config, adapter_path=None)

    backbone = EarthDialBackbone(backbone_config)
    backbone.load(allow_download=False)
    model = backbone.model

    if backbone_config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Inputs reach the checkpointed blocks as embeddings, so without this the graph
        # has no input requiring grad and every backward pass silently does nothing.
        model.enable_input_require_grads()

    peft_model = get_peft_model(model, build_peft_config(cfg))

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    _LOG.info(
        "LoRA attached: %.2fM trainable of %.2fM total (%.3f%%)",
        trainable / 1e6,
        total / 1e6,
        100.0 * trainable / max(total, 1),
    )
    if trainable == 0:
        raise RuntimeError(
            "LoRA produced zero trainable parameters. `lora_target_modules` "
            f"{cfg.lora_target_modules} matched nothing in the loaded model."
        )

    peft_model._satquery_tokenizer = backbone.tokenizer
    peft_model._satquery_backbone_config = backbone_config
    return peft_model


def build_datasets(cfg: LoraTrainConfig) -> tuple[Any, Any | None]:
    """Build `(train_dataset, eval_dataset)` from the configured mix.

    Components whose corpus is not on disk are skipped with a warning rather than
    aborting the run. Nothing is downloaded automatically, and on this machine
    BigEarthNet.txt's imagery (~120 GB) does not fit, so a run that insisted on the full
    40/40/20 mix could never start. What actually trained is logged, and the mix weights
    that were honoured are logged with it, so a later result is never mistaken for the
    full mix.
    """
    from satquery.data.mixer import load_mix_spec

    spec = load_mix_spec(cfg.resolved_data_config())
    builders = {"vrsbench": _build_vrsbench, "cdvqa": _build_cdvqa}

    available: list[Any] = []
    honoured: dict[str, float] = {}
    for component in spec.components:
        builder = builders.get(component.name)
        if builder is None:
            _LOG.warning("no loader wired for mix component %r; skipped", component.name)
            continue
        try:
            parts = builder(component)
        except (FileNotFoundError, NotImplementedError) as exc:
            # Two reasons a component drops out, both legitimate and both loud: its
            # corpus is not on disk, or its loader is still a stub. Either way the run
            # continues on what IS available rather than refusing to start.
            _LOG.warning("mix component %r skipped: %s", component.name, exc)
            continue
        available.extend(parts)
        honoured[component.name] = component.weight

    if not available:
        raise FileNotFoundError(
            "no mix component has data on disk. Fetch at least one corpus; "
            "scripts/download_datasets.py prints the commands."
        )

    _LOG.info(
        "training on %d component view(s) from %s; weights honoured: %s",
        len(available),
        sorted(honoured),
        honoured,
    )
    if len(honoured) < len(spec.components):
        missing = [c.name for c in spec.components if c.name not in honoured]
        _LOG.warning(
            "MIX INCOMPLETE: %s absent, so the trained distribution is NOT %s. Any result "
            "from this run must say so.",
            missing,
            spec.weights,
        )

    from satquery.data.mixer import ConcatSampleDataset

    train = ConcatSampleDataset(available)
    eval_dataset = _build_eval_view(cfg)
    return train, eval_dataset


def _build_vrsbench(component: Any) -> list[Any]:
    """One dataset view per VRSBench subset, so all three tasks are adapted."""
    from satquery.data.datasets.vrsbench import VRSBenchDataset

    subsets = component.options.get("subsets") or ["caption", "vqa", "referring"]
    views = [VRSBenchDataset(split="train", subset=s) for s in subsets]
    for view in views:
        len(view)  # forces the index load, so a missing corpus fails here
    return views


def _build_cdvqa(component: Any) -> list[Any]:
    """CDVQA training view. Raises FileNotFoundError when the corpus is absent."""
    from satquery.data.datasets.cdvqa import CDVQADataset

    view = CDVQADataset(split="train")
    len(view)
    return [view]


def _build_eval_view(cfg: LoraTrainConfig) -> Any | None:
    """A small held-out slice for periodic loss, or None when unavailable."""
    from satquery.data.datasets.vrsbench import VRSBenchDataset
    from satquery.data.mixer import SubsetSampleDataset

    try:
        view = VRSBenchDataset(split="val", subset="vqa")
        return SubsetSampleDataset(view, range(min(64, len(view))))
    except (FileNotFoundError, NotImplementedError):
        _LOG.warning("no eval corpus available; training will run without eval")
        return None


def build_trainer(cfg: LoraTrainConfig) -> Trainer:
    """Assemble the `transformers.Trainer` for this run.

    Calls `build_model` and `build_datasets`, so until those are written this raises
    `NotImplementedError` from whichever is reached first. That is deliberate: a
    Trainer that silently trained on a placeholder dataset would pass a smoke test
    and ruin an eval three weeks later.

    Args:
        cfg: Run configuration.

    Returns:
        A `Trainer` wired with the LoRA-wrapped model, the mixed dataset and both
        bookkeeping callbacks.

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

    from satquery.data.collate import InternVLCollator

    args = build_training_arguments(cfg)
    model = build_model(cfg)
    train_dataset, eval_dataset = build_datasets(cfg)

    backbone_config = model._satquery_backbone_config
    collator = InternVLCollator(
        tokenizer=model._satquery_tokenizer,
        model=model,
        image_size=backbone_config.image_size,
        # The tile budget is read from the SAME model config the eval path reads, so
        # training and inference cannot silently tile differently. On an 8 GB card this
        # must stay at 1: seven tiles is ~1,792 image tokens before any text.
        max_tiles=backbone_config.max_tiles,
        max_length=cfg.max_sequence_length,
    )
    _LOG.info(
        "collator: image_size=%d max_tiles=%d -> %d image tokens per sample",
        backbone_config.image_size,
        backbone_config.max_tiles,
        collator.num_image_token * backbone_config.max_tiles,
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[
            ConstantsFingerprintCallback(
                run_dir=cfg.resolved_output_dir(),
                run_name=cfg.run_name,
                model_config=str(cfg.resolved_model_config()),
                data_config=str(cfg.resolved_data_config()),
            ),
            ThroughputCallback(log_every_n_steps=cfg.throughput_log_every_n_steps),
        ],
    )


def _resolve_resume(cfg: LoraTrainConfig, output_dir: Path) -> Path | None:
    """Resolve `cfg.resume_from_checkpoint` to a directory, or None for a fresh run.

    Raises rather than silently starting from scratch when a checkpoint was asked for and
    cannot be found: a run that quietly restarts from step 0 wastes GPU hours and its loss
    curve looks plausible.
    """
    requested = cfg.resume_from_checkpoint
    if not requested:
        return None

    if requested == "auto":
        candidates = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]),
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"resume_from_checkpoint='auto' but no checkpoint-* exists under "
                f"{output_dir}. Set it to null for a fresh run."
            )
        return candidates[0]

    path = Path(requested)
    path = path if path.is_absolute() else output_dir / path
    if not path.is_dir():
        raise FileNotFoundError(
            f"resume_from_checkpoint={requested!r} resolves to {path}, which does not "
            "exist. Set it to null for a fresh run rather than restarting silently."
        )
    return path


def train(cfg: LoraTrainConfig) -> Path:
    """Run the adaptation and return the directory the adapter was saved to.

    The frozen-constants fingerprint is logged and written into the run directory
    before anything else happens, so a run that dies mid-training still leaves behind
    the preprocessing contract it was started under.

    Args:
        cfg: Run configuration.

    Returns:
        Path of the saved LoRA adapter directory.

    Raises:
        NotImplementedError: Propagated from `build_trainer` while the backbone
            loading and dataset construction are unwritten.
    """
    from satquery.utils.seed import seed_everything

    seed_everything(cfg.seed)

    output_dir = cfg.resolved_output_dir()
    fingerprint = constants_fingerprint()
    record = write_constants_fingerprint(
        output_dir,
        run_name=cfg.run_name,
        stage="lora",
        config=cfg.to_dict(),
    )
    _LOG.info("run %s -> %s", cfg.run_name, output_dir)
    _LOG.info("frozen preprocessing constants fingerprint %s (recorded at %s)", fingerprint, record)
    _LOG.info("seed %d, tracker %s", cfg.seed, cfg.report_to or tracker_backend())

    trainer = build_trainer(cfg)
    resume = _resolve_resume(cfg, output_dir)
    if resume is not None:
        _LOG.info("resuming from %s (optimizer and scheduler state restored)", resume)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    adapter_dir = output_dir / ADAPTER_SUBDIR
    trainer.model.save_pretrained(str(adapter_dir))
    write_constants_fingerprint(adapter_dir, run_name=cfg.run_name, stage="lora")
    _LOG.info("adapter saved to %s", adapter_dir)
    return adapter_dir
