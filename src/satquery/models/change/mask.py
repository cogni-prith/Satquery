"""Pixel-level binary change mask head, and the tool that serves it.

**OPTIONAL.** The problem statement lists a pixel-accurate change mask as a nice-to-have,
not a mandatory row on the scoring table -- the mandatory bi-temporal requirement is
satisfied by `change.vqa_head` (CDVQA accuracy) and `vlm.change_description`. Per
CLAUDE.md this module is therefore **the first thing to cut if the team falls behind**.
Nothing else in the repo may take a hard dependency on it.

Single responsibility: given two co-registered dates, emit a single-channel binary
mask raster plus a display overlay. It answers no question and returns no class label.

Candidate architectures, all trainable on LEVIR-CD and SECOND:

- **BiT** -- Chen, Qi and Shi, "Remote Sensing Image Change Detection with Transformers",
  IEEE TGRS 2021 (arXiv 2103.00208), github.com/justchenhao/BIT_CD. ResNet backbone with a
  bitemporal image transformer over compact semantic tokens. Strongest accuracy per FLOP
  of the three and the default choice.
- **ChangeFormer** -- Bandara and Patel, "A Transformer-Based Siamese Network for Change
  Detection", IGARSS 2022 (arXiv 2201.01293), github.com/wgcban/ChangeFormer. Hierarchical
  Siamese transformer encoder plus MLP decoder; no separate CNN backbone.
- **TinyCD** -- Codegoni, Lombardi and Ferrari, "TINYCD: A (Not So) Deep Learning Model
  For Change Detection", Neural Computing and Applications 2023 (arXiv 2207.13159),
  github.com/AndreaCodegoni/Tiny_model_4_CD. Roughly 0.3 M parameters; the one to pick if
  the mask head has to share a GPU with the VLM at inference time.

Training data is LEVIR-CD (building change, 0.5 m aerial) and SECOND (multi-class
semantic change, 0.5-3 m). Both sit an order of magnitude finer than the 10 m Sentinel
adaptation set and much closer to the hidden Cartosat-2S evaluation set, so this head is
the one component whose training resolution already matches test resolution.

`torch` and `timm` are optional `gpu` extras: this module must import on a CPU-only
laptop with neither installed, so they are imported inside function bodies only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from satquery.models.base import BaseTool
from satquery.models.registry import REGISTRY
from satquery.serve.contracts import ToolRequest, ToolResult, ToolSpec
from satquery.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np
    import torch

__all__ = ["MASK_ARCHITECTURES", "ChangeMaskConfig", "ChangeMaskHead", "ChangeMaskTool"]

_LOG = get_logger(__name__)

#: Architectures accepted by `ChangeMaskConfig.architecture`, cited in the module docstring.
MASK_ARCHITECTURES: tuple[str, ...] = ("bit", "changeformer", "tinycd")


@dataclass(frozen=True, slots=True)
class ChangeMaskConfig:
    """Hyperparameters of the pixel-level change mask head."""

    architecture: str = "bit"
    """One of `MASK_ARCHITECTURES`. See the module docstring for the papers."""

    backbone: str = "resnet18"
    """`timm` encoder for the architectures that take one (`bit`, `tinycd`)."""

    pretrained: bool = True
    """Initialise the backbone from ImageNet weights before change fine-tuning."""

    image_size: int = 256
    """Training crop side in pixels. LEVIR-CD ships 1024x1024 tiles, cropped to this."""

    in_channels: int = 3
    """Channels per date entering the encoder. Three for rendered RGB or pseudo-RGB SAR."""

    embed_dim: int = 64
    """Width of the shared feature map the change decoder operates on."""

    threshold: float = 0.5
    """Probability above which a pixel is labelled changed. Overridable per request."""

    train_datasets: tuple[str, ...] = ("levir_cd", "second")
    """Dataset directory names under the data root, resolved via `utils.paths.dataset_dir`."""

    def __post_init__(self) -> None:
        if self.architecture not in MASK_ARCHITECTURES:
            raise ValueError(
                f"architecture={self.architecture!r} is not supported; choose one of "
                f"{list(MASK_ARCHITECTURES)}"
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold={self.threshold} must lie in [0.0, 1.0]")
        for name in ("image_size", "in_channels", "embed_dim"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name}={value} must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain mapping, for checkpoint metadata."""
        return asdict(self)

    @classmethod
    def from_config(cls, config: DictConfig | dict[str, Any]) -> ChangeMaskConfig:
        """Build a config from an already-loaded OmegaConf node or plain mapping.

        Raises:
            ValueError: The mapping carries a key that is not a field of this class.
        """
        if isinstance(config, DictConfig):
            resolved = OmegaConf.to_container(config, resolve=True)
        else:
            resolved = dict(config)
        if not isinstance(resolved, dict):
            raise ValueError(f"expected a mapping of change-mask fields, got {type(resolved)!r}")

        known = {field.name for field in fields(cls)}
        payload = {str(key): value for key, value in resolved.items()}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"unknown change-mask config keys {unknown}; known keys are {sorted(known)}"
            )
        if "train_datasets" in payload:
            payload["train_datasets"] = tuple(payload["train_datasets"])
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path, *, key: str | None = None) -> ChangeMaskConfig:
        """Load a config from a YAML file under `configs/`.

        Raises:
            FileNotFoundError: `path` does not exist.
            ValueError: `key` is absent from the file, or a field is unknown.
        """
        resolved_path = Path(path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"change-mask config not found: {resolved_path}")
        node = OmegaConf.load(resolved_path)
        if key is not None:
            if key not in node:
                raise ValueError(f"{resolved_path} has no top-level key {key!r}")
            node = node[key]
        if not isinstance(node, DictConfig):
            raise ValueError(f"{resolved_path} did not parse to a mapping")
        return cls.from_config(node)


class ChangeMaskHead:
    """Bi-temporal segmentation network producing a binary change probability map.

    Every method is an honest stub. Training is `transformers.Trainer` over the
    LEVIR-CD and SECOND loaders; nothing here writes a loop.
    """

    def __init__(self, config: ChangeMaskConfig) -> None:
        self.config = config
        self._module: Any | None = None

    @property
    def is_built(self) -> bool:
        """Whether `build()` has produced a live module."""
        return self._module is not None

    def build(self) -> Any:
        """Instantiate the configured change-detection architecture.

        Raises:
            NotImplementedError: always, until the architecture is vendored or written.
        """
        raise NotImplementedError(
            f"ChangeMaskHead.build is not implemented for architecture "
            f"'{self.config.architecture}'. Missing: the network definition itself (BiT from "
            "github.com/justchenhao/BIT_CD, ChangeFormer from github.com/wgcban/ChangeFormer, "
            "or TinyCD from github.com/AndreaCodegoni/Tiny_model_4_CD) and the optional 'gpu' "
            f"extras torch and timm (backbone '{self.config.backbone}'), none of which are "
            "present. This head is optional per the problem statement and is the first thing "
            "to cut."
        )

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> torch.Tensor:
        """Return per-pixel change logits for a batch of bi-temporal pairs.

        Args:
            x_t1: Earlier date, shape `(B, in_channels, image_size, image_size)`.
            x_t2: Later date, same shape, co-registered with `x_t1`.

        Returns:
            Logits of shape `(B, 1, image_size, image_size)`.

        Raises:
            NotImplementedError: always, until `build()` exists.
        """
        raise NotImplementedError(
            "ChangeMaskHead.forward is not implemented. Missing: ChangeMaskHead.build, which "
            "must run before any forward pass, and the optional 'gpu' extra torch."
        )

    def predict_mask(
        self,
        x_t1: torch.Tensor,
        x_t2: torch.Tensor,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Return a boolean change mask at the input resolution.

        Args:
            x_t1: Earlier date.
            x_t2: Later date, co-registered.
            threshold: Probability cut. Defaults to `ChangeMaskConfig.threshold`.

        Raises:
            NotImplementedError: always, until a trained checkpoint exists.
        """
        raise NotImplementedError(
            "ChangeMaskHead.predict_mask is not implemented. Missing: ChangeMaskHead.forward "
            "and a trained checkpoint at "
            f"<artifact_root>/checkpoints/change_mask/{self.config.architecture}/model.pt, "
            f"produced by training on {list(self.config.train_datasets)}."
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        """Write module weights plus config and the constants fingerprint to `path`.

        Raises:
            NotImplementedError: always, until `build()` exists.
        """
        raise NotImplementedError(
            f"ChangeMaskHead.save_checkpoint is not implemented (target {Path(path)}). "
            "Missing: a built module to take a state_dict from, and the optional 'gpu' extra "
            "torch for torch.save."
        )

    def load_checkpoint(self, path: str | Path) -> ChangeMaskHead:
        """Restore weights written by `save_checkpoint`.

        Must assert the checkpoint's recorded constants fingerprint matches the running
        process before loading any tensor.

        Raises:
            NotImplementedError: always, until `save_checkpoint` exists.
        """
        raise NotImplementedError(
            f"ChangeMaskHead.load_checkpoint is not implemented (source {Path(path)}). "
            "Missing: the checkpoint format written by ChangeMaskHead.save_checkpoint, the "
            "trained weights themselves, and the optional 'gpu' extra torch."
        )


class ChangeMaskTool(BaseTool):
    """Serves `change.mask`: a pixel-level binary change raster plus a display overlay.

    Intended flow once the head is trained: crop or resample both dates to
    `ChangeMaskConfig.image_size`, reorder optical bands via `preprocess.optical`, run
    `ChangeMaskHead.predict_mask`, write the single-channel mask and an RGB overlay under
    `utils.paths.artifact_dir`, and return their paths on `Evidence`. `confidence` is the
    mean predicted probability over the pixels labelled changed.

    Optional per the problem statement. The mandatory bi-temporal rows are covered by
    `change.vqa_head` and `vlm.change_description`, so cut this first if schedule slips.
    """

    def __init__(
        self, config: ChangeMaskConfig | None = None, spec: ToolSpec | None = None
    ) -> None:
        super().__init__(spec or REGISTRY.get_spec("change.mask"))
        self.config = config or ChangeMaskConfig()
        self.head = ChangeMaskHead(self.config)

    def _run(self, request: ToolRequest) -> ToolResult:
        """Produce the change mask and overlay for a bi-temporal pair.

        Raises:
            NotImplementedError: always. `BaseTool.run` converts this into a
                `ToolResult` with `error` set, so the trace stays well formed.
        """
        _LOG.debug("change.mask invoked for request %s", request.request_id)
        raise NotImplementedError(
            "change.mask has no trained mask checkpoint. Missing: weights at "
            f"<artifact_root>/checkpoints/change_mask/{self.config.architecture}/model.pt "
            f"from training on {list(self.config.train_datasets)}, the implementations of "
            "ChangeMaskHead.build / .load_checkpoint / .predict_mask, and the mask and "
            "overlay raster writers in satquery.io."
        )
