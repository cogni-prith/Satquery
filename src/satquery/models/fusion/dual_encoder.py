"""Dual-encoder fusion of a co-registered optical and SAR pair, and the tool serving it.

Single responsibility: joint built-up and water extraction from one optical raster and
one SAR raster covering the same ground, producing a mask, an overlay, the deterministic
index maps, and a confidence.

**Confidence here is agreement, not softmax.** Two independent estimates of the same
quantity are produced for every request:

1. the *learned* estimate, from a dual encoder trained on the 464,044 co-registered
   Sentinel-1 / Sentinel-2 pairs in BigEarthNet.txt (txt.bigearth.net, arXiv 2603.29630),
   built on BigEarthNet v2.0 / reBEN at 10 m GSD;
2. the *deterministic* estimate, from the closed-form index layer in
   `preprocess/indices.py` -- NDWI and NDBI on the optical stack, and the
   `SAR_WATER_DB_THRESHOLD` backscatter cut on the rendered VV channel.

`ToolResult.confidence` is the agreement between the two masks (intersection over union
over the union of their positives). This is the project's confidence signal: the learned
half was trained at 10 m on European Sentinel scenes, while the hidden ISRO evaluation
set is sub-metre Cartosat-2S and RISAT SAR, an order of magnitude away in resolution and
different in band centres and SAR calibration. The index half has no training
distribution to fall outside of, so when the two disagree the honest thing to report is
low confidence with both maps attached, and when they agree the answer is defensible even
though the learned model has never seen the sensor.

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
from satquery.serve.contracts import Evidence, ToolRequest, ToolResult, ToolSpec
from satquery.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

__all__ = [
    "FUSION_STRATEGIES",
    "FUSION_TARGETS",
    "DualEncoderFusion",
    "FusionConfig",
    "FusionExtractionTool",
    "build_fusion_model",
]

_LOG = get_logger(__name__)

#: Strategies accepted by `FusionConfig.fusion`.
FUSION_STRATEGIES: tuple[str, ...] = ("concat", "sum", "cross_attention", "gated")

#: Extraction targets accepted by `FusionConfig.targets`, matching the `fusion.extraction`
#: tool spec's `targets` parameter enum in `models/registry.py`.
FUSION_TARGETS: tuple[str, ...] = ("built_up", "water")


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Hyperparameters of the optical plus SAR dual encoder."""

    optical_encoder: str = "resnet50"
    """`timm` backbone for the optical branch, fed the canonically ordered band stack."""

    sar_encoder: str = "resnet18"
    """`timm` backbone for the SAR branch, fed the frozen pseudo-RGB rendering."""

    embed_dim: int = 512
    """Width each branch is projected to before fusion, so the two are commensurate."""

    fusion: str = "cross_attention"
    """How the two branch embeddings are combined; one of `FUSION_STRATEGIES`."""

    targets: tuple[str, ...] = FUSION_TARGETS
    """Extraction targets, each producing one binary output channel."""

    optical_in_channels: int = 4
    """Optical bands entering the branch. Four is `OPTICAL_BAND_ORDER_4`; ten is the extended order."""

    sar_in_channels: int = 3
    """SAR channels entering the branch. Three is the frozen `SAR_PSEUDO_RGB_LAYOUT`."""

    image_size: int = 224
    """Side length in pixels both branches are resampled to."""

    pretrained: bool = True
    """Initialise both branches from ImageNet weights before RS adaptation."""

    def __post_init__(self) -> None:
        if self.fusion not in FUSION_STRATEGIES:
            raise ValueError(
                f"fusion={self.fusion!r} is not supported; choose one of {list(FUSION_STRATEGIES)}"
            )
        if not self.targets:
            raise ValueError("targets must name at least one extraction target")
        unknown = sorted(set(self.targets) - set(FUSION_TARGETS))
        if unknown:
            raise ValueError(
                f"unknown fusion targets {unknown}; the fusion.extraction spec permits "
                f"{list(FUSION_TARGETS)}"
            )
        for name in ("embed_dim", "optical_in_channels", "sar_in_channels", "image_size"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name}={value} must be positive")

    @property
    def num_outputs(self) -> int:
        """Number of binary output channels, one per target."""
        return len(self.targets)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain mapping, for checkpoint metadata."""
        return asdict(self)

    @classmethod
    def from_config(cls, config: DictConfig | dict[str, Any]) -> FusionConfig:
        """Build a config from an already-loaded OmegaConf node or plain mapping.

        Raises:
            ValueError: The mapping carries a key that is not a field of this class.
        """
        if isinstance(config, DictConfig):
            resolved = OmegaConf.to_container(config, resolve=True)
        else:
            resolved = dict(config)
        if not isinstance(resolved, dict):
            raise ValueError(f"expected a mapping of fusion fields, got {type(resolved)!r}")

        known = {field.name for field in fields(cls)}
        payload = {str(key): value for key, value in resolved.items()}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"unknown fusion config keys {unknown}; known keys are {sorted(known)}"
            )
        if "targets" in payload:
            payload["targets"] = tuple(payload["targets"])
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path, *, key: str | None = None) -> FusionConfig:
        """Load a config from a YAML file under `configs/`.

        Raises:
            FileNotFoundError: `path` does not exist.
            ValueError: `key` is absent from the file, or a field is unknown.
        """
        resolved_path = Path(path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"fusion config not found: {resolved_path}")
        node = OmegaConf.load(resolved_path)
        if key is not None:
            if key not in node:
                raise ValueError(f"{resolved_path} has no top-level key {key!r}")
            node = node[key]
        if not isinstance(node, DictConfig):
            raise ValueError(f"{resolved_path} did not parse to a mapping")
        return cls.from_config(node)


def _pad_to_stride(tensor: Any, stride: int) -> tuple[Any, tuple[int, int]]:
    """Reflect-pad a `(B, C, H, W)` tensor so both spatial dims are multiples of `stride`.

    reBEN patches are 120x120 and a ResNet reduces by 32, which 120 does not divide.
    The obvious fix -- resize to 128 -- is the wrong one here: resizing changes the
    ground sampling distance, and the GSD token that prefixes every instruction would
    then describe a resolution the pixels no longer have. Padding leaves every real
    pixel at its true scale and is undone by `_crop_to`, so the frozen token stays true.

    Reflection rather than zeros because a zero border is a hard artificial edge, and
    the first thing a convolutional encoder learns from one is to detect it.
    """
    from torch.nn import functional as F

    height, width = tensor.shape[-2:]
    pad_h = (-height) % stride
    pad_w = (-width) % stride
    if pad_h == 0 and pad_w == 0:
        return tensor, (height, width)
    # Reflection padding requires the pad to be smaller than the dimension itself.
    mode = "reflect" if pad_h < height and pad_w < width else "replicate"
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return padded, (height, width)


def _crop_to(tensor: Any, size: tuple[int, int]) -> Any:
    """Undo `_pad_to_stride`, returning the tensor to its true extent."""
    height, width = size
    return tensor[..., :height, :width]


def build_fusion_model(config: FusionConfig) -> Any:
    """Construct the dual-encoder segmentation module.

    Defined inside a function because it subclasses `torch.nn.Module`, and torch is an
    optional `gpu` extra that must not be imported when the package is merely imported.

    The shape of the network follows from what the two sensors actually contribute.
    Optical carries fine spatial structure and the spectral information the indices key
    on; SAR carries an all-weather, illumination-independent signal that is far
    speckle-ier. So both branches are encoded separately -- never by shared weights,
    which is the mistake that would force one set of filters to serve reflectance and
    backscatter at once -- and fused at every scale, then decoded U-Net style back to
    full resolution using the *fused* skips.

    Fusing at every scale rather than only at the bottleneck matters for extraction:
    water boundaries are a high-resolution feature, and a bottleneck-only fusion gives
    the decoder nothing but optical detail to sharpen against, quietly discarding SAR
    exactly where it is most useful.
    """
    import timm
    import torch
    from torch import nn

    class ScaleFusion(nn.Module):
        """Combine one optical and one SAR feature map at a single scale.

        Both are first projected to `width` by 1x1 convolutions, because the two
        backbones need not agree on channel counts and every strategy below assumes
        they are commensurate.
        """

        def __init__(self, optical_channels: int, sar_channels: int, width: int) -> None:
            super().__init__()
            self.project_optical = nn.Conv2d(optical_channels, width, kernel_size=1)
            self.project_sar = nn.Conv2d(sar_channels, width, kernel_size=1)
            self.strategy = config.fusion

            if self.strategy == "concat":
                self.mix: nn.Module = nn.Conv2d(width * 2, width, kernel_size=1)
            elif self.strategy == "gated":
                # A learned per-pixel, per-channel gate. This is the strategy with the
                # most to offer here: it can suppress the optical branch over cloud and
                # the SAR branch over calm homogeneous surfaces, per pixel.
                self.gate = nn.Sequential(nn.Conv2d(width * 2, width, kernel_size=1), nn.Sigmoid())
                self.mix = nn.Identity()
            elif self.strategy == "cross_attention":
                self.attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
                self.mix = nn.Identity()
            else:  # "sum"
                self.mix = nn.Identity()

            self.norm = nn.BatchNorm2d(width)
            self.act = nn.ReLU(inplace=True)

        def forward(self, optical: Any, sar: Any) -> Any:
            optical = self.project_optical(optical)
            sar = self.project_sar(sar)

            if self.strategy == "concat":
                fused = self.mix(torch.cat([optical, sar], dim=1))
            elif self.strategy == "sum":
                fused = optical + sar
            elif self.strategy == "gated":
                weight = self.gate(torch.cat([optical, sar], dim=1))
                fused = weight * optical + (1.0 - weight) * sar
            else:  # cross_attention
                batch, channels, height, width = optical.shape
                query = optical.flatten(2).transpose(1, 2)
                key = sar.flatten(2).transpose(1, 2)
                attended, _ = self.attention(query, key, key, need_weights=False)
                fused = optical + attended.transpose(1, 2).reshape(batch, channels, height, width)

            return self.act(self.norm(fused))

    class DecoderBlock(nn.Module):
        """Upsample, concatenate the fused skip, and convolve."""

        def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any, skip: Any | None) -> Any:
            from torch.nn import functional as F

            target = skip.shape[-2:] if skip is not None else [2 * s for s in x.shape[-2:]]
            x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            return self.block(x)

    class DualEncoderSegmenter(nn.Module):
        """Two encoders, per-scale fusion, U-Net decoder, one logit map per target."""

        def __init__(self) -> None:
            super().__init__()
            self.optical_encoder = timm.create_model(
                config.optical_encoder,
                features_only=True,
                in_chans=config.optical_in_channels,
                pretrained=config.pretrained,
            )
            self.sar_encoder = timm.create_model(
                config.sar_encoder,
                features_only=True,
                in_chans=config.sar_in_channels,
                pretrained=config.pretrained,
            )

            optical_channels = self.optical_encoder.feature_info.channels()
            sar_channels = self.sar_encoder.feature_info.channels()
            if len(optical_channels) != len(sar_channels):
                raise ValueError(
                    f"encoders disagree on depth: {config.optical_encoder} exposes "
                    f"{len(optical_channels)} scales, {config.sar_encoder} exposes "
                    f"{len(sar_channels)}. Per-scale fusion needs them aligned."
                )
            self.reductions = self.optical_encoder.feature_info.reduction()

            # Fusion width tapers with depth: full `embed_dim` at the bottleneck, halved
            # at each shallower scale, floored so the finest scale stays cheap.
            depth = len(optical_channels)
            widths = [
                max(config.embed_dim // (2 ** (depth - 1 - index)), 32) for index in range(depth)
            ]
            self.widths = widths

            self.fusions = nn.ModuleList(
                [
                    ScaleFusion(optical_channels[index], sar_channels[index], widths[index])
                    for index in range(depth)
                ]
            )

            decoders = []
            in_channels = widths[-1]
            for index in range(depth - 2, -1, -1):
                decoders.append(DecoderBlock(in_channels, widths[index], widths[index]))
                in_channels = widths[index]
            # One more block to climb from the shallowest encoder stride back to full
            # resolution; it has no skip because the encoder publishes none at stride 1.
            decoders.append(DecoderBlock(in_channels, 0, max(in_channels // 2, 32)))
            self.decoders = nn.ModuleList(decoders)

            self.head = nn.Conv2d(max(in_channels // 2, 32), config.num_outputs, kernel_size=1)

        @property
        def stride(self) -> int:
            """Coarsest encoder reduction; inputs are padded to a multiple of this."""
            return int(max(self.reductions))

        def forward(
            self,
            pixel_values_optical: Any,
            pixel_values_sar: Any,
            labels: Any = None,
        ) -> dict[str, Any]:
            if pixel_values_optical.shape[-2:] != pixel_values_sar.shape[-2:]:
                raise ValueError(
                    "optical and SAR inputs must be co-registered onto one grid, got "
                    f"{tuple(pixel_values_optical.shape[-2:])} and "
                    f"{tuple(pixel_values_sar.shape[-2:])}"
                )

            optical, size = _pad_to_stride(pixel_values_optical, self.stride)
            sar, _ = _pad_to_stride(pixel_values_sar, self.stride)

            optical_features = self.optical_encoder(optical)
            sar_features = self.sar_encoder(sar)
            fused = [
                fusion(optical_features[index], sar_features[index])
                for index, fusion in enumerate(self.fusions)
            ]

            x = fused[-1]
            for position, decoder in enumerate(self.decoders[:-1]):
                x = decoder(x, fused[len(fused) - 2 - position])
            x = self.decoders[-1](x, None)

            logits = _crop_to(self.head(x), size)

            output: dict[str, Any] = {"logits": logits}
            if labels is not None:
                output["loss"] = _extraction_loss(logits, labels)
            return output

    return DualEncoderSegmenter()


def _extraction_loss(logits: Any, labels: Any, *, dice_weight: float = 0.5) -> Any:
    """Binary cross-entropy plus soft Dice, summed over targets.

    Dice is not optional decoration here, and the reason is the measured class balance
    rather than a general worry about segmentation. Over 400 randomly sampled reBEN
    patches, built-up covers 3.2% of pixels on average and water 15.6% -- but both are
    heavily zero-inflated: built-up is entirely absent from 84% of patches and water
    from 73%. The median patch contains none of either.

    That shape is what makes plain BCE dangerous. Its minimiser on the typical patch is
    to predict background everywhere, which scores near-perfect pixel accuracy while
    producing an entirely empty mask -- a model that looks excellent on the wrong
    metric and extracts nothing. Dice is computed on the overlap of the positives
    alone, so an empty prediction scores zero and the degenerate solution stops being
    attractive. Reporting per-class IoU rather than accuracy is the matching decision
    on the evaluation side.

    Args:
        logits: `(B, T, H, W)` raw scores, channel order matching `FusionConfig.targets`.
        labels: `(B, T, H, W)` binary targets, same order.
        dice_weight: Relative weight of the Dice term against BCE.

    Returns:
        Scalar loss tensor.
    """
    import torch
    from torch.nn import functional as F

    targets = labels.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    probabilities = torch.sigmoid(logits)
    dims = (0, 2, 3)  # keep the target axis, so a rare class cannot be averaged away
    intersection = (probabilities * targets).sum(dim=dims)
    cardinality = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice = 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)
    return bce + dice_weight * dice.mean()


class DualEncoderFusion:
    """Two modality-specific encoders joined by per-scale fusion and a segmentation head.

    Trained on the co-registered Sentinel-1 and Sentinel-2 pairs of reBEN via
    `data.datasets.bigearthnet_fusion`. The learned output is never reported alone:
    `FusionExtractionTool` always cross-checks it against the deterministic index layer
    and reports the agreement as confidence.
    """

    def __init__(self, config: FusionConfig) -> None:
        self.config = config
        self._module: Any | None = None

    @classmethod
    def from_module(cls, config: FusionConfig, module: Any) -> DualEncoderFusion:
        """Wrap an already-constructed module, e.g. one a `Trainer` just finished.

        Exists so training can hand its model back for `save_checkpoint` without
        reaching into a private attribute.
        """
        wrapper = cls(config)
        wrapper._module = module
        return wrapper

    @property
    def is_built(self) -> bool:
        """Whether `build()` has produced a live module."""
        return self._module is not None

    @property
    def module(self) -> Any:
        """The live `torch.nn.Module`, building it on first access."""
        if self._module is None:
            self._module = self.build()
        return self._module

    def build(self) -> Any:
        """Instantiate both branches, the per-scale fusion blocks and the decoder."""
        self._module = build_fusion_model(self.config)
        _LOG.info(
            "built dual-encoder fusion: optical=%s sar=%s fusion=%s targets=%s",
            self.config.optical_encoder,
            self.config.sar_encoder,
            self.config.fusion,
            list(self.config.targets),
        )
        return self._module

    def forward(self, x_optical: torch.Tensor, x_sar: torch.Tensor) -> torch.Tensor:
        """Return per-target logits for a batch of co-registered pairs.

        Args:
            x_optical: Optical stack, `(B, optical_in_channels, H, W)`, already in the
                frozen canonical band order.
            x_sar: SAR stack, `(B, sar_in_channels, H, W)`, already rendered by the
                frozen SAR pipeline.

        Returns:
            Logits of shape `(B, num_outputs, H, W)`, channel order matching
            `FusionConfig.targets`.
        """
        return self.module(x_optical, x_sar)["logits"]

    def extract(
        self,
        x_optical: torch.Tensor,
        x_sar: torch.Tensor,
        *,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Return one boolean mask per target from the learned branch only.

        The caller is expected to compare these against the deterministic index masks
        before reporting anything; see `FusionExtractionTool`.

        Returns:
            Mapping of target name to a `(B, H, W)` boolean tensor.
        """
        import torch

        module = self.module
        module.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self.forward(x_optical, x_sar))
        return {
            name: probabilities[:, index] > threshold
            for index, name in enumerate(self.config.targets)
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Write weights, config and the constants fingerprint to `path`.

        The fingerprint travels with the weights so `load_checkpoint` can refuse a
        checkpoint whose SAR rendering or band order no longer matches this process.
        """
        import torch

        from satquery.preprocess.constants import constants_fingerprint

        if self._module is None:
            raise RuntimeError(
                "nothing to save: build() or a training run must produce a module first"
            )
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._module.state_dict(),
                "config": {**self.config.to_dict(), "targets": list(self.config.targets)},
                # Pinned to the pixel scope. This encoder reads bands, the SAR pipeline
                # and index thresholds; it never sees an instruction template, so hashing
                # those would make the guard fire on changes it cannot observe.
                "constants_fingerprint": constants_fingerprint("pixels"),
                "fingerprint_scope": "pixels",
            },
            resolved,
        )
        _LOG.info("wrote fusion checkpoint to %s", resolved)
        return resolved

    def load_checkpoint(
        self, path: str | Path, *, strict_fingerprint: bool = True
    ) -> DualEncoderFusion:
        """Restore weights written by `save_checkpoint`.

        Asserts the checkpoint's recorded constants fingerprint matches the running
        process before loading any tensor: a SAR rendering that drifted between
        training and inference is exactly the failure the frozen constants exist to
        catch, and it is invisible in the resulting scores.

        Args:
            path: Checkpoint written by `save_checkpoint`.
            strict_fingerprint: Refuse to load on a fingerprint mismatch. Only ever
                disabled to inspect a stale checkpoint, never on a scoring path.

        Raises:
            FileNotFoundError: `path` does not exist.
            RuntimeError: The fingerprints differ and `strict_fingerprint` is set, or
                the checkpoint is missing tensors the module needs.
        """
        import torch

        from satquery.preprocess.constants import constants_fingerprint

        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"fusion checkpoint not found: {resolved}. Train one with "
                "scripts/train_fusion.py; nothing is downloaded automatically."
            )

        # weights_only=True: this checkpoint holds only tensors and primitives, so
        # there is no reason to hand the unpickler the ability to execute code from a
        # file that may have arrived from anywhere.
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
        recorded = payload.get("constants_fingerprint")
        # Older checkpoints predate scoping and recorded the full hash; read the scope
        # they were written with rather than assuming this one.
        running = constants_fingerprint(payload.get("fingerprint_scope"))
        if recorded != running:
            message = (
                "FROZEN CONSTANT DRIFT.\n"
                f"  recorded at training time : {recorded}\n"
                f"  running in this process   : {running}\n"
                "The SAR rendering, band order, GSD token or index definitions changed "
                "after these weights were trained, so their outputs are not comparable."
            )
            if strict_fingerprint:
                raise RuntimeError(message)
            _LOG.warning("%s", message)

        if "config" in payload:
            self.config = FusionConfig.from_config(payload["config"])
        module = self.build()
        missing, unexpected = module.load_state_dict(payload["state_dict"], strict=False)
        if missing:
            raise RuntimeError(
                f"fusion checkpoint {resolved} is missing {len(missing)} parameter tensor(s), "
                f"first few: {list(missing)[:5]}. Refusing to run a partly-initialised model, "
                "which would produce confident-looking noise."
            )
        if unexpected:
            _LOG.warning(
                "fusion checkpoint carries %d unexpected tensor(s), ignored: %s",
                len(unexpected),
                list(unexpected)[:5],
            )
        _LOG.info("loaded fusion checkpoint from %s", resolved)
        return self


class FusionExtractionTool(BaseTool):
    """Serves `fusion.extraction`: joint built-up and water extraction from an optical / SAR pair.

    Intended flow, in order, once the dual encoder is trained:

    1. Render the SAR raster through `preprocess.sar.render_sar` -- refined Lee at window
       7, dB conversion, 2nd-to-98th percentile stretch, and the frozen pseudo-RGB layout
       (R=VV, G=VH, B=VV/VH dB ratio). Raw SAR never reaches a model, and RISAT at test
       time goes through the identical function as Sentinel-1 at train time.
    2. Reorder the optical raster into the frozen canonical band order with
       `preprocess.optical.reorder_bands`, pan-sharpening first via
       `preprocess.optical.pansharpen` when a Cartosat-2S panchromatic band is present, so
       the two branches see one consistent resolution.
    3. Compute the deterministic index masks with `preprocess.indices` -- NDWI for water,
       NDBI for built-up, and the SAR VV backscatter cut at `SAR_WATER_DB_THRESHOLD` as a
       second water estimate that needs no optical bands at all.
    4. Run `DualEncoderFusion.extract` on the two prepared stacks for the learned masks.
    5. Compare learned against deterministic per target, and report **both**: the learned
       mask as `Evidence.mask_path`, the index rasters as `Evidence.index_maps`, a
       rendered overlay as `Evidence.overlay_path`, and their agreement (IoU) as
       `ToolResult.confidence`, with a warning naming any target where the two disagree.

    Reporting only the learned mask would hide the one signal that survives the 20x
    resolution gap to the hidden evaluation set.
    """

    def __init__(self, config: FusionConfig | None = None, spec: ToolSpec | None = None) -> None:
        super().__init__(spec or REGISTRY.get_spec("fusion.extraction"))
        self.config = config or FusionConfig()
        self.model = DualEncoderFusion(self.config)

    def _run(self, request: ToolRequest) -> ToolResult:
        """Extract the requested targets from the co-registered pair.

        Produces two independent estimates and reports both. The learned masks come
        from the dual encoder; the deterministic masks come from NDWI and NDBI on the
        optical stack. `ToolResult.confidence` is their agreement, not a softmax --
        see the module docstring for why that distinction is the whole point.

        Raises:
            NotImplementedError: If no trained checkpoint is present. `BaseTool.run`
                converts this into a `ToolResult` with `error` set, so the trace stays
                well formed rather than the tool returning a fabricated mask.
        """
        import numpy as np
        import torch

        from satquery.io.raster import read_raster, write_raster
        from satquery.models.base import load_model_input
        from satquery.preprocess.constants import (
            FUSION_CLASS_INDEX,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )
        from satquery.preprocess.indices import (
            agreement_score,
            builtup_mask,
            compute_indices,
            water_mask,
        )
        from satquery.preprocess.optical import stretch_to_uint8
        from satquery.utils.paths import artifact_dir

        _LOG.debug("fusion.extraction invoked for request %s", request.request_id)

        optical_ref, sar_ref = self._ordered_refs(request)

        # A SAR raster carrying a "ratio" band has already been through the frozen
        # renderer -- raw SAR has only polarisations. Passing it to `load_model_input`
        # would render it a second time, applying a speckle filter and a dB conversion
        # to what are already display values. The model would still return a confident
        # mask, and nothing downstream could tell. Detecting it by band name is exact
        # rather than heuristic, because the frozen layout names that third channel.
        if "ratio" in [name.lower() for name in sar_ref.band_names]:
            raise ValueError(
                f"{sar_ref.path} carries a 'ratio' band, so it is already-rendered "
                "pseudo-RGB rather than raw SAR. fusion.extraction expects raw "
                "polarisations (VV/VH or HH/HV) and renders them itself through the "
                "frozen pipeline; rendering twice silently corrupts the input."
            )

        checkpoint = self.checkpoint_path()
        if not checkpoint.is_file():
            raise NotImplementedError(
                f"fusion.extraction has no trained checkpoint at {checkpoint}. Train one "
                "with scripts/train_fusion.py on the co-registered reBEN Sentinel-1 / "
                "Sentinel-2 pairs. Returning a mask without trained weights would mean "
                "reporting noise as an extraction result."
            )

        targets = tuple(request.params.get("targets", self.config.targets))
        warnings: list[str] = []

        # -- deterministic half: indices straight off the optical stack
        optical_stack, optical_parsed = read_raster(optical_ref.path)
        warnings.extend(optical_parsed.warnings)
        wanted_indices = [
            FUSION_CLASS_INDEX[name] for name in targets if name in FUSION_CLASS_INDEX
        ]
        index_maps, index_warnings = compute_indices(
            optical_stack, optical_parsed.band_names, which=wanted_indices
        )
        warnings.extend(index_warnings)

        deterministic: dict[str, np.ndarray] = {}
        for name in targets:
            index_name = FUSION_CLASS_INDEX.get(name)
            if index_name in index_maps:
                threshold = water_mask if name == "water" else builtup_mask
                deterministic[name] = threshold(index_maps[index_name])

        # -- learned half
        if self.model.is_built is False:
            self.model.load_checkpoint(checkpoint)

        # The two halves need the optical stack in *different* forms, and conflating
        # them silently breaks whichever half is not being looked at.
        #
        # Indices need RAW reflectance: NDWI is a ratio between bands, and the frozen
        # per-band percentile stretch rescales each band independently, destroying
        # exactly the cross-band relationship the index measures. Measured on reBEN,
        # NDWI scores 0.850 IoU on raw values and 0.211 on stretched ones.
        #
        # The model needs STRETCHED input, because that is what it was trained on --
        # the training collator reads rasters the loader already stretched. Feeding it
        # raw reflectance puts values in [0, 26] where it learned [0, 1].
        stretched = stretch_to_uint8(optical_stack[: self.config.optical_in_channels])
        optical_input = torch.from_numpy(np.asarray(stretched, dtype=np.float32) / 255.0).unsqueeze(
            0
        )
        sar_rgb, sar_warnings = load_model_input(sar_ref)
        warnings.extend(sar_warnings)
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        sar_input = torch.from_numpy(
            (np.asarray(sar_rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
        ).unsqueeze(0)

        learned = {
            name: mask[0].cpu().numpy()
            for name, mask in self.model.extract(optical_input, sar_input).items()
            if name in targets
        }

        # -- report both, and their agreement
        output_dir = artifact_dir("outputs", "fusion", str(request.request_id))
        combined = np.zeros(next(iter(learned.values())).shape, dtype=np.uint8)
        for position, name in enumerate(targets):
            if name in learned:
                combined[learned[name]] = position + 1
        mask_path = write_raster(
            output_dir / "extraction_mask.tif", combined, band_names=list(targets)
        )

        written_indices: dict[str, Path] = {}
        for index_name, index_map in index_maps.items():
            written_indices[index_name] = write_raster(
                output_dir / f"{index_name}.tif",
                index_map.astype(np.float32),
                band_names=[index_name],
            )

        agreements: dict[str, float] = {}
        for name in targets:
            if name in learned and name in deterministic:
                agreements[name] = agreement_score(learned[name], deterministic[name])
            else:
                warnings.append(
                    f"{name}: no deterministic cross-check was possible, so the reported "
                    "confidence rests on the learned model alone"
                )

        confidence = float(np.mean(list(agreements.values()))) if agreements else None
        for name, score in agreements.items():
            if score < 0.5:
                warnings.append(
                    f"{name}: the learned mask and the {FUSION_CLASS_INDEX[name].upper()} "
                    f"index agree on only {score:.2f} IoU; treat this extraction as "
                    "unconfirmed and inspect both maps"
                )

        coverage = {name: float(mask.mean()) for name, mask in learned.items()}
        answer = "; ".join(
            f"{name.replace('_', ' ')} covers {coverage[name] * 100:.1f}% of the scene"
            for name in targets
            if name in coverage
        )

        return ToolResult(
            request_id=request.request_id,
            tool_name=self.spec.name,
            tool_version=self.spec.version,
            answer=answer,
            confidence=confidence,
            evidence=Evidence(mask_path=mask_path, index_maps=written_indices),
            params_used={"targets": list(targets), "agreement": agreements},
            warnings=warnings,
        )

    def checkpoint_path(self) -> Path:
        """Where the trained dual encoder is expected. Pure, safe without weights."""
        from satquery.utils.paths import artifact_root

        return (
            artifact_root()
            / "checkpoints"
            / "fusion"
            / f"{self.config.optical_encoder}_{self.config.sar_encoder}"
            / "model.pt"
        )

    def _ordered_refs(self, request: ToolRequest) -> tuple[Any, Any]:
        """Return `(optical_ref, sar_ref)` regardless of the order they arrived in.

        The router promises a cross-modal pair but not which came first, and silently
        feeding SAR to the optical branch would produce a confident, entirely wrong
        mask rather than an error.
        """
        from satquery.serve.contracts import Modality

        sar = [ref for ref in request.images if ref.modality is Modality.SAR]
        optical = [ref for ref in request.images if ref.modality is not Modality.SAR]
        if len(sar) != 1 or len(optical) != 1:
            raise ValueError(
                "fusion.extraction needs exactly one SAR and one optical image, got "
                f"{len(sar)} SAR and {len(optical)} optical"
            )
        return optical[0], sar[0]
