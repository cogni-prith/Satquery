"""Wrapper around the EarthDial-4B vision-language backbone.

EarthDial is an InternVL-derived remote sensing VLM that already covers RGB, SAR,
multispectral and bi-temporal input, which is why the architecture picks it as the primary
backbone: we LoRA on top of it rather than adapting a generic VLM from scratch.

Two checkpoints, two registry entries, so modality routing is real rather than
cosmetic:

- optical / RGB -> ``akshaydudhane/EarthDial_4B_RGB``
- SAR, multispectral, bi-temporal -> ``akshaydudhane/EarthDial_4B_MS``

Both repo ids are VERIFIED against the Hub. EarthDial is a fine-tune of
``OpenGVLab/InternVL2-4B``: InternViT-300M plus Phi-3-mini, 448 px tiles, dynamic
tiling up to six patches plus a thumbnail.

Missing remote code
-------------------
The published checkpoints declare an ``auto_map`` pointing at
``modeling_internvl_chat.py``, ``modeling_phi3.py`` and their configuration modules,
but do not actually ship those files. ``trust_remote_code=True`` therefore fails with
a bare "file not found" until the modules are copied in from the upstream InternVL2-4B
repo. :func:`ensure_remote_code` does that, and :meth:`load` calls it. This is a defect
in the published repo, not in our usage.

Fallback order when a checkpoint misbehaves, by project rule: GeoChat, then Qwen2.5-VL-7B
fine-tuned from our own mix. Do not restructure around a fallback until the primary has
failed an actual inference smoke test.

GSD token contract
------------------
Every instruction string arriving at this backbone MUST already carry its GSD token,
built with :func:`satquery.preprocess.gsd.prefix_instruction` from the affine
transform of the input raster. The backbone never adds, rewrites or repairs the
token: doing so here would create a second place where the frozen format lives and
would let the training path and the inference path drift apart. An instruction with
no token is a caller bug, not something for this module to paper over.

Image preprocessing here is deliberately torch- and torchvision-free: tiling and
normalisation run on NumPy plus Pillow, so :func:`dynamic_tiles` is unit-testable on a
CPU-only machine and so the pipeline does not inherit torchvision's ABI coupling to a
specific torch build.

Nothing in this module imports torch, transformers or peft at module scope; those
are optional ``gpu`` extras and this package must import on a CPU-only laptop.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from satquery.preprocess.constants import IMAGENET_MEAN, IMAGENET_STD
from satquery.utils.logging import get_logger
from satquery.utils.paths import artifact_root

__all__ = [
    "REMOTE_CODE_FILES",
    "REMOTE_CODE_SOURCE",
    "BackboneConfig",
    "EarthDialBackbone",
    "dynamic_tiles",
    "ensure_remote_code",
]

_LOG = get_logger(__name__)

#: Ordered fallback backbones, consulted only after the primary fails a real smoke test.
FALLBACK_REPO_IDS: tuple[str, ...] = ("MBZUAI/GeoChat-7B", "Qwen/Qwen2.5-VL-7B-Instruct")

#: Upstream repo the EarthDial checkpoints are fine-tuned from, and the source of the
#: custom modelling code their own ``auto_map`` references but does not ship.
REMOTE_CODE_SOURCE = "OpenGVLab/InternVL2-4B"

#: Source-level fixes applied to the vendored remote code after it is copied in.
#:
#: The InternVL modules were written against transformers 4.37 (mid-2024). Modern
#: transformers builds a model under a meta-device context so that weights can be
#: streamed in shard by shard, and code that materialises a real value during
#: ``__init__`` now fails. Each entry is ``(filename, old, new, reason)``, applied
#: literally and idempotently, and every one must be numerically equivalent -- these
#: are compatibility shims, never behaviour changes.
REMOTE_CODE_PATCHES: tuple[tuple[str, str, str, str], ...] = (
    (
        "modeling_internvl_chat.py",
        "from transformers.modeling_utils import PreTrainedModel",
        "from transformers.generation import GenerationMixin\n"
        "from transformers.modeling_utils import PreTrainedModel",
        "transformers 5 requires a generative model to inherit GenerationMixin explicitly; "
        "4.37 mixed it into PreTrainedModel",
    ),
    (
        "modeling_internvl_chat.py",
        "class InternVLChatModel(PreTrainedModel):\n    config_class = InternVLChatConfig",
        "class InternVLChatModel(PreTrainedModel, GenerationMixin):\n"
        "    # transformers 5 reads this when resolving tied weights; 4.37 had no such\n"
        "    # attribute. Empty is correct here: this model ties nothing across towers.\n"
        "    all_tied_weights_keys: dict = {}\n"
        "    config_class = InternVLChatConfig",
        "transformers 5 expects `all_tied_weights_keys` and explicit GenerationMixin "
        "inheritance on every generative model",
    ),
    (
        "modeling_phi3.py",
        "from transformers.modeling_utils import PreTrainedModel",
        "from transformers.generation import GenerationMixin\n"
        "from transformers.modeling_utils import PreTrainedModel",
        "the inner language model needs GenerationMixin too, for the same reason as the "
        "outer chat model",
    ),
    (
        "modeling_phi3.py",
        "class Phi3ForCausalLM(Phi3PreTrainedModel):",
        "class Phi3ForCausalLM(Phi3PreTrainedModel, GenerationMixin):",
        "InternVLChatModel.generate delegates to language_model.generate, which does not "
        "exist unless the Phi-3 head inherits GenerationMixin under transformers 5",
    ),
    (
        "modeling_phi3.py",
        "past_key_value.get_usable_length(kv_seq_len, self.layer_idx)",
        "past_key_value.get_seq_length(self.layer_idx)",
        "Cache.get_usable_length was removed in transformers 5. For a DynamicCache it "
        "returned get_seq_length(layer_idx) exactly, since get_max_length() was None",
    ),
    (
        "modeling_phi3.py",
        "past_key_values.get_usable_length(seq_length)",
        "past_key_values.get_seq_length()",
        "same removal, at the model level where no layer index is passed",
    ),
    (
        "modeling_phi3.py",
        "past_length = past_key_values.seen_tokens",
        "past_length = past_key_values.get_seq_length()",
        "Cache.seen_tokens was deprecated then removed; get_seq_length() is its "
        "documented replacement",
    ),
    (
        "modeling_intern_vit.py",
        "dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]",
        "dpr = [\n"
        "            config.drop_path_rate * i / max(config.num_hidden_layers - 1, 1)\n"
        "            for i in range(config.num_hidden_layers)\n"
        "        ]",
        "torch.linspace(...).item() is illegal under the meta-device init that modern "
        "transformers uses; the closed form is identical for every n >= 1",
    ),
)

#: The modules that must be present next to the weights for ``trust_remote_code`` to work.
REMOTE_CODE_FILES: tuple[str, ...] = (
    "configuration_intern_vit.py",
    "configuration_internvl_chat.py",
    "configuration_phi3.py",
    "conversation.py",
    "modeling_intern_vit.py",
    "modeling_internvl_chat.py",
    "modeling_phi3.py",
)


@dataclass(slots=True)
class BackboneConfig:
    """Everything needed to instantiate one EarthDial checkpoint.

    Field names are the contract with ``configs/model/*.yaml`` and with the training
    config that reads those same files. Keep them stable.
    """

    hf_repo_id: str
    revision: str | None = None
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str = "sdpa"
    """`sdpa`, `eager`, or `flash_attention_2`. See the note where this is used: eager
    materialises the whole attention matrix and OOMs on a long visual sequence."""
    load_in_4bit: bool = False
    image_size: int = 448
    max_tiles: int = 6
    trust_remote_code: bool = True
    adapter_path: Path | None = None

    @classmethod
    def from_config(cls, cfg: DictConfig | dict[str, Any]) -> BackboneConfig:
        """Build a config from an OmegaConf node or plain mapping.

        Keys outside the dataclass -- the ``lora:`` and ``inference:`` blocks the same
        YAML file carries for the trainer -- are ignored rather than rejected, so one
        model YAML can serve both the backbone and the training entry script.
        """
        if isinstance(cfg, DictConfig | ListConfig):
            raw = OmegaConf.to_container(cfg, resolve=True)
        else:
            raw = dict(cfg)
        if not isinstance(raw, dict):
            raise TypeError(f"model config must resolve to a mapping, got {type(raw)!r}")

        known = {f.name for f in fields(cls)}
        kwargs = {str(key): value for key, value in raw.items() if str(key) in known}
        if "hf_repo_id" not in kwargs:
            raise KeyError(
                "model config is missing the required key 'hf_repo_id'; "
                f"got keys {sorted(str(k) for k in raw)}"
            )
        adapter = kwargs.get("adapter_path")
        # A relative adapter path is resolved against the artifact root, not the working
        # directory: checkpoints live under $SATQUERY_ARTIFACT_ROOT and a config should
        # not depend on where the process happens to be launched from.
        if adapter:
            candidate = Path(adapter)
            kwargs["adapter_path"] = (
                candidate if candidate.is_absolute() else artifact_root() / candidate
            )
        else:
            kwargs["adapter_path"] = None
        return cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, path: str | Path) -> BackboneConfig:
        """Load ``configs/model/<name>.yaml`` and build a config from its top level."""
        return cls.from_config(OmegaConf.load(Path(path)))

    @property
    def is_adapted(self) -> bool:
        """True when a LoRA adapter path is configured (it may still not exist on disk)."""
        return self.adapter_path is not None


def _tile_grid(
    width: int, height: int, tile_size: int, max_tiles: int, min_tiles: int = 1
) -> tuple[int, int]:
    """Choose the (columns, rows) tile grid whose aspect ratio best matches the image.

    InternVL's dynamic tiling: enumerate every grid whose tile count lies within
    ``[min_tiles, max_tiles]`` and pick the one closest in aspect ratio to the input.
    Ties break toward the larger grid, but only when the image is big enough to justify
    it -- otherwise a small image gets upsampled into many near-empty tiles.
    """
    aspect = width / height
    candidates = sorted(
        {
            (cols, rows)
            for n in range(min_tiles, max_tiles + 1)
            for cols in range(1, n + 1)
            for rows in range(1, n + 1)
            if min_tiles <= cols * rows <= max_tiles
        },
        key=lambda grid: grid[0] * grid[1],
    )

    best = (1, 1)
    best_diff = float("inf")
    area = width * height
    for cols, rows in candidates:
        difference = abs(aspect - cols / rows)
        if difference < best_diff:
            best_diff, best = difference, (cols, rows)
        elif difference == best_diff and area > 0.5 * tile_size * tile_size * cols * rows:
            best = (cols, rows)
    return best


def dynamic_tiles(
    image: np.ndarray,
    *,
    tile_size: int = 448,
    max_tiles: int = 6,
    use_thumbnail: bool = True,
) -> np.ndarray:
    """Split an RGB image into normalised InternVL tiles.

    Pure, deterministic and free of torch and torchvision, so it is testable on CPU and
    identical on the training and inference paths.

    Args:
        image: ``(height, width, 3)`` uint8 RGB. This is exactly the layout
            :func:`satquery.preprocess.sar.render_sar` produces, so a rendered SAR
            scene feeds straight in.
        tile_size: Tile edge in pixels; the checkpoint's ``force_image_size``.
        max_tiles: Dynamic-tiling budget, the checkpoint's ``max_dynamic_patch``.
        use_thumbnail: Append a whole-image thumbnail tile when more than one tile was
            produced, so the model still sees global context.

    Returns:
        ``(n_tiles, 3, tile_size, tile_size)`` float32, ImageNet-normalised.
    """
    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected a (height, width, 3) RGB image, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    height, width = array.shape[:2]
    cols, rows = _tile_grid(width, height, tile_size, max_tiles)

    pil = Image.fromarray(array)
    resized = pil.resize((cols * tile_size, rows * tile_size), Image.BICUBIC)

    tiles = [
        np.asarray(
            resized.crop(
                (
                    column * tile_size,
                    row * tile_size,
                    (column + 1) * tile_size,
                    (row + 1) * tile_size,
                )
            )
        )
        for row in range(rows)
        for column in range(cols)
    ]

    if use_thumbnail and len(tiles) > 1:
        tiles.append(np.asarray(pil.resize((tile_size, tile_size), Image.BICUBIC)))

    stack = np.stack(tiles).astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    stack = (stack - mean) / std
    return np.ascontiguousarray(stack.transpose(0, 3, 1, 2))


def ensure_remote_code(local_dir: str | Path, source: str = REMOTE_CODE_SOURCE) -> list[str]:
    """Copy the custom modelling modules the EarthDial repo references but does not ship.

    The published checkpoints declare an ``auto_map`` pointing at
    ``modeling_internvl_chat.py`` and friends, then omit them. Without this,
    ``trust_remote_code=True`` fails on a missing file. EarthDial is a fine-tune of
    ``OpenGVLab/InternVL2-4B`` with an identical architecture, so its modules load these
    weights unchanged.

    Args:
        local_dir: Snapshot directory holding the EarthDial weights.
        source: Upstream repo to take the modules from.

    Returns:
        The filenames that had to be copied in. Empty when nothing was missing.
    """
    from huggingface_hub import hf_hub_download

    target = Path(local_dir)
    copied: list[str] = []
    for filename in REMOTE_CODE_FILES:
        destination = target / filename
        if destination.exists():
            continue
        fetched = hf_hub_download(source, filename)
        destination.write_bytes(Path(fetched).read_bytes())
        copied.append(filename)

    if copied:
        _LOG.info(
            "copied %d missing remote-code module(s) from %s into %s: %s",
            len(copied),
            source,
            target,
            ", ".join(copied),
        )
    patch_remote_code(target)
    return copied


def patch_remote_code(local_dir: str | Path) -> list[str]:
    """Apply the transformers-compatibility fixes in :data:`REMOTE_CODE_PATCHES`.

    Idempotent: a patch whose replacement text is already present is skipped, so this
    is safe to call on every load. A patch whose *original* text is absent and whose
    replacement is also absent is reported loudly, because that means the upstream file
    changed shape and the shim needs revisiting rather than silently doing nothing.

    Returns:
        The filenames actually modified.
    """
    target = Path(local_dir)
    changed: list[str] = []

    for filename, old, new, reason in REMOTE_CODE_PATCHES:
        path = target / filename
        if not path.exists():
            continue
        source = path.read_text()

        # Deciding "is this already applied?" needs care in both directions.
        #
        # An ADDITIVE patch keeps the original text inside its replacement (adding an
        # import above an existing one). There, `old` still matches after patching, so
        # testing `old` would re-apply it on every call and stack up duplicate lines.
        # The replacement text is the reliable marker.
        #
        # A SUBSTITUTING patch replaces the original outright, but its replacement text
        # may legitimately already appear elsewhere in the file at call sites the patch
        # does not touch. There, testing `new` would skip a patch that is still needed,
        # so the original text is the reliable marker.
        additive = old in new
        if additive and new in source:
            continue
        if old in source:
            path.write_text(source.replace(old, new))
            changed.append(filename)
            _LOG.info("patched %s: %s", filename, reason)
        elif new not in source:
            _LOG.warning(
                "remote-code patch for %s no longer applies: neither the original nor the "
                "patched text was found. Upstream changed; re-check the shim (%s).",
                filename,
                reason,
            )

    return changed


class EarthDialBackbone:
    """Lazily-loaded handle on one EarthDial checkpoint plus an optional LoRA adapter.

    Construction is free and side-effect free; weights are touched only by
    :meth:`load`. That is what lets the registry hold a factory per tool without
    ``import satquery`` pulling several billion parameters onto a laptop.
    """

    def __init__(self, config: BackboneConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._adapter_path: Path | None = None
        self._local_dir: Path | None = None

    @property
    def is_loaded(self) -> bool:
        """True once :meth:`load` has completed successfully."""
        return self._model is not None

    @property
    def model(self) -> Any:
        """The underlying model, loading it on first access."""
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """The tokenizer, loading the backbone on first access."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def snapshot(self, allow_download: bool = True) -> Path:
        """Return the local checkpoint directory, fetching it if allowed.

        Args:
            allow_download: When False, raise instead of pulling ~8 GB over the network.
        """
        if self._local_dir is not None:
            return self._local_dir

        from huggingface_hub import snapshot_download

        self._local_dir = Path(
            snapshot_download(
                self.config.hf_repo_id,
                revision=self.config.revision,
                local_files_only=not allow_download,
            )
        )
        return self._local_dir

    def load(self, *, allow_download: bool = True) -> None:
        """Materialise the backbone weights onto the configured device.

        Steps, in order: resolve the snapshot, repair the missing remote code, then
        build the tokenizer and model. Quantisation is applied when
        ``config.load_in_4bit`` is set -- which it must be on an 8 GB card, since the
        bf16 weights alone are 8.29 GB.
        """
        if self._model is not None:
            return

        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        local_dir = self.snapshot(allow_download=allow_download)
        ensure_remote_code(local_dir)

        dtype = getattr(torch, self.config.dtype)
        # transformers 5 renamed the argument; 4.x still expects `torch_dtype`.
        dtype_kwarg = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

        kwargs: dict[str, Any] = {
            dtype_kwarg: dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.config.trust_remote_code,
            # Scaled dot-product attention, not the eager path.
            #
            # flash-attn is not installed here, and without this the model falls back to
            # eager, which materialises the full (heads, seq, seq) attention matrix in
            # float32. At 6 tiles plus a thumbnail that is ~1,800 image tokens, and a
            # single layer's softmax asks for 436 MB. Measured: an OOM on an 8 GB card
            # with 3 GB free, inside Phi-3's softmax.
            #
            # SDPA computes the same result without ever holding that matrix. The model
            # ships the implementation (PHI3_ATTENTION_CLASSES), it was simply never
            # requested. Falls back to eager below if this build refuses it, because a
            # slower model that runs beats a faster one that OOMs.
            "attn_implementation": self.config.attn_implementation,
        }

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        kwargs["device_map"] = self.config.device_map

        _LOG.info(
            "loading %s from %s (dtype=%s, 4bit=%s)",
            self.config.hf_repo_id,
            local_dir,
            self.config.dtype,
            self.config.load_in_4bit,
        )
        try:
            self._model = AutoModel.from_pretrained(str(local_dir), **kwargs).eval()
        except (ValueError, TypeError) as exc:
            # Some remote-code builds reject the argument outright rather than ignoring it.
            _LOG.warning(
                "attn_implementation=%s refused (%s); falling back to the build default",
                kwargs.get("attn_implementation"),
                exc,
            )
            kwargs.pop("attn_implementation", None)
            self._model = AutoModel.from_pretrained(str(local_dir), **kwargs).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(local_dir), trust_remote_code=self.config.trust_remote_code, use_fast=False
        )

        if self.config.is_adapted:
            try:
                self.attach_adapter(self.config.adapter_path)
            except Exception:
                # Do not leave a half-configured backbone behind. The base weights are
                # already loaded at this point, so a swallowed or retried failure would
                # let every later call quietly run UNADAPTED and score as though the
                # adapter were applied -- which is indistinguishable from a real result
                # in a table. Discard the model so the next call fails the same way
                # instead of silently degrading.
                self._model = None
                self._tokenizer = None
                raise

    def attach_adapter(self, adapter_path: str | Path | None = None) -> None:
        """Load a trained LoRA adapter on top of the base weights.

        This is what turns the generic backbone into the remote-sensing-adapted model
        the problem statement requires. Without it the model is unadapted and, per
        the architecture, fails SIH26167 outright.
        """
        from peft import PeftModel

        # Refuse to stack an adapter on an adapter, and check it BEFORE validating the
        # path: this is a fact about the backbone's state, not about the argument, so it
        # holds even when the requested path is bogus.
        #
        # Nested PeftModels do not compose -- both end up inert and the model silently
        # behaves as the unadapted base, which is indistinguishable from "the fine-tune
        # achieved nothing". That has now bitten twice: in training (a fresh LoRA over a
        # config-loaded adapter) and in evaluation (an explicit adapter over the same
        # config-loaded one), where every checkpoint reported base-model scores to four
        # decimal places.
        if isinstance(self._model, PeftModel):
            raise RuntimeError(
                f"an adapter is already attached to {self.config.hf_repo_id}; attaching "
                f"another on top would nest PeftModels and silently disable both. Load "
                "the backbone with adapter_path=None and attach exactly one adapter."
            )

        path = Path(adapter_path or self.config.adapter_path or "")
        if not path or str(path) == ".":
            raise ValueError("no adapter path was given and none is set on the config")
        if not path.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {path}. Train one with `make train-lora`; "
                "there is no adapter in the repo and none is downloaded."
            )

        if self._model is None:
            self.load()
            # `load()` attaches the config's adapter when one is named, so re-check.
            if isinstance(self._model, PeftModel):
                raise RuntimeError(
                    f"loading {self.config.hf_repo_id} already attached "
                    f"{self.config.adapter_path}; attaching {path} on top would nest "
                    "PeftModels and silently disable both."
                )

        self._model = PeftModel.from_pretrained(self._model, str(path)).eval()
        self._adapter_path = path
        _LOG.info("attached LoRA adapter from %s", path)

    def encode_image(self, image: np.ndarray) -> Any:
        """Tile and normalise one RGB image into a model-ready pixel tensor.

        Returns:
            ``(n_tiles, 3, image_size, image_size)`` tensor in the backbone's dtype.
        """
        import torch

        tiles = dynamic_tiles(
            image, tile_size=self.config.image_size, max_tiles=self.config.max_tiles
        )
        tensor = torch.from_numpy(tiles).to(getattr(torch, self.config.dtype))
        if self._model is not None:
            tensor = tensor.to(next(self._model.parameters()).device)
        return tensor

    def generate(
        self,
        images: list[np.ndarray],
        instruction: str,
        **generation_kwargs: Any,
    ) -> str:
        """Answer `instruction` about `images`.

        Args:
            images: One or two ``(height, width, 3)`` uint8 RGB arrays. A rendered SAR
                scene from ``preprocess.sar.render_sar`` is already in this layout.
            instruction: The prompt. It MUST already carry its ``<gsd:...m>`` token --
                see the module docstring. This method never adds one.
            **generation_kwargs: Passed through to the model's generation config
                (``max_new_tokens``, ``temperature``, ``num_beams``, ...).

        Returns:
            The model's answer text.
        """
        import torch

        if not images:
            raise ValueError("generate needs at least one image")
        if self._model is None:
            self.load()

        tile_batches = [
            dynamic_tiles(image, tile_size=self.config.image_size, max_tiles=self.config.max_tiles)
            for image in images
        ]
        num_patches_list = [batch.shape[0] for batch in tile_batches]

        device = next(self._model.parameters()).device
        pixel_values = (
            torch.from_numpy(np.concatenate(tile_batches, axis=0))
            .to(getattr(torch, self.config.dtype))
            .to(device)
        )

        # InternVL expects one <image> placeholder per image when several are supplied.
        question = instruction
        if len(images) > 1 and "<image>" not in question:
            question = "".join(f"Image-{i + 1}: <image>\n" for i in range(len(images))) + question

        config: dict[str, Any] = {"max_new_tokens": 128, "do_sample": False}
        config.update(generation_kwargs)
        if config.get("temperature"):
            config["do_sample"] = True
        else:
            config.pop("temperature", None)

        with torch.inference_mode():
            return self._model.chat(
                tokenizer=self._tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=config,
                num_patches_list=num_patches_list if len(images) > 1 else None,
            )

    def unload(self) -> None:
        """Release the weights and empty the CUDA cache."""
        self._model = None
        self._tokenizer = None
        self._adapter_path = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "not loaded"
        adapter = f", adapter={self._adapter_path}" if self._adapter_path else ""
        return f"EarthDialBackbone({self.config.hf_repo_id!r}, {state}{adapter})"
