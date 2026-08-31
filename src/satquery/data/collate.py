"""Batch assembly for the unified `Sample` record.

Three functions with three different maturities, deliberately:

- `collate_samples` is the tensor-producing collator handed to a torch DataLoader.
  It needs a processor and tokenizer from the backbone checkpoint, so it is an
  honest stub.
- `group_by_answer_type` and `assert_homogeneous_task` are pure, need no GPU and no
  weights, and are fully implemented and useful now -- the eval harness slices by
  answer type, and the batch sampler needs task homogeneity enforced.

The task-homogeneity rule is the load-bearing one. A single-image VQA sample carries
one `ImageRef` and a bi-temporal change sample carries two; a batch mixing them has
no single image-tensor shape and no single prompt template, and the failure surfaces
deep inside the model as a shape error rather than here as a sentence. The batch
sampler must group by `TaskType` and this function is what proves it did.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar

from satquery.data.schema import AnswerType, Sample
from satquery.preprocess.constants import BOX_COORDINATE_SCALE
from satquery.serve.contracts import TaskType
from satquery.utils.logging import get_logger

__all__ = [
    "FusionCollator",
    "PackedFusionCollator",
    "assert_homogeneous_task",
    "collate_samples",
    "group_by_answer_type",
]

logger = get_logger(__name__)


def collate_samples(samples: list[Sample]) -> dict[str, Any]:
    """Assemble a list of `Sample` records into one model-ready batch.

    The intended implementation prefixes every instruction via `Sample.prompt()` so
    the frozen GSD token is present, renders each `ImageRef` through the preprocessing
    pipeline (SAR through `preprocess.sar`, optical through `preprocess.optical`),
    runs the backbone's processor over image and text, and pads to the batch's longest
    sequence -- returning tensors only, never a `Sample`.

    Args:
        samples: Records for one batch. Must be task-homogeneous; call
            `assert_homogeneous_task` first.

    Returns:
        A mapping of tensor names to batched tensors.

    Raises:
        NotImplementedError: Always, until the processor wiring exists.
    """
    raise NotImplementedError(
        "collate_samples is not implemented. Missing: the processor/tokenizer wiring for the "
        "backbone -- transformers.AutoProcessor and AutoTokenizer loaded from the EarthDial-4B "
        "checkpoint named in configs/model (akshaydudhane/EarthDial_4B_RGB plus the non-optical "
        "checkpoint), the ImageRef-to-pixel-tensor path through satquery.preprocess, and the "
        f"padding/label-masking policy. Received {len(samples)} sample(s)."
    )


def group_by_answer_type(samples: list[Sample]) -> dict[AnswerType, list[Sample]]:
    """Partition samples by how they are scored, preserving input order within a group.

    Fully implemented. `AnswerType` is exactly the axis the loss and the metric branch
    on -- generation for OPEN_TEXT, exact match for CLOSED_SET and BINARY, box
    regression for BOXES -- so this is the grouping the eval harness reports against
    and the one a multi-objective step splits on.

    Args:
        samples: Records to partition. May be empty.

    Returns:
        A plain dict from answer type to its samples. Answer types absent from the
        input are absent from the result rather than mapping to an empty list.
    """
    grouped: defaultdict[AnswerType, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.answer_type].append(sample)
    return dict(grouped)


def assert_homogeneous_task(samples: list[Sample]) -> TaskType:
    """Return the single `TaskType` shared by every sample, or fail loudly.

    Fully implemented. Called by the collator's caller before batching; see the module
    docstring for why a mixed-task batch cannot be collated at all.

    Args:
        samples: Records for one batch.

    Returns:
        The task every sample in the batch carries.

    Raises:
        ValueError: If `samples` is empty, or if more than one task is present. The
            message names each offending task and one sample id per task so the
            batch sampler's grouping bug is locatable from the log alone.
    """
    if not samples:
        raise ValueError("cannot determine a batch task from an empty sample list")

    first_by_task: dict[TaskType, str] = {}
    for sample in samples:
        first_by_task.setdefault(sample.task, sample.sample_id)

    if len(first_by_task) > 1:
        offenders = ", ".join(
            f"{task.value} (e.g. sample {sample_id!r})"
            for task, sample_id in sorted(first_by_task.items(), key=lambda item: item[0].value)
        )
        raise ValueError(
            f"batch mixes {len(first_by_task)} tasks: {offenders}. A batch must be "
            "task-homogeneous: tasks differ in image count and prompt template, so they have no "
            "common collated shape. Group by TaskType in the batch sampler."
        )

    return next(iter(first_by_task))


def _conv_template_factory(model: Any):
    """Return the checkpoint's own `get_conv_template`, unwrapping PEFT if needed.

    Taken from the loaded model's module rather than imported by path: the remote code
    lives under a hashed `transformers_modules.<sha>` package whose name is not known
    ahead of time, and using the model's own function guarantees the template matches
    the weights.
    """
    import sys

    # Unwrap repeatedly, not once: a model can end up wrapped more than once (a PeftModel
    # around a PeftModel), and a single unwrap then lands on peft's module rather than the
    # checkpoint's remote code.
    candidates = [model]
    current = model
    for _ in range(4):
        unwrap = getattr(current, "get_base_model", None)
        if unwrap is None:
            break
        current = unwrap()
        candidates.append(current)

    for candidate in candidates:
        module = sys.modules.get(type(candidate).__module__)
        factory = getattr(module, "get_conv_template", None)
        if factory is not None:
            return factory
    raise AttributeError(
        "could not find get_conv_template on the model's module. The checkpoint's remote "
        "code may not be loaded; run `make fetch-model`."
    )


@dataclass
class InternVLCollator:
    """Turn `Sample` records into the batch `InternVLChatModel.forward` expects.

    The model does not take images and text separately. It expands one `<image>`
    placeholder into ``num_image_token * n_tiles`` copies of `<IMG_CONTEXT>`, embeds the
    text, and then *overwrites* the embeddings at those positions with vision features.
    So the count of `<IMG_CONTEXT>` tokens in `input_ids` must equal the number of tiles
    actually passed in `pixel_values`, exactly. A mismatch does not raise -- the model
    catches it and silently truncates -- so it is asserted here instead.

    Label masking: the prompt is tokenised twice, once with the answer and once without,
    and everything up to the answer is set to `IGNORE_INDEX`. Training on the prompt
    tokens as well would teach the model to generate questions.

    The conversation template, its system message and the image token strings all come
    from the loaded model rather than being restated, so this cannot drift from what the
    checkpoint was trained with.
    """

    tokenizer: Any
    model: Any
    image_size: int = 448
    max_tiles: int = 1
    max_length: int = 2048
    ignore_index: int = -100

    IMG_START: ClassVar[str] = "<img>"
    IMG_END: ClassVar[str] = "</img>"
    IMG_CONTEXT: ClassVar[str] = "<IMG_CONTEXT>"

    def __post_init__(self) -> None:
        """Resolve the image-context token id and hand it to the model, as `chat` does."""
        self._conv_factory = _conv_template_factory(self.model)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT)
        inner = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        inner.img_context_token_id = self.img_context_token_id
        self.num_image_token = int(inner.num_image_token)
        self._inner = inner

    def _prompt(self, instruction: str, answer: str | None) -> str:
        """Render one turn through the model's own conversation template."""
        template = self._conv_factory(self._inner.template)
        template.system_message = self._inner.system_message
        template.append_message(template.roles[0], f"<image>\n{instruction}")
        template.append_message(template.roles[1], answer)
        return template.get_prompt()

    def encode(self, sample: Sample) -> dict[str, Any]:
        """Encode a single sample. Separated from `__call__` so it is unit-testable."""
        import numpy as np
        import torch
        from PIL import Image

        from satquery.models.vlm.backbone import dynamic_tiles

        with Image.open(sample.images[0].path) as handle:
            pixels = dynamic_tiles(
                np.asarray(handle.convert("RGB")),
                tile_size=self.image_size,
                max_tiles=self.max_tiles,
            )
        n_tiles = pixels.shape[0]

        image_tokens = (
            self.IMG_START + self.IMG_CONTEXT * self.num_image_token * n_tiles + self.IMG_END
        )
        # `Sample.prompt()` is what supplies the frozen GSD token.
        instruction = sample.prompt()
        target = target_text(sample)

        prompt_only = self._prompt(instruction, None).replace("<image>", image_tokens, 1)
        full = self._prompt(instruction, target).replace("<image>", image_tokens, 1)

        prompt_ids = self.tokenizer(prompt_only, return_tensors="pt", add_special_tokens=False)
        full_ids = self.tokenizer(
            full,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = full_ids["input_ids"][0]
        labels = input_ids.clone()
        labels[: prompt_ids["input_ids"].shape[1]] = self.ignore_index

        n_context = int((input_ids == self.img_context_token_id).sum())
        expected = self.num_image_token * n_tiles
        if n_context != expected:
            raise ValueError(
                f"{sample.sample_id}: {n_context} <IMG_CONTEXT> tokens survived tokenisation "
                f"but {expected} tiles' worth were expected. The prompt was almost certainly "
                f"truncated at max_length={self.max_length}; raise it or lower max_tiles."
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": torch.from_numpy(pixels),
            "n_tiles": n_tiles,
        }

    def __call__(self, samples: list[Sample]) -> dict[str, Any]:
        """Collate a batch, right-padding to the longest sequence in it."""
        import torch

        encoded = [self.encode(s) for s in samples]
        longest = max(e["input_ids"].shape[0] for e in encoded)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        input_ids, labels, masks = [], [], []
        for e in encoded:
            length = e["input_ids"].shape[0]
            padding = longest - length
            input_ids.append(torch.cat([e["input_ids"], torch.full((padding,), pad_id)]))
            labels.append(torch.cat([e["labels"], torch.full((padding,), self.ignore_index)]))
            masks.append(torch.cat([torch.ones(length), torch.zeros(padding)]))

        pixel_values = torch.cat([e["pixel_values"] for e in encoded], dim=0)
        return {
            "input_ids": torch.stack(input_ids).long(),
            "labels": torch.stack(labels).long(),
            "attention_mask": torch.stack(masks).long(),
            "pixel_values": pixel_values.to(torch.bfloat16),
            # Every tile we pass is a real tile; the model uses this to drop padding tiles.
            "image_flags": torch.ones(pixel_values.shape[0], 1, dtype=torch.long),
        }


def target_text(sample: Sample) -> str:
    """The string the model should learn to produce for this sample.

    Grounding targets are re-encoded back into the `{<x1><y1><x2><y2>}` form on the
    frozen 0-100 scale, because that is the format the checkpoint emits and the format
    `parse_boxes` reads back. Training on pixel coordinates would teach a format nothing
    downstream can parse.
    """
    if sample.answer_type is AnswerType.BOXES:
        image = sample.images[0]
        width = float(image.width or 1)
        height = float(image.height or 1)
        parts = []
        for box in sample.boxes:
            parts.append(
                f"{{<{box.x_min / width * BOX_COORDINATE_SCALE:.0f}><{box.y_min / height * BOX_COORDINATE_SCALE:.0f}><{box.x_max / width * BOX_COORDINATE_SCALE:.0f}><{box.y_max / height * BOX_COORDINATE_SCALE:.0f}>}}"
            )
        return "".join(parts)
    return sample.answer_text or ""


@dataclass
class ChangeHeadCollator:
    """Batch `Sample` records for the two-tower CDVQA head.

    Emits `pixel_values_t1`, `pixel_values_t2`, `question_ids` and `labels`, matching
    `build_change_head`'s forward signature.

    Images are ImageNet-normalised with the same frozen constants the VLM path uses, so
    the two models never disagree about what a pixel means.
    """

    vocabulary: dict[str, int]
    image_size: int = 256
    max_question_length: int = 24

    def encode(self, sample: Sample) -> dict[str, Any]:
        """Encode one bi-temporal record. Separated from `__call__` for testability."""
        import numpy as np
        import torch
        from PIL import Image

        from satquery.models.change.siamese import answer_index, encode_question
        from satquery.preprocess.constants import IMAGENET_MEAN, IMAGENET_STD

        if len(sample.images) != 2:
            raise ValueError(
                f"{sample.sample_id}: the change head needs two dates, got {len(sample.images)}"
            )

        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        dates = []
        for ref in sample.images:
            with Image.open(ref.path) as handle:
                resized = handle.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.BILINEAR
                )
            array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            dates.append(torch.from_numpy((array - mean) / std))

        return {
            "pixel_values_t1": dates[0],
            "pixel_values_t2": dates[1],
            "question_ids": torch.tensor(
                encode_question(sample.instruction, self.vocabulary, self.max_question_length),
                dtype=torch.long,
            ),
            "labels": torch.tensor(answer_index(sample.answer_text or ""), dtype=torch.long),
        }

    def __call__(self, samples: list[Sample]) -> dict[str, Any]:
        """Stack a batch. Every field is fixed-length, so no padding is needed."""
        import torch

        encoded = [self.encode(s) for s in samples]
        return {key: torch.stack([e[key] for e in encoded]) for key in encoded[0]}


@dataclass(slots=True)
class FusionCollator:
    """Batch `Sample` records for the dual-encoder fusion segmenter.

    Emits `pixel_values_optical`, `pixel_values_sar` and `labels`, matching
    `build_fusion_model`'s forward signature.

    Unlike the VLM and change-head collators this one does **not** resize. The fusion
    head is a segmenter whose output is a georeferenced mask, and resizing would change
    the ground sampling distance the GSD token claims. `build_fusion_model` pads
    internally to the encoder stride instead, so patches arrive at their true scale.

    Optical bands are scaled to [0, 1] rather than ImageNet-normalised, because there
    are four of them and the three-channel ImageNet statistics do not describe a NIR
    band at all. The SAR branch takes the frozen pseudo-RGB rendering, which is a uint8
    image, so it does use the ImageNet constants -- its encoder is initialised from
    ImageNet weights and expects that input distribution.

    The rasters this reads are RAW: unstretched reflectance and unrendered linear
    backscatter, exactly what a real request carries. The frozen `stretch_to_uint8` and
    `render_sar` are applied here, so this collator and `FusionExtractionTool` perform
    identical arithmetic on identical inputs -- which is the property that makes an
    evaluation through this path meaningful.
    """

    targets: tuple[str, ...]
    optical_bands: int = 4

    def encode(self, sample: Sample) -> dict[str, Any]:
        """Encode one co-registered pair. Separated from `__call__` for testability."""
        import numpy as np
        import torch

        from satquery.io.raster import read_raster
        from satquery.preprocess.constants import (
            FUSION_EXTRACTION_CLASSES,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

        if len(sample.images) != 2:
            raise ValueError(
                f"{sample.sample_id}: fusion needs an optical and a SAR image, got "
                f"{len(sample.images)}"
            )
        if sample.mask_path is None:
            raise ValueError(f"{sample.sample_id}: fusion training needs a mask_path")

        from satquery.preprocess.optical import stretch_to_uint8
        from satquery.preprocess.sar import render_sar

        optical, _ = read_raster(sample.images[0].path)
        if optical.shape[0] != self.optical_bands:
            raise ValueError(
                f"{sample.sample_id}: expected {self.optical_bands} optical bands, got "
                f"{optical.shape[0]}"
            )
        stretched = stretch_to_uint8(optical)
        optical_tensor = torch.from_numpy(np.asarray(stretched, dtype=np.float32) / 255.0)

        sar, _ = read_raster(sample.images[1].path)
        if sar.shape[0] < 2:
            raise ValueError(
                f"{sample.sample_id}: SAR raster needs two polarisations, got {sar.shape[0]}"
            )
        rendered, _ = render_sar(sar[0], sar[1])
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        sar_tensor = torch.from_numpy(
            (np.transpose(np.asarray(rendered, dtype=np.float32), (2, 0, 1)) / 255.0 - mean) / std
        )

        mask, _ = read_raster(sample.mask_path)
        codes = np.asarray(mask[0])
        # The mask is categorical -- 0 background, then FUSION_EXTRACTION_CLASSES in
        # order offset by one -- and is expanded here into one binary plane per
        # requested target. Indexing through the frozen tuple rather than positionally
        # keeps a `targets` subset selecting the right planes.
        planes = [
            (codes == FUSION_EXTRACTION_CLASSES.index(name) + 1).astype(np.float32)
            for name in self.targets
        ]
        return {
            "pixel_values_optical": optical_tensor,
            "pixel_values_sar": sar_tensor,
            "labels": torch.from_numpy(np.stack(planes)),
        }

    def __call__(self, samples: list[Sample]) -> dict[str, Any]:
        """Stack a list of encoded pairs into one batch."""
        import torch

        encoded = [self.encode(sample) for sample in samples]
        return {key: torch.stack([item[key] for item in encoded]) for key in encoded[0]}


@dataclass(slots=True)
class PackedFusionCollator:
    """Batch fusion records straight out of a `PackedFusionDataset` memmap.

    Same tensors as `FusionCollator`, sourced from the packed cache instead of three
    GeoTIFFs per patch. It holds a reference to the dataset because the packed bytes
    live there; `Sample.metadata["packed_index"]` is the row to fetch.

    The arithmetic is deliberately identical to `FusionCollator` -- optical scaled to
    [0, 1], SAR ImageNet-normalised, mask expanded to one binary plane per target --
    because the packer already applied the frozen `stretch_to_uint8` and `render_sar`.
    `tests/test_fusion_packed.py` asserts the two collators agree bit for bit rather
    than leaving that to inspection.
    """

    datasets: dict[str, Any]
    """Packed datasets by `str(cache_dir)`. Keyed rather than singular because
    `Trainer` shares one collator across train and eval, and looking an eval row up in
    the training memmap yields a valid-looking tensor for the wrong patch."""

    targets: tuple[str, ...]

    def encode(self, sample: Sample) -> dict[str, Any]:
        """Encode one packed record. Separated from `__call__` for testability."""
        import numpy as np
        import torch

        from satquery.preprocess.constants import (
            FUSION_EXTRACTION_CLASSES,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

        index = sample.metadata.get("packed_index")
        cache = sample.metadata.get("packed_cache")
        if index is None or cache is None:
            raise ValueError(
                f"{sample.sample_id}: PackedFusionCollator needs metadata['packed_index'] "
                "and ['packed_cache']; this record did not come from a PackedFusionDataset"
            )
        dataset = self.datasets.get(str(cache))
        if dataset is None:
            raise KeyError(
                f"{sample.sample_id}: no packed dataset registered for {cache!r}; "
                f"registered caches are {sorted(self.datasets)}"
            )
        optical, sar, mask = dataset.arrays(int(index))

        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

        planes = np.stack(
            [
                (mask == FUSION_EXTRACTION_CLASSES.index(name) + 1).astype(np.float32)
                for name in self.targets
            ]
        )
        return {
            "pixel_values_optical": torch.from_numpy(np.asarray(optical, dtype=np.float32) / 255.0),
            "pixel_values_sar": torch.from_numpy(
                (np.asarray(sar, dtype=np.float32) / 255.0 - mean) / std
            ),
            "labels": torch.from_numpy(planes),
        }

    def __call__(self, samples: list[Sample]) -> dict[str, Any]:
        """Stack a list of encoded pairs into one batch."""
        import torch

        encoded = [self.encode(sample) for sample in samples]
        return {key: torch.stack([item[key] for item in encoded]) for key in encoded[0]}
