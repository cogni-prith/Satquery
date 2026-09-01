#!/usr/bin/env python3
"""Train the land-cover segmenter on the packed reBEN cache.

Uses `transformers.Trainer` rather than a hand-written loop, as CLAUDE.md requires: the
checkpointing, resume, mixed precision and logging are the parts a hand-rolled loop gets
subtly wrong, and this run has already been interrupted once by a sleeping disk.

Expects `scripts/pack_landcover.py` to have written the cache. It reads no LMDB and
downloads nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.models.segmentation.landcover import LandCoverConfig, LandCoverSegmenter
from satquery.preprocess.constants import (
    LANDCOVER_CLASSES,
    LANDCOVER_IGNORE_INDEX,
    constants_fingerprint,
)
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("train-seg")


class PackedLandCover:
    """The packed cache as a torch dataset. Held in memory; the cache is sized to fit."""

    def __init__(self, path: Path, augment: bool = False) -> None:
        blob = np.load(path)
        self.images = blob["images"]
        self.masks = blob["masks"]
        self.augment = augment
        self.rng = np.random.default_rng(1337)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image = self.images[index].astype(np.float32)
        mask = self.masks[index].astype(np.int64)

        if self.augment:
            # Flips and rotations only. No colour jitter: the model is being asked to read
            # reflectance, and perturbing band values would teach it to ignore the very
            # signal the spectral indices key on.
            if self.rng.random() < 0.5:
                image, mask = image[:, :, ::-1], mask[:, ::-1]
            if self.rng.random() < 0.5:
                image, mask = image[:, ::-1, :], mask[::-1, :]
            turns = int(self.rng.integers(0, 4))
            if turns:
                image = np.rot90(image, turns, axes=(1, 2))
                mask = np.rot90(mask, turns)

        return {
            "pixel_values": np.ascontiguousarray(image),
            "labels": np.ascontiguousarray(mask),
        }


def build_metrics():
    """Per-class IoU, reported per class and never as a single mean.

    A mean IoU over five classes on reBEN is dominated by forest and farmland and can look
    healthy while water and built-up are near zero -- and those two are what the whole
    system's area answers rest on.
    """
    import torch

    def compute(eval_pred):
        logits, labels = eval_pred
        logits = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits
        labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        predicted = logits.argmax(dim=1)

        out: dict[str, float] = {}
        valid = labels != LANDCOVER_IGNORE_INDEX
        for index, name in enumerate(LANDCOVER_CLASSES):
            if index == LANDCOVER_IGNORE_INDEX:
                continue
            p = (predicted == index) & valid
            t = (labels == index) & valid
            union = (p | t).sum().item()
            # nan, not 0.0, when a class is absent from the eval split: nothing was
            # measured, and a zero would read as a failure to find it.
            out[f"iou_{name}"] = float((p & t).sum().item() / union) if union else float("nan")
        scored = [v for v in out.values() if not np.isnan(v)]
        out["iou_mean_present"] = float(np.mean(scored)) if scored else float("nan")
        return out

    return compute


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="scratch/landcover")
    parser.add_argument("--out", default="train/landcover")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    configure_logging()
    seed_everything(args.seed)
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint())

    import torch
    from transformers import Trainer, TrainingArguments

    from satquery.utils.paths import artifact_root

    root = Path(__file__).resolve().parents[1]
    train_path = root / args.cache / "train.npz"
    if not train_path.exists():
        LOG.error("no cache at %s. Run scripts/pack_landcover.py first.", train_path)
        return 1

    full = PackedLandCover(train_path, augment=True)
    # Held out from the same pack rather than a separate split: this measures whether the
    # head learned, not whether it generalises across regions. The reBEN validation split
    # is the honest test and is scored separately by `make eval`.
    holdout = max(1, len(full) // 10)
    indices = np.random.default_rng(args.seed).permutation(len(full))
    train_set = torch.utils.data.Subset(full, indices[holdout:].tolist())
    eval_source = PackedLandCover(train_path, augment=False)
    eval_set = torch.utils.data.Subset(eval_source, indices[:holdout].tolist())
    LOG.info("train %d patches, held out %d", len(train_set), len(eval_set))

    model = LandCoverSegmenter(LandCoverConfig())
    module = model.build()

    class SegTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(pixel_values=inputs["pixel_values"])
            loss = self.segmenter.loss(outputs.logits, labels)
            inputs["labels"] = labels
            return (loss, outputs) if return_outputs else loss

    SegTrainer.segmenter = model

    output_dir = artifact_root() / args.out
    trainer = SegTrainer(
        model=module,
        args=TrainingArguments(
            output_dir=str(output_dir),
            max_steps=args.steps,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=torch.cuda.is_available(),
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            # 250, not 500: this machine's external disk has already put one run to sleep
            # mid-training, and a checkpoint is cheap next to losing an hour.
            save_steps=250,
            save_total_limit=4,
            remove_unused_columns=False,
            report_to=[],
            dataloader_num_workers=2,
            seed=args.seed,
        ),
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=build_metrics(),
    )

    LOG.info("starting; output -> %s", output_dir)
    trainer.train()
    saved = model.save(output_dir / "final")
    LOG.info("segmenter saved to %s", saved)

    metrics = trainer.evaluate()
    for key, value in sorted(metrics.items()):
        if key.startswith("eval_iou"):
            LOG.info("  %-24s %s", key, f"{value:.4f}" if value == value else "nan (absent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
