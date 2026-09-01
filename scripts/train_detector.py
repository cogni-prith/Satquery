#!/usr/bin/env python3
"""Fine-tune a detector on the packed VRSBench boxes.

Faster R-CNN with a pretrained ResNet-50 FPN backbone, box predictor replaced for this
vocabulary. Pretrained, not from scratch: 4,000 images is nowhere near enough to learn
general object features, and a scratch detector would mostly learn the class prior.

Two limits of the supervision, both structural and neither fixable by training longer.

VRSBench annotates one box per referring expression, so a scene with forty vehicles may
carry one labelled box. Every unannotated instance is fed to the loss as background --
the model is actively punished for finding real objects. That caps recall, and the cap is
a property of the labels, not of the model.

And the vocabulary is 26 fixed classes. A detector fitted to it detects those 26 things.
The tool is named `detector.openvocab`, which this training data cannot deliver, so the
spec description says closed-vocabulary and lists what it covers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.preprocess.constants import constants_fingerprint
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything

LOG = get_logger("train-detector")


class PackedDetection:
    """The packed cache as a torchvision detection dataset."""

    def __init__(self, path: Path, indices: np.ndarray, size: int = 384) -> None:
        blob = np.load(path, allow_pickle=True)
        self.images = blob["images"]
        self.boxes = blob["boxes"]
        self.labels = blob["labels"]
        self.vocabulary = [str(v) for v in blob["vocabulary"]]
        self.indices = indices
        self.size = size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        import torch

        index = int(self.indices[position])
        image = torch.from_numpy(self.images[index].astype(np.float32) / 255.0).permute(2, 0, 1)
        # Normalised boxes to absolute pixels. Kept normalised in the cache precisely so
        # the resize already applied has only one place to be undone.
        boxes = torch.from_numpy(np.asarray(self.boxes[index], dtype=np.float32) * self.size)
        # torchvision's label 0 is background, so the vocabulary shifts up by one.
        labels = torch.from_numpy(np.asarray(self.labels[index], dtype=np.int64) + 1)
        return image, {"boxes": boxes, "labels": labels}


def collate(batch):
    return tuple(zip(*batch, strict=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="scratch/detection/train.npz")
    parser.add_argument("--out", default="train/detector")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    configure_logging()
    seed_everything(args.seed)
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint("pixels"))

    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    from satquery.utils.paths import artifact_root

    root = Path(__file__).resolve().parents[1]
    cache = root / args.cache
    if not cache.exists():
        LOG.error("no cache at %s; run scripts/pack_detection.py first", cache)
        return 1

    blob = np.load(cache, allow_pickle=True)
    total = len(blob["images"])
    vocabulary = [str(v) for v in blob["vocabulary"]]
    del blob

    order = np.random.default_rng(args.seed).permutation(total)
    holdout = max(1, total // 10)
    train_set = PackedDetection(cache, order[holdout:])
    eval_set = PackedDetection(cache, order[:holdout])
    LOG.info(
        "train %d images, held out %d, %d classes", len(train_set), len(eval_set), len(vocabulary)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(vocabulary) + 1)
    model.to(device)

    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=2
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs * len(loader)
    )

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for step, (images, targets) in enumerate(loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimiser.zero_grad()
            loss.backward()
            # Detection losses spike when a batch happens to hold a tiny box; without
            # this a single bad batch can wreck a run that was converging.
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimiser.step()
            schedule.step()
            running += float(loss)
            if step % 100 == 0:
                LOG.info(
                    "  epoch %d step %d/%d  loss %.4f", epoch + 1, step, len(loader), running / step
                )
        LOG.info("epoch %d done, mean loss %.4f", epoch + 1, running / max(len(loader), 1))

    output = artifact_root() / args.out
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "image_size": 384,
            "architecture": "fasterrcnn_resnet50_fpn",
            "constants_fingerprint": constants_fingerprint("pixels"),
            "fingerprint_scope": "pixels",
        },
        output / "model.pt",
    )
    LOG.info("detector saved to %s", output / "model.pt")

    from scripts.eval_detector import evaluate

    evaluate(model, eval_set, device, vocabulary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
