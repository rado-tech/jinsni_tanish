"""Fine-tune wav2vec2-base as a transformer baseline (GPU required).

Reference point for the from-scratch CNN. Three design choices carry most of the
result:

* the convolutional feature encoder stays frozen — it encodes low-level acoustics
  that transfer as-is, and freezing it cuts ~30% of the step time;
* the backbone learns at 3e-5 while the freshly initialised head learns at 1e-3;
  a single large learning rate would destroy the pretrained weights;
* mixed precision roughly halves step time on any modern GPU.

Takes ~15 minutes for three epochs on a T4.

Usage:
    python scripts/06_finetune_w2v2.py [--quick] [--epochs 3] [--batch-size 16]

Outputs (models/):
    w2v2_best/, w2v2_history.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

MODEL_ID = "facebook/wav2vec2-base"
SAMPLE_RATE = 16_000
CROP_SAMPLES = SAMPLE_RATE * 2      # same 2 s context as the CNN
LR_BACKBONE, LR_HEAD = 3e-5, 1e-3
WEIGHT_DECAY, WARMUP_RATIO = 0.01, 0.1
SEED = 42
LABELS = ["Ayol", "Erkak"]

# Colab runs from /content; a local checkout runs from the project root.
CWD = Path.cwd()
AUDIO_DIR = CWD / "audio" if (CWD / "audio").exists() else CWD / "data/processed/audio"
METADATA = (
    CWD / "metadata.csv" if (CWD / "metadata.csv").exists()
    else CWD / "data/processed/metadata.csv"
)
MODEL_DIR = CWD / "models"


class AudioDataset(Dataset):
    """Serves raw 2-second waveforms; wav2vec2 has its own feature encoder."""

    def __init__(self, df, train_mode, seed=SEED):
        self.files = df["filename"].to_numpy()
        self.labels = df["gender"].map({g: i for i, g in enumerate(LABELS)}).to_numpy()
        self.train_mode = train_mode
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        waveform, _ = sf.read(AUDIO_DIR / self.files[index], dtype="float32")

        if len(waveform) <= CROP_SAMPLES:
            start = 0
        elif self.train_mode:
            start = self.rng.integers(0, len(waveform) - CROP_SAMPLES + 1)
        else:
            start = (len(waveform) - CROP_SAMPLES) // 2

        crop = waveform[start:start + CROP_SAMPLES]
        if len(crop) < CROP_SAMPLES:
            # Zero is silence in the raw-waveform domain (unlike the dB domain).
            crop = np.pad(crop, (0, CROP_SAMPLES - len(crop)))
        return crop.astype(np.float32), int(self.labels[index])


def build_model() -> Wav2Vec2ForSequenceClassification:
    """Load pretrained weights and freeze the convolutional feature encoder.

    A warning about newly initialised classifier weights is expected — that head
    does not exist in the pretraining checkpoint.
    """
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS)
    )
    model.freeze_feature_encoder()
    return model


def build_optimizer(model) -> torch.optim.Optimizer:
    """Discriminative learning rates: cautious backbone, fast classifier head."""
    head, backbone = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target = head if ("classifier" in name or "projector" in name) else backbone
        target.append(param)

    return torch.optim.AdamW(
        [{"params": backbone, "lr": LR_BACKBONE}, {"params": head, "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = correct = seen = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_values=x).logits
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()               # per step, not per epoch

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += y.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = seen = 0
    targets, predictions = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_values=x).logits
            loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == y).sum().item()
        seen += y.size(0)
        targets.append(y.cpu().numpy())
        predictions.append(predicted.cpu().numpy())
    return (
        total_loss / seen,
        correct / seen,
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def make_collate(feature_extractor):
    def collate(batch):
        waveforms = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = feature_extractor(
            waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        return encoded.input_values, labels

    return collate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected; this will be extremely slow.")
    print(f"device={device} audio={AUDIO_DIR}")

    df = pd.read_csv(METADATA)
    epochs = 1 if args.quick else args.epochs
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    collate = make_collate(feature_extractor)

    loaders = {}
    for split in ("train", "val"):
        subset = df[df["split"] == split].reset_index(drop=True)
        if args.quick:
            subset = subset.iloc[: 400 if split == "train" else 200]
        is_train = split == "train"
        loaders[split] = DataLoader(
            AudioDataset(subset, is_train),
            batch_size=args.batch_size,
            shuffle=is_train,
            num_workers=2,
            collate_fn=collate,
            pin_memory=(device == "cuda"),
            drop_last=is_train,
        )
    print(f"train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} "
          f"epochs={epochs}")

    model = build_model().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable={trainable:,} frozen={frozen:,}")
    assert frozen > 0, "feature encoder was not frozen"

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)
    total_steps = len(loaders["train"]) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps
    )
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    history, best_accuracy = [], 0.0
    targets = predictions = np.array([])

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler, scaler, device
        )
        val_loss, val_acc, targets, predictions = evaluate(
            model, loaders["val"], criterion, device
        )
        elapsed = time.time() - started
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "seconds": round(elapsed, 1),
        })

        marker = ""
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            model.save_pretrained(MODEL_DIR / "w2v2_best")
            feature_extractor.save_pretrained(MODEL_DIR / "w2v2_best")
            (MODEL_DIR / "w2v2_best" / "labels.json").write_text(
                json.dumps({"labels": LABELS, "crop_samples": CROP_SAMPLES,
                            "sample_rate": SAMPLE_RATE}),
                encoding="utf-8",
            )
            marker = "  <- saved"
        print(f"epoch {epoch}/{epochs}  train {train_loss:.4f}/{train_acc:.4f}  "
              f"val {val_loss:.4f}/{val_acc:.4f}  ({elapsed / 60:.1f} min){marker}")

    print(f"\nBest validation accuracy: {best_accuracy:.4f}")
    for i, label in enumerate(LABELS):
        mask = targets == i
        if mask.sum():
            print(f"  {label:6} recall = {(predictions[mask] == i).mean():.4f}")

    (MODEL_DIR / "w2v2_history.json").write_text(
        json.dumps({
            "history": history, "best_val_acc": best_accuracy, "model_id": MODEL_ID,
            "trainable": trainable, "frozen": frozen, "quick": args.quick,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
