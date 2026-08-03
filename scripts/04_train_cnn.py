"""Train the CNN from scratch on cached log-mel features.

Random 2-second crops act as augmentation during training; evaluation always
uses the centre crop so the metric is reproducible. The checkpoint with the best
validation accuracy is kept, not the last one.

Takes roughly 55 minutes on four CPU cores, or a few minutes on any GPU.

Usage:
    python scripts/04_train_cnn.py [--quick] [--epochs 15] [--lr 3e-3]

Outputs (models/):
    cnn_best.pt, cnn_history.json, cnn_curves.png
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from genderid.config import CROP_FRAMES, LABELS, N_MELS, PAD_DB
from genderid.model import GenderCNN

BATCH_SIZE = 64
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


class MelDataset(Dataset):
    """Serves normalised 2-second crops from the memory-mapped cache."""

    def __init__(self, rows, labels, mels, lengths, mean, std, train_mode, seed=SEED):
        self.rows = np.asarray(rows)
        self.labels = np.asarray(labels)
        self.mels = mels
        self.lengths = lengths
        self.mean = mean[:, None]
        self.std = std[:, None]
        self.train_mode = train_mode
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        length = self.lengths[row]
        mel = np.asarray(self.mels[row], dtype=np.float32)

        if length <= CROP_FRAMES:
            start = 0
        elif self.train_mode:
            start = self.rng.integers(0, length - CROP_FRAMES + 1)
        else:
            start = (length - CROP_FRAMES) // 2

        crop = mel[:, start:start + CROP_FRAMES]
        if crop.shape[1] < CROP_FRAMES:
            crop = np.pad(
                crop, ((0, 0), (0, CROP_FRAMES - crop.shape[1])), constant_values=PAD_DB
            )

        x = torch.from_numpy((crop - self.mean) / self.std).float().unsqueeze(0)
        return x, torch.tensor(self.labels[index], dtype=torch.long)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = seen = 0
    for x, y in loader:
        optimizer.zero_grad()          # PyTorch accumulates gradients otherwise
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += y.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()                       # switches BatchNorm to running statistics
    total_loss = correct = seen = 0
    targets, predictions = [], []
    for x, y in loader:
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == y).sum().item()
        seen += y.size(0)
        targets.append(y.numpy())
        predictions.append(predicted.numpy())
    return (
        total_loss / seen,
        correct / seen,
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def make_loader(df, split, mels, lengths, mean, std, quick):
    subset = df[df["split"] == split]
    if quick:
        subset = subset.iloc[: 750 if split == "train" else 400]
    labels = subset["gender"].map({g: i for i, g in enumerate(LABELS)}).to_numpy()
    assert not pd.isna(labels).any(), f"unmapped gender labels in split '{split}'"

    is_train = split == "train"
    dataset = MelDataset(
        subset.index.to_numpy(), labels, mels, lengths, mean, std, is_train
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=is_train, num_workers=0)


def plot_curves(history, out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, ylabel in ((axes[0], "loss", "loss"), (axes[1], "acc", "accuracy")):
        ax.plot(epochs, [h[f"train_{metric}"] for h in history], "o-", label="train")
        ax.plot(epochs, [h[f"val_{metric}"] for h in history], "o-", label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smoke test on a subset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(args.threads)
    MODEL_DIR.mkdir(exist_ok=True)
    epochs = 2 if args.quick else args.epochs

    df = pd.read_csv(PROCESSED_DIR / "metadata.csv")
    mels = np.load(PROCESSED_DIR / "mels.npy", mmap_mode="r")
    lengths = np.load(PROCESSED_DIR / "mel_lengths.npy")
    stats = json.loads((PROCESSED_DIR / "norm_stats.json").read_text())
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    loaders = {
        split: make_loader(df, split, mels, lengths, mean, std, args.quick)
        for split in ("train", "val")
    }
    print(f"train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} "
          f"epochs={epochs}")

    model = GenderCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    x_batch, _ = next(iter(loaders["val"]))
    assert x_batch.shape[1:] == (1, N_MELS, CROP_FRAMES), f"bad input {x_batch.shape}"
    assert model(x_batch).shape == (x_batch.shape[0], len(LABELS))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history, best_accuracy = [], 0.0
    targets = predictions = np.array([])

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer
        )
        val_loss, val_acc, targets, predictions = evaluate(
            model, loaders["val"], criterion
        )
        scheduler.step()
        elapsed = time.time() - started

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "seconds": round(elapsed, 1),
        })

        marker = ""
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "crop_frames": CROP_FRAMES,
                "labels": LABELS,
                "mean": mean.tolist(),
                "std": std.tolist(),
            }, MODEL_DIR / "cnn_best.pt")
            marker = "  <- saved"
        print(f"epoch {epoch:2d}/{epochs}  train {train_loss:.4f}/{train_acc:.4f}  "
              f"val {val_loss:.4f}/{val_acc:.4f}  ({elapsed / 60:.1f} min){marker}")

    print(f"\nBest validation accuracy: {best_accuracy:.4f}")
    for i, label in enumerate(LABELS):
        mask = targets == i
        if mask.sum():
            print(f"  {label:6} recall = {(predictions[mask] == i).mean():.4f} "
                  f"({mask.sum()} samples)")

    (MODEL_DIR / "cnn_history.json").write_text(
        json.dumps({
            "history": history, "best_val_acc": best_accuracy,
            "params": n_params, "quick": args.quick,
        }, indent=2),
        encoding="utf-8",
    )
    plot_curves(history, MODEL_DIR / "cnn_curves.png")


if __name__ == "__main__":
    main()
