"""Score the test split with the fine-tuned wav2vec2 model.

Normally run on the GPU machine that produced the checkpoint; the resulting CSV
(~200 KB) is small enough to move back to the workstation, where 08_compare.py
reads it. That keeps the comparison honest without shipping a 360 MB model.

Usage:
    python scripts/07_predict_w2v2.py [--model models/w2v2_best]

Outputs:
    models/w2v2_test_preds.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

SAMPLE_RATE = 16_000
CROP_SAMPLES = SAMPLE_RATE * 2
LABELS = ["Ayol", "Erkak"]

CWD = Path.cwd()
AUDIO_DIR = CWD / "audio" if (CWD / "audio").exists() else CWD / "data/processed/audio"
METADATA = (
    CWD / "metadata.csv" if (CWD / "metadata.csv").exists()
    else CWD / "data/processed/metadata.csv"
)


class TestSet(Dataset):
    def __init__(self, df):
        self.files = df["filename"].to_numpy()
        self.labels = df["gender"].map({g: i for i, g in enumerate(LABELS)}).to_numpy()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        waveform, _ = sf.read(AUDIO_DIR / self.files[index], dtype="float32")
        start = (
            0 if len(waveform) <= CROP_SAMPLES
            else (len(waveform) - CROP_SAMPLES) // 2   # centre crop, deterministic
        )
        crop = waveform[start:start + CROP_SAMPLES]
        if len(crop) < CROP_SAMPLES:
            crop = np.pad(crop, (0, CROP_SAMPLES - len(crop)))
        return crop.astype(np.float32), int(self.labels[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CWD / "models" / "w2v2_best")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--out", type=Path, default=CWD / "models" / "w2v2_test_preds.csv"
    )
    args = parser.parse_args()

    if not args.model.exists():
        sys.exit(f"{args.model} not found - train it with 06_finetune_w2v2.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = pd.read_csv(METADATA).query("split == 'test'").reset_index(drop=True)
    print(f"device={device} clips={len(test)}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model = (
        Wav2Vec2ForSequenceClassification.from_pretrained(args.model).to(device).eval()
    )

    def collate(batch):
        waveforms = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = feature_extractor(
            waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        return encoded.input_values, labels

    dataset = TestSet(test)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        collate_fn=collate,
    )

    predictions, probabilities = [], []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Predicting"):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(input_values=x.to(device)).logits
            probs = torch.softmax(logits.float(), dim=1)
            predictions.append(probs.argmax(dim=1).cpu().numpy())
            probabilities.append(probs[:, 1].cpu().numpy())

    out = pd.DataFrame({
        "filename": test["filename"],
        "y_true": dataset.labels,
        "y_pred": np.concatenate(predictions),
        "prob_male": np.concatenate(probabilities).round(6),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    accuracy = (out.y_true == out.y_pred).mean()
    errors = int((out.y_true != out.y_pred).sum())
    print(f"\nTest accuracy: {accuracy:.4f} ({errors} errors / {len(out)})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
