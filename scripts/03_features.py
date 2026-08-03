"""Precompute log-mel spectrograms into a memory-mapped cache.

Recomputing features every epoch would waste ~3 minutes per pass. One cache of
(N, 64, 400) float16 costs ~1.4 GB on disk and is read lazily via np.memmap.

Normalisation statistics are computed over the **training split only** —
including val/test would leak information into the model through the feature
scaling.

Usage:
    python scripts/03_features.py

Outputs (data/processed/):
    mels.npy, mel_lengths.npy, norm_stats.json, mel_examples.png
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
import soundfile as sf
from tqdm import tqdm

from genderid.config import HOP_LENGTH, N_FFT, N_MELS, PAD_DB, SAMPLE_RATE
from genderid.features import fit_frames, log_mel

DEFAULT_MAX_FRAMES = 400  # 4 s; training crops 2 s out, longer clips are cut
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
AUDIO_DIR = PROCESSED_DIR / "audio"


def compute_train_stats(mels, lengths, train_rows) -> tuple[np.ndarray, np.ndarray]:
    """Per-mel-band mean and std over unpadded training frames.

    Accumulated in a streaming fashion so the full 1.2 GB never enters RAM.
    """
    total = np.zeros(N_MELS, dtype=np.float64)
    total_sq = np.zeros(N_MELS, dtype=np.float64)
    n_frames = 0

    for row in tqdm(train_rows, desc="Statistics"):
        frames = np.asarray(mels[row][:, : lengths[row]], dtype=np.float64)
        total += frames.sum(axis=1)
        total_sq += np.square(frames).sum(axis=1)
        n_frames += frames.shape[1]

    mean = total / n_frames
    variance = np.maximum(total_sq / n_frames - np.square(mean), 0.0)
    return mean, np.maximum(np.sqrt(variance), 1e-5)


def plot_examples(df, mels, lengths, mean, std, out_path: Path) -> None:
    """Side-by-side female/male spectrograms — harmonic spacing is visible."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    image = None
    for col, gender in enumerate(["Ayol", "Erkak"]):
        rows = df.index[(df["gender"] == gender) & (df["split"] == "train")][:2]
        for i, row in enumerate(rows):
            ax = axes[i, col]
            image = ax.imshow(
                mels[row][:, : lengths[row]].astype(np.float32),
                aspect="auto", origin="lower", cmap="magma", vmin=-80, vmax=0,
            )
            ax.set_title(f"{gender} - {df.loc[row, 'filename'][:14]}", fontsize=9)
            ax.set_ylabel("mel band")
            if i == 1:
                ax.set_xlabel("time (frames, 10 ms each)")
    fig.colorbar(image, ax=axes, label="dB", shrink=0.7)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help="frames cached per clip (400 = 4 s)",
    )
    args = parser.parse_args()
    max_frames = args.max_frames

    started = time.time()
    df = pd.read_csv(PROCESSED_DIR / "metadata.csv")
    n_clips = len(df)
    print(f"{n_clips} clips -> cache shape ({n_clips}, {N_MELS}, {max_frames})")

    cache_path = PROCESSED_DIR / "mels.npy"
    mels = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float16, shape=(n_clips, N_MELS, max_frames)
    )
    lengths = np.zeros(n_clips, dtype=np.int32)

    for i, filename in enumerate(tqdm(df["filename"], desc="Log-mel")):
        waveform, sample_rate = sf.read(AUDIO_DIR / filename, dtype="float32")
        assert sample_rate == SAMPLE_RATE, f"{filename}: unexpected {sample_rate} Hz"
        mel = log_mel(waveform)
        lengths[i] = min(mel.shape[1], max_frames)
        mels[i] = fit_frames(mel, max_frames).astype(np.float16)

    mels.flush()
    np.save(PROCESSED_DIR / "mel_lengths.npy", lengths)
    print(f"Cache written: {cache_path.stat().st_size / 1024 ** 3:.2f} GB")

    train_rows = df.index[df["split"] == "train"].to_numpy()
    mean, std = compute_train_stats(mels, lengths, train_rows)

    (PROCESSED_DIR / "norm_stats.json").write_text(
        json.dumps({
            "sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
            "n_mels": N_MELS, "max_frames": max_frames, "pad_db": PAD_DB,
            "mean": mean.tolist(), "std": std.tolist(),
        }, indent=2),
        encoding="utf-8",
    )

    sample = np.random.default_rng(0).choice(train_rows, 500, replace=False)
    normalised = np.concatenate(
        [(np.asarray(mels[i][:, : lengths[i]], dtype=np.float64) - mean[:, None])
         / std[:, None] for i in sample],
        axis=1,
    )
    print(f"Normalised train sample: mean={normalised.mean():+.4f} "
          f"std={normalised.std():.4f}  (expect ~0 and ~1)")

    plot_examples(df, mels, lengths, mean, std, PROCESSED_DIR / "mel_examples.png")
    print(f"Done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
