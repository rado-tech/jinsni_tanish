"""Inspect the source dataset without downloading all of it.

Streams the first N rows of uzbekvoice-filtered and reports label balance,
speaker concentration and clip durations. Run this before 02_prepare.py to
sanity-check that the dataset schema is what the pipeline expects.

Usage:
    python scripts/01_explore.py [--rows 2000]

Outputs (data/):
    explore_report.json, duration_hist.png, raw/sample_{Ayol,Erkak}.wav
"""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "DavronSherbaev/uzbekvoice-filtered"
AUDIO_COLUMN = "path"        # the Audio feature lives under "path", not "audio"
GENDER_COLUMN = "gender"
SPEAKER_COLUMN = "client_id"
GENDERS = ("Ayol", "Erkak")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000, help="rows to stream")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)

    first = next(iter(dataset))
    print("Columns:", list(first.keys()))
    print("Sample rate:", first[AUDIO_COLUMN]["sampling_rate"], "Hz\n")

    gender_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    durations: list[float] = []
    saved_samples: dict[str, str] = {}

    rows = itertools.islice(dataset, args.rows)
    for row in tqdm(rows, total=args.rows, desc="Scanning"):
        waveform = row[AUDIO_COLUMN]["array"]
        sample_rate = row[AUDIO_COLUMN]["sampling_rate"]
        gender = row.get(GENDER_COLUMN) or "UNKNOWN"

        gender_counts[gender] += 1
        speaker_counts[row.get(SPEAKER_COLUMN) or "UNKNOWN"] += 1
        durations.append(len(waveform) / sample_rate)

        if gender in GENDERS and gender not in saved_samples:
            name = f"sample_{gender}.wav"
            sf.write(RAW_DIR / name, waveform, sample_rate)
            saved_samples[gender] = name

    durations.sort()
    report = {
        "rows_scanned": args.rows,
        "gender_counts": dict(gender_counts),
        "unique_speakers": len(speaker_counts),
        "top5_speakers_by_clips": [
            {"speaker": s[:8] + "...", "clips": c}
            for s, c in speaker_counts.most_common(5)
        ],
        "duration_stats": {
            "min_s": round(durations[0], 2),
            "median_s": round(durations[len(durations) // 2], 2),
            "mean_s": round(sum(durations) / len(durations), 2),
            "max_s": round(durations[-1], 2),
        },
        "saved_samples": saved_samples,
    }

    plt.figure(figsize=(8, 4))
    plt.hist(durations, bins=50)
    plt.xlabel("Duration (s)")
    plt.ylabel("Clips")
    plt.title(f"Clip durations (first {args.rows} rows)")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "duration_hist.png", dpi=120)

    (DATA_DIR / "explore_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
