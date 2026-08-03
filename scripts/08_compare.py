"""Evaluate both models on the held-out test split.

The test split has not influenced any decision up to this point — learning rate,
epoch count and checkpoint selection were all driven by validation. Do not tune
anything in response to these numbers, or the test set becomes a second
validation set.

Beyond headline accuracy this reports:

* a confusion matrix and per-class recall;
* McNemar's test, the correct paired comparison when two models are scored on
  the same examples — it counts only the cases where they disagree;
* accuracy sliced by clip duration, age bracket and region, because an average
  can hide a badly served subgroup;
* CPU latency and model size, which decide what can run in real time.

Usage:
    python scripts/08_compare.py

Outputs (models/):
    comparison.json, comparison.png
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

from genderid.config import CROP_FRAMES, LABELS, N_MELS, PAD_DB
from genderid.model import GenderCNN

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MIN_SLICE_SIZE = 30
CHI2_CRITICAL = 3.841  # p = 0.05, one degree of freedom


@torch.no_grad()
def predict_cnn(checkpoint_path, rows, mels, lengths):
    """Centre-crop every test clip and score it in a single batch."""
    state = torch.load(checkpoint_path, map_location="cpu")
    mean = np.array(state["mean"], dtype=np.float32)[:, None]
    std = np.array(state["std"], dtype=np.float32)[:, None]

    model = GenderCNN()
    model.load_state_dict(state["state_dict"])
    model.eval()

    crops = []
    for row in rows:
        length = lengths[row]
        mel = np.asarray(mels[row], dtype=np.float32)
        start = 0 if length <= CROP_FRAMES else (length - CROP_FRAMES) // 2
        crop = mel[:, start:start + CROP_FRAMES]
        if crop.shape[1] < CROP_FRAMES:
            crop = np.pad(
                crop, ((0, 0), (0, CROP_FRAMES - crop.shape[1])), constant_values=PAD_DB
            )
        crops.append((crop - mean) / std)

    x = torch.from_numpy(np.stack(crops)).float().unsqueeze(1)
    probs = torch.softmax(model(x), dim=1)
    return probs.argmax(dim=1).numpy(), probs[:, 1].numpy()


def compute_metrics(y_true, y_pred) -> dict:
    """Confusion matrix and derived metrics, with male (index 1) as positive."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "recall_female": tn / (tn + fp) if tn + fp else 0.0,
        "recall_male": recall,
    }


def slice_report(groups, y_true, y_pred, min_size=MIN_SLICE_SIZE) -> list[dict]:
    """Accuracy per group, worst first. Groups below min_size are dropped."""
    correct = y_true == y_pred
    values = np.asarray(groups)
    rows = []
    for group in pd.unique(values):
        if pd.isna(group):
            continue
        mask = values == group
        if mask.sum() < min_size:
            continue
        rows.append({
            "group": str(group),
            "n": int(mask.sum()),
            "accuracy": round(float(correct[mask].mean()), 4),
            "errors": int((~correct[mask]).sum()),
        })
    return sorted(rows, key=lambda r: r["accuracy"])


def benchmark_latency(fn, x, n_warmup=5, n_runs=30) -> tuple[float, float]:
    """Median and p90 latency in milliseconds.

    Warm-up calls are discarded (allocation and cache effects make the first
    call an order of magnitude slower), and the median is reported rather than
    the mean so a single scheduling hiccup cannot skew the result.
    """
    for _ in range(n_warmup):
        fn(x)
    timings = []
    for _ in range(n_runs):
        started = time.perf_counter()
        fn(x)
        timings.append(time.perf_counter() - started)
    return float(np.median(timings) * 1000), float(np.percentile(timings, 90) * 1000)


def mcnemar(correct_a, correct_b) -> tuple[int, int, float, bool]:
    """Paired significance test. Returns (only_a, only_b, chi2, significant)."""
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    if only_a + only_b == 0:
        return only_a, only_b, 0.0, False
    chi2 = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b)
    return only_a, only_b, chi2, chi2 > CHI2_CRITICAL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for the benchmark")
    parser.add_argument("--min-slice", type=int, default=MIN_SLICE_SIZE)
    args = parser.parse_args()

    df = pd.read_csv(PROCESSED_DIR / "metadata.csv")
    test = df[df["split"] == "test"]
    y_true = test["gender"].map({g: i for i, g in enumerate(LABELS)}).to_numpy()
    print(f"Test split: {len(test)} clips, {test['client_id'].nunique()} speakers\n")

    mels = np.load(PROCESSED_DIR / "mels.npy", mmap_mode="r")
    lengths = np.load(PROCESSED_DIR / "mel_lengths.npy")

    results = {}
    cnn_pred, _ = predict_cnn(
        MODEL_DIR / "cnn_best.pt", test.index.to_numpy(), mels, lengths
    )
    results["cnn"] = compute_metrics(y_true, cnn_pred)

    preds_path = MODEL_DIR / "w2v2_test_preds.csv"
    have_w2v2 = preds_path.exists()
    w2v2_pred = None
    if have_w2v2:
        stored = pd.read_csv(preds_path).set_index("filename").loc[test["filename"]]
        assert (stored["y_true"].to_numpy() == y_true).all(), (
            "labels in w2v2_test_preds.csv do not match metadata.csv — "
            "the two models were evaluated on different splits"
        )
        w2v2_pred = stored["y_pred"].to_numpy()
        results["w2v2"] = compute_metrics(y_true, w2v2_pred)
    else:
        print(f"{preds_path} missing - reporting the CNN only.\n")

    models = [("cnn", "CNN"), ("w2v2", "wav2vec2")]
    print(f"{'Model':10} {'Accuracy':>9} {'Error':>8} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>7}")
    for key, name in models:
        if key not in results:
            continue
        m = results[key]
        print(f"{name:10} {m['accuracy']:9.4f} {100 * (1 - m['accuracy']):7.2f}% "
              f"{m['precision']:10.4f} {m['recall']:8.4f} {m['f1']:7.4f}")

    print(f"\n{'Model':10} {'TN':>6} {'FP':>5} {'FN':>5} {'TP':>6} "
          f"{'R_female':>9} {'R_male':>8}")
    for key, name in models:
        if key not in results:
            continue
        m = results[key]
        print(f"{name:10} {m['tn']:6} {m['fp']:5} {m['fn']:5} {m['tp']:6} "
              f"{m['recall_female']:9.4f} {m['recall_male']:8.4f}")

    if have_w2v2:
        only_cnn, only_w2v2, chi2, significant = mcnemar(
            y_true == cnn_pred, y_true == w2v2_pred
        )
        print(f"\nMcNemar: CNN-only correct={only_cnn}, wav2vec2-only correct="
              f"{only_w2v2}, chi2={chi2:.2f} -> "
              f"{'significant' if significant else 'not significant'} "
              f"(critical {CHI2_CRITICAL})")
        results["mcnemar"] = {
            "only_cnn": only_cnn, "only_w2v2": only_w2v2,
            "chi2": round(chi2, 3), "significant": bool(significant),
        }

    duration_bins = pd.cut(
        test["duration_s"], [0, 2, 3, 4, 5, 100],
        labels=["<2s", "2-3s", "3-4s", "4-5s", ">5s"],
    )
    slices = {
        "duration": duration_bins,
        "age": test["year_of_birth"],
        "region": test["accent_region"],
    }
    results["slices"] = {}
    for name, groups in slices.items():
        print(f"\n--- Slice: {name} (worst 5) ---")
        print(f"    {'group':20} {'n':>5} {'CNN':>8} {'w2v2':>8}")
        cnn_slice = slice_report(groups, y_true, cnn_pred, args.min_slice)
        w2v2_slice = (
            {r["group"]: r["accuracy"]
             for r in slice_report(groups, y_true, w2v2_pred, args.min_slice)}
            if have_w2v2 else {}
        )
        for row in cnn_slice[:5]:
            other = w2v2_slice.get(row["group"])
            print(f"    {row['group']:20} {row['n']:5} {row['accuracy']:8.4f} "
                  f"{other:8.4f}" if other is not None else
                  f"    {row['group']:20} {row['n']:5} {row['accuracy']:8.4f} {'-':>8}")
        results["slices"][name] = cnn_slice

    print("\n--- CPU latency (2 s audio, batch size 1) ---")
    torch.set_num_threads(args.threads)
    state = torch.load(MODEL_DIR / "cnn_best.pt", map_location="cpu")
    cnn = GenderCNN()
    cnn.load_state_dict(state["state_dict"])
    cnn.eval()

    with torch.no_grad():
        median, p90 = benchmark_latency(cnn, torch.randn(1, 1, N_MELS, CROP_FRAMES))
    cnn_mb = (MODEL_DIR / "cnn_best.pt").stat().st_size / 1024 ** 2
    cnn_params = sum(p.numel() for p in cnn.parameters())
    results["speed"] = {
        "cnn_median_ms": round(median, 2), "cnn_p90_ms": round(p90, 2),
        "cnn_mb": round(cnn_mb, 1), "cnn_params": cnn_params,
    }
    print(f"  CNN      : {median:7.2f} ms (p90 {p90:.2f}) | {cnn_mb:6.1f} MB | "
          f"{cnn_params:,} params")

    w2v2_dir = MODEL_DIR / "w2v2_best"
    if w2v2_dir.exists():
        from transformers import Wav2Vec2ForSequenceClassification

        w2v2 = Wav2Vec2ForSequenceClassification.from_pretrained(w2v2_dir).eval()
        with torch.no_grad():
            median2, p902 = benchmark_latency(
                lambda t: w2v2(input_values=t).logits, torch.randn(1, 32000),
                n_warmup=3, n_runs=15,
            )
        size_mb = sum(
            f.stat().st_size for f in w2v2_dir.rglob("*") if f.is_file()
        ) / 1024 ** 2
        params = sum(p.numel() for p in w2v2.parameters())
        results["speed"].update({
            "w2v2_median_ms": round(median2, 2), "w2v2_p90_ms": round(p902, 2),
            "w2v2_mb": round(size_mb, 1), "w2v2_params": params,
        })
        print(f"  wav2vec2 : {median2:7.2f} ms (p90 {p902:.2f}) | {size_mb:6.1f} MB | "
              f"{params:,} params")
        print(f"  -> CNN is {median2 / median:.0f}x faster, "
              f"{size_mb / cnn_mb:.0f}x smaller")
    else:
        print(f"  wav2vec2 : skipped ({w2v2_dir} not present)")

    (MODEL_DIR / "comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = [n for k, n in models if k in results]
    errors = [100 * (1 - results[k]["accuracy"]) for k, _ in models if k in results]
    axes[0].bar(names, errors, color=["#4C78A8", "#F58518"][: len(names)])
    axes[0].set_ylabel("Test error (%)")
    axes[0].set_title("Lower is better")
    for i, value in enumerate(errors):
        axes[0].text(i, value, f"{value:.2f}%", ha="center", va="bottom")

    duration_slice = results["slices"]["duration"]
    axes[1].bar(
        [r["group"] for r in duration_slice],
        [r["accuracy"] for r in duration_slice],
        color="#4C78A8",
    )
    axes[1].set_ylim(
        min(0.9, min(r["accuracy"] for r in duration_slice) - 0.02), 1.005
    )
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("CNN accuracy by clip duration")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "comparison.png", dpi=120)
    print(f"\nWrote {MODEL_DIR / 'comparison.json'} and comparison.png")


if __name__ == "__main__":
    main()
