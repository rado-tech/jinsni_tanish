"""Build a balanced, quality-filtered, speaker-disjoint training set.

Streams uzbekvoice-filtered and writes WAV files plus a metadata CSV. Three
properties matter more than raw volume:

* **Quality gate** — applied to metadata before the audio is decoded, so bad
  clips cost nothing.
* **Per-speaker cap** — prolific contributors dominate the raw dataset (the top
  five accounted for 23% of a 2,000-row sample). Capping restores diversity.
* **Speaker-disjoint split** — every clip from one ``client_id`` lands in a
  single split. Without this the model memorises voices and test accuracy is
  meaningless.

Usage:
    python scripts/02_prepare.py [--per-gender 15000] [--max-per-speaker 40]

Outputs (data/processed/):
    audio/*.wav, metadata.csv, prepare_report.json
"""

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "DavronSherbaev/uzbekvoice-filtered"
AUDIO_COLUMN = "path"
GENDERS = ("Ayol", "Erkak")
MIN_DURATION_S, MAX_DURATION_S = 1.5, 10.0
TEST_FRACTION = VAL_FRACTION = 0.10
SEED = 42

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
AUDIO_DIR = PROCESSED_DIR / "audio"

CSV_FIELDS = [
    "filename", "gender", "client_id", "duration_s",
    "year_of_birth", "accent_region", "native_language", "split",
]


def is_good_clip(row: dict) -> bool:
    """Metadata-only quality gate; never touches the audio payload."""
    duration = row.get("duration") or 0.0
    return (
        row.get("gender") in GENDERS
        and MIN_DURATION_S <= duration <= MAX_DURATION_S
        and (row.get("reported_count") or 0) == 0
        and (row.get("downvotes_count") or 0) <= (row.get("upvotes_count") or 0)
    )


def should_take(gender, speaker, per_gender, per_speaker, quota, cap) -> bool:
    """Enforce the per-gender quota and the per-speaker cap."""
    return per_gender[gender] < quota and per_speaker[speaker] < cap


def find_conflicting_speakers(speaker_genders: dict[str, set[str]]) -> set[str]:
    """Speakers labelled with more than one gender.

    Which of their labels is correct is unknowable, so all their clips are
    dropped. Cheap insurance against label noise.
    """
    return {speaker for speaker, genders in speaker_genders.items() if len(genders) > 1}


def assign_splits(rows: list[dict], seed: int) -> dict[str, str]:
    """Map each speaker to train / val / test, keeping gender balance.

    Speakers are shuffled deterministically and greedily poured into the test
    bucket, then val, then train. Buckets overshoot their target by at most one
    speaker's worth of clips, which is negligible at dataset scale.
    """
    assignment: dict[str, str] = {}
    rng = random.Random(seed)

    for gender in GENDERS:
        counts: Counter[str] = Counter(
            row["client_id"] for row in rows if row["gender"] == gender
        )
        # sorted() before shuffle: dict ordering must not affect the outcome,
        # otherwise the seed would not guarantee a reproducible split.
        speakers = sorted(counts)
        rng.shuffle(speakers)

        total = sum(counts.values())
        test_target, val_target = total * TEST_FRACTION, total * VAL_FRACTION
        test_clips = val_clips = 0

        for speaker in speakers:
            if test_clips < test_target:
                assignment[speaker] = "test"
                test_clips += counts[speaker]
            elif val_clips < val_target:
                assignment[speaker] = "val"
                val_clips += counts[speaker]
            else:
                assignment[speaker] = "train"

    return assignment


def collect(quota: int, cap: int) -> tuple[list[dict], Counter]:
    """Stream the dataset until both gender quotas are filled."""
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    per_gender: Counter[str] = Counter()
    per_speaker: Counter[str] = Counter()
    speaker_genders: defaultdict[str, set] = defaultdict(set)
    rows: list[dict] = []
    stats: Counter[str] = Counter()

    progress = tqdm(total=quota * len(GENDERS), desc="Collecting")
    for row in dataset:
        if all(per_gender[g] >= quota for g in GENDERS):
            break
        stats["scanned"] += 1

        if not is_good_clip(row):
            stats["rejected_quality"] += 1
            continue

        gender, speaker = row["gender"], row["client_id"]
        if not should_take(gender, speaker, per_gender, per_speaker, quota, cap):
            stats["rejected_quota"] += 1
            continue

        waveform = row[AUDIO_COLUMN]["array"]
        sample_rate = row[AUDIO_COLUMN]["sampling_rate"]
        if len(waveform) == 0:
            stats["empty_audio"] += 1
            continue

        filename = f"{row['id']}.wav"
        sf.write(AUDIO_DIR / filename, waveform, sample_rate, subtype="PCM_16")
        rows.append({
            "filename": filename,
            "gender": gender,
            "client_id": speaker,
            "duration_s": round(len(waveform) / sample_rate, 2),
            "year_of_birth": row.get("year_of_birth") or "",
            "accent_region": row.get("accent_region") or "",
            "native_language": row.get("native_language") or "",
        })
        per_gender[gender] += 1
        per_speaker[speaker] += 1
        speaker_genders[speaker].add(gender)
        progress.update(1)
    progress.close()

    conflicting = find_conflicting_speakers(speaker_genders)
    stats["dropped_conflicting_labels"] = sum(
        1 for r in rows if r["client_id"] in conflicting
    )
    stats["conflicting_speakers"] = len(conflicting)
    return [r for r in rows if r["client_id"] not in conflicting], stats


def verify_no_leakage(rows: list[dict]) -> dict[str, set[str]]:
    """Assert that no speaker appears in more than one split."""
    speakers_by_split: defaultdict[str, set] = defaultdict(set)
    for row in rows:
        speakers_by_split[row["split"]].add(row["client_id"])

    for split in ("train", "val", "test"):
        assert speakers_by_split[split], f"split '{split}' is empty"
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = speakers_by_split[a] & speakers_by_split[b]
        assert not shared, f"leakage: {len(shared)} speakers shared by '{a}' and '{b}'"
    return speakers_by_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-gender", type=int, default=15000)
    parser.add_argument("--max-per-speaker", type=int, default=40)
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rows, stats = collect(args.per_gender, args.max_per_speaker)

    split_of = assign_splits(rows, SEED)
    for row in rows:
        row["split"] = split_of[row["client_id"]]
    speakers_by_split = verify_no_leakage(rows)

    with open(PROCESSED_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "total_clips": len(rows),
        "total_speakers": sum(len(v) for v in speakers_by_split.values()),
        "stream_stats": dict(stats),
        "splits": {
            split: {
                "clips": len(subset),
                "share_pct": round(100 * len(subset) / len(rows), 1),
                **{g: sum(1 for r in subset if r["gender"] == g) for g in GENDERS},
                "speakers": len(speakers_by_split[split]),
            }
            for split in ("train", "val", "test")
            for subset in [[r for r in rows if r["split"] == split]]
        },
        "seed": SEED,
        "minutes": round((time.time() - started) / 60, 1),
    }
    (PROCESSED_DIR / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nAll invariants passed - the split is speaker-disjoint.")


if __name__ == "__main__":
    main()
