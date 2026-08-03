"""Re-download exactly the clips listed in an existing metadata.csv.

Used when training the transformer baseline on a machine that has a GPU but not
the prepared dataset (typically Colab). Re-running 02_prepare.py there could
produce a different selection; driving the download from metadata.csv keeps both
tracks on an identical split, which is what makes the later comparison valid.

Usage:
    python scripts/05_fetch_audio.py --metadata metadata.csv --out audio
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "DavronSherbaev/uzbekvoice-filtered"
AUDIO_COLUMN = "path"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("metadata.csv"))
    parser.add_argument("--out", type=Path, default=Path("audio"))
    args = parser.parse_args()

    if not args.metadata.exists():
        sys.exit(f"{args.metadata} not found - upload it alongside this script.")

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = {Path(f).stem: f for f in pd.read_csv(args.metadata)["filename"]}
    on_disk = {p.name for p in args.out.glob("*.wav")}
    todo = {k: v for k, v in wanted.items() if v not in on_disk}

    print(f"wanted={len(wanted)} on_disk={len(on_disk)} to_download={len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    started = time.time()
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    progress = tqdm(total=len(todo), desc="Downloading")
    scanned = 0

    for row in dataset:
        scanned += 1
        filename = todo.pop(str(row["id"]), None)
        if filename is None:
            continue
        sf.write(
            args.out / filename,
            row[AUDIO_COLUMN]["array"],
            row[AUDIO_COLUMN]["sampling_rate"],
            subtype="PCM_16",
        )
        progress.update(1)
        if not todo:
            break
    progress.close()

    fetched = len(list(args.out.glob("*.wav")))
    print(f"scanned {scanned} rows, {fetched}/{len(wanted)} files on disk, "
          f"{(time.time() - started) / 60:.1f} min")
    if todo:
        print(f"WARNING: {len(todo)} clips were not found - the upstream dataset "
              "may have changed since metadata.csv was generated.")


if __name__ == "__main__":
    main()
