"""Download pretrained weights from the Hugging Face Hub.

Checkpoints are not committed to git. Run this once after cloning to use the
models without retraining. To train instead, follow steps 02-04 (CNN) and
05-06 (wav2vec2).

Usage:
    python scripts/00_get_model.py              # CNN only (0.2 MB)
    python scripts/00_get_model.py --w2v2       # also the transformer (361 MB)

Outputs:
    models/cnn_best.pt
    models/w2v2_best/           (with --w2v2)
"""

import argparse
import shutil
import sys
from pathlib import Path

CNN_REPO = "rado-tech/uzbek-gender-cnn"
W2V2_REPO = "rado-tech/uzbek-gender-wav2vec2"
CHECKPOINT_NAME = "cnn_best.pt"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

TRAIN_INSTEAD = (
    "\nTo train the model yourself instead:\n"
    "  python scripts/02_prepare.py\n"
    "  python scripts/03_features.py\n"
    "  python scripts/04_train_cnn.py"
)


def _require_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is not installed.\n  pip install huggingface_hub")
    return hf_hub_download, snapshot_download


def fetch_cnn(repo: str, force: bool) -> None:
    target = MODEL_DIR / CHECKPOINT_NAME
    if target.exists() and not force:
        print(f"{target} already exists ({target.stat().st_size / 1024:.0f} KB). "
              "Use --force to replace.")
        return

    hf_hub_download, _ = _require_hub()
    print(f"Downloading {CHECKPOINT_NAME} from {repo} ...")
    try:
        cached = hf_hub_download(repo_id=repo, filename=CHECKPOINT_NAME)
    except Exception as error:
        sys.exit(f"Download failed: {error}\n{TRAIN_INSTEAD}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached, target)

    from genderid import GenderClassifier

    classifier = GenderClassifier(target)
    print(f"Saved {target} ({target.stat().st_size / 1024:.0f} KB) - "
          f"validation accuracy {classifier.val_accuracy:.4f}")


def fetch_w2v2(repo: str, force: bool) -> None:
    """Fetch the transformer baseline.

    Only needed to reproduce the latency comparison in 08_compare.py; the
    deployed application does not use it.
    """
    target = MODEL_DIR / "w2v2_best"
    if target.exists() and not force:
        print(f"{target} already exists. Use --force to replace.")
        return

    _, snapshot_download = _require_hub()
    print(f"Downloading {repo} (~361 MB) ...")
    try:
        cached = snapshot_download(repo_id=repo)
    except Exception as error:
        sys.exit(f"Download failed: {error}")

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(cached, target, ignore=shutil.ignore_patterns(".*"))
    size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024 ** 2
    print(f"Saved {target} ({size_mb:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=CNN_REPO, help="Hugging Face repo for the CNN")
    parser.add_argument("--w2v2", action="store_true", help="also fetch the transformer")
    parser.add_argument("--w2v2-repo", default=W2V2_REPO)
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args()

    fetch_cnn(args.repo, args.force)
    if args.w2v2:
        fetch_w2v2(args.w2v2_repo, args.force)


if __name__ == "__main__":
    main()
