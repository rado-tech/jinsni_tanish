"""Download the pretrained checkpoint from the Hugging Face Hub.

Model weights are not committed to git. Run this once after cloning if you want
to use the model without retraining it. To train instead, follow steps 02-04.

Usage:
    python scripts/00_get_model.py [--repo <user>/uzbek-gender-cnn]

Outputs:
    models/cnn_best.pt
"""

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_REPO = "rado-tech/uzbek-gender-cnn"
CHECKPOINT_NAME = "cnn_best.pt"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face model repo")
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args()

    target = MODEL_DIR / CHECKPOINT_NAME
    if target.exists() and not args.force:
        size_kb = target.stat().st_size / 1024
        print(f"{target} already exists ({size_kb:.0f} KB). Use --force to replace.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed.\n"
            "  pip install huggingface_hub"
        )

    print(f"Downloading {CHECKPOINT_NAME} from {args.repo} ...")
    try:
        cached = hf_hub_download(repo_id=args.repo, filename=CHECKPOINT_NAME)
    except Exception as error:
        sys.exit(
            f"Download failed: {error}\n\n"
            "If the repository does not exist yet, train the model instead:\n"
            "  python scripts/02_prepare.py\n"
            "  python scripts/03_features.py\n"
            "  python scripts/04_train_cnn.py"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached, target)
    print(f"Saved {target} ({target.stat().st_size / 1024:.0f} KB)")

    from genderid import GenderClassifier

    classifier = GenderClassifier(target)
    print(f"Loaded OK - validation accuracy {classifier.val_accuracy:.4f}")


if __name__ == "__main__":
    main()
