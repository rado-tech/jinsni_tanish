"""Build and verify deployment bundles.

Two checks run before anything is copied, and both abort the build on failure:

1. **Feature parity** — the deployment path computes mel spectrograms with
   ``torch.stft`` instead of librosa. If the two ever diverge the model degrades
   silently, so their outputs are compared on real clips.
2. **Prediction parity** — the packaged pipeline must return the same
   probabilities as the training-time code on the same audio.

Usage:
    python scripts/09_export.py

Outputs (build/):
    spaces/    upload to a Hugging Face Space (Gradio SDK)
    railway/   push to Railway (or any Docker host)
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from genderid.config import CROP_FRAMES, MEL_FILTERS_FILE, N_FFT, N_MELS, SAMPLE_RATE
from genderid.features import fit_frames, log_mel, mel_filterbank
from genderid.model import GenderCNN

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
APP_DIR = ROOT / "app"
BUILD_DIR = ROOT / "build"

N_PARITY_CLIPS = 60
MEL_TOLERANCE_DB = 0.05
PROB_TOLERANCE = 1e-3

SPACES_LAYOUT = [
    (APP_DIR / "gradio_app.py", "app.py"),          # HF Spaces expects app.py
    (APP_DIR / "requirements-spaces.txt", "requirements.txt"),
    (APP_DIR / "hf_space_README.md", "README.md"),
]
RAILWAY_LAYOUT = [
    (APP_DIR / "__init__.py", "app/__init__.py"),
    (APP_DIR / "main.py", "app/main.py"),
    (APP_DIR / "gradio_app.py", "app/gradio_app.py"),
    (APP_DIR / "telegram_bot.py", "app/telegram_bot.py"),
    (APP_DIR / "requirements-railway.txt", "requirements.txt"),
    (APP_DIR / "Dockerfile", "Dockerfile"),
    (APP_DIR / "railway.json", "railway.json"),
]


def step(message: str) -> None:
    print(f"\n=== {message} ===")


def check_mel_parity(clips: list[Path]) -> None:
    """torch.stft mel vs librosa mel on real audio."""
    import librosa  # dev-only

    worst = 0.0
    for path in clips:
        waveform, _ = sf.read(path, dtype="float32")
        window = waveform[: SAMPLE_RATE * 2]
        if len(window) < SAMPLE_RATE * 2:
            window = np.pad(window, (0, SAMPLE_RATE * 2 - len(window)))

        reference = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=window, sr=SAMPLE_RATE, n_fft=N_FFT,
                hop_length=160, n_mels=N_MELS,
            ),
            ref=np.max,
        )[:, :CROP_FRAMES]
        ours = fit_frames(log_mel(window), CROP_FRAMES)
        worst = max(worst, float(np.abs(reference - ours).max()))

    print(f"   worst difference over {len(clips)} clips: {worst:.6f} dB")
    if worst >= MEL_TOLERANCE_DB:
        sys.exit("FEATURE PARITY FAILED - do not deploy this build.")
    print("   OK - librosa is not needed at inference time")


def check_prediction_parity(clips: list[Path]) -> None:
    """Packaged classifier vs a checkpoint loaded directly."""
    from genderid import GenderClassifier

    classifier = GenderClassifier(MODEL_DIR / "cnn_best.pt")
    state = torch.load(MODEL_DIR / "cnn_best.pt", map_location="cpu")
    reference = GenderCNN()
    reference.load_state_dict(state["state_dict"])
    reference.eval()
    mean = np.array(state["mean"], dtype=np.float32)[:, None]
    std = np.array(state["std"], dtype=np.float32)[:, None]

    worst = 0.0
    for path in clips:
        waveform, _ = sf.read(path, dtype="float32")
        window = waveform[: SAMPLE_RATE * 2]
        if len(window) < SAMPLE_RATE * 2:
            window = np.pad(window, (0, SAMPLE_RATE * 2 - len(window)))

        features = (fit_frames(log_mel(window), CROP_FRAMES) - mean) / std
        with torch.no_grad():
            x = torch.from_numpy(features).float()[None, None]
            expected = torch.softmax(reference(x), dim=1)[0, 1].item()
        worst = max(worst, abs(expected - classifier.predict_window(window)))

    print(f"   worst probability difference: {worst:.2e}")
    if worst >= PROB_TOLERANCE:
        sys.exit("PREDICTION PARITY FAILED - do not deploy this build.")
    print("   OK - packaged pipeline matches the training-time model")


def copy_bundle(name: str, layout: list[tuple[Path, str]]) -> None:
    """Assemble one bundle. The directory is reused, never deleted: it may hold
    a .git folder whose read-only objects break rmtree on Windows."""
    target = BUILD_DIR / name
    target.mkdir(parents=True, exist_ok=True)

    for source, relative in layout:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)

    package = target / "genderid"
    package.mkdir(exist_ok=True)
    for module in (ROOT / "genderid").glob("*.py"):
        shutil.copy(module, package / module.name)
    shutil.copy(MEL_FILTERS_FILE, package / MEL_FILTERS_FILE.name)

    weights = target / "models"
    weights.mkdir(exist_ok=True)
    shutil.copy(MODEL_DIR / "cnn_best.pt", weights / "cnn_best.pt")

    size_kb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1024
    print(f"   build/{name}: {size_kb:.0f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips", type=int, default=N_PARITY_CLIPS,
        help="clips used for the parity checks",
    )
    args = parser.parse_args()

    if not (MODEL_DIR / "cnn_best.pt").exists():
        sys.exit("models/cnn_best.pt not found - run scripts/04_train_cnn.py first.")

    step("1. Mel filterbank")
    filters = mel_filterbank()
    print(f"   {tuple(filters.shape)} -> {MEL_FILTERS_FILE.name} "
          f"({MEL_FILTERS_FILE.stat().st_size / 1024:.0f} KB)")

    metadata = pd.read_csv(PROCESSED_DIR / "metadata.csv")
    sample = metadata[metadata["split"] == "test"].sample(
        args.clips, random_state=0
    )
    clips = [PROCESSED_DIR / "audio" / name for name in sample["filename"]]

    step("2. Feature parity: torch.stft vs librosa")
    check_mel_parity(clips)

    step("3. Prediction parity: packaged pipeline vs checkpoint")
    check_prediction_parity(clips)

    step("4. ffmpeg availability (needed by the Telegram bot)")
    if shutil.which("ffmpeg"):
        from genderid import GenderClassifier

        result = GenderClassifier(MODEL_DIR / "cnn_best.pt").predict_file(clips[0])
        print(f"   OK - decoded a test clip: {result.label} (p={result.probability:.3f})")
    else:
        print("   ffmpeg not on PATH - the bot cannot be tested locally.")
        print("   The Docker image installs it, so deployment is unaffected.")

    step("5. Bundles")
    copy_bundle("spaces", SPACES_LAYOUT)
    copy_bundle("railway", RAILWAY_LAYOUT)

    print("\nDone. See docs/DEPLOYMENT.md for the upload steps.")


if __name__ == "__main__":
    main()
