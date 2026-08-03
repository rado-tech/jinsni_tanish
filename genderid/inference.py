"""End-to-end inference: audio in, gender prediction out."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from genderid.config import (
    CONF_HIGH,
    CONF_LOW,
    CROP_FRAMES,
    DEFAULT_CHECKPOINT,
    LABELS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
)
from genderid.features import (
    decode_audio,
    fit_frames,
    is_speech,
    log_mel,
    resample,
    sliding_windows,
)
from genderid.model import GenderCNN

NO_SPEECH = "no_speech"
TOO_SHORT = "too_short"
UNCERTAIN = "uncertain"


@dataclass
class Prediction:
    """Result of analysing one recording.

    Attributes:
        label: one of ``LABELS``, or ``UNCERTAIN`` / ``NO_SPEECH`` / ``TOO_SHORT``.
        probability: p(male) averaged over speech windows; ``None`` when no
            speech was found.
        confidence: probability of the reported class, ``None`` when undecided.
        n_windows: total windows analysed.
        n_speech: windows that passed voice activity detection.
        duration: input length in seconds.
        window_probabilities: per-window p(male), useful for debugging.
        window_speech: per-window VAD flags.
    """

    label: str
    probability: float | None
    confidence: float | None
    n_windows: int
    n_speech: int
    duration: float
    window_probabilities: list[float] = field(default_factory=list)
    window_speech: list[bool] = field(default_factory=list)

    @property
    def is_decided(self) -> bool:
        return self.label in LABELS


class GenderClassifier:
    """Loads a trained checkpoint and predicts speaker gender.

    Example:
        >>> from genderid import GenderClassifier
        >>> clf = GenderClassifier("models/cnn_best.pt")
        >>> clf.predict_file("sample.wav").label
        'Erkak'
    """

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, threads: int = 2):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}\n"
                "Download the released weights:  python scripts/00_get_model.py\n"
                "Or train your own:              python scripts/04_train_cnn.py"
            )
        torch.set_num_threads(threads)

        state = torch.load(checkpoint, map_location="cpu")
        self.model = GenderCNN()
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.labels = state.get("labels", LABELS)
        self.mean = np.asarray(state["mean"], dtype=np.float32)[:, None]
        self.std = np.asarray(state["std"], dtype=np.float32)[:, None]
        self.val_accuracy = state.get("val_acc")

    # --- feature helpers ----------------------------------------------------

    def _features(self, window: np.ndarray) -> np.ndarray:
        """One 2 s window -> normalised (n_mels, CROP_FRAMES) array."""
        mel = fit_frames(log_mel(window), CROP_FRAMES)
        return (mel - self.mean) / self.std

    @torch.no_grad()
    def _probabilities(self, feature_batch: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(feature_batch)).float().unsqueeze(1)
        return torch.softmax(self.model(x), dim=1)[:, 1].numpy()

    # --- public API ---------------------------------------------------------

    @torch.no_grad()
    def predict_window(self, window: np.ndarray, sr: int = SAMPLE_RATE) -> float:
        """p(male) for a single window. Used by the low-latency streaming path."""
        y = resample(window, sr)
        if len(y) < WINDOW_SAMPLES:
            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
        return float(self._probabilities([self._features(y[:WINDOW_SAMPLES])])[0])

    def predict_waveform(self, waveform: np.ndarray, sr: int) -> Prediction:
        """Analyse an in-memory waveform with overlapping windows."""
        y = resample(np.asarray(waveform, dtype=np.float32), sr)
        duration = len(y) / SAMPLE_RATE
        if len(y) < SAMPLE_RATE // 2:
            return Prediction(TOO_SHORT, None, None, 0, 0, duration)

        windows, flags = sliding_windows(y)
        probs = self._probabilities([self._features(w) for w in windows])
        return self._aggregate(probs, flags, duration)

    def predict_file(self, path: str | Path) -> Prediction:
        """Analyse any ffmpeg-readable audio file (wav, mp3, ogg/opus, m4a...)."""
        return self.predict_waveform(decode_audio(path), SAMPLE_RATE)

    # --- aggregation --------------------------------------------------------

    def _aggregate(
        self, probs: np.ndarray, flags: list[bool], duration: float
    ) -> Prediction:
        """Average p(male) over speech windows and apply the decision band.

        Averaging probabilities beats majority voting: a window that reports
        0.51 should not carry the same weight as one reporting 0.99.
        """
        probs = np.asarray(probs, dtype=float)
        mask = np.asarray(flags, dtype=bool)
        speech = probs[mask]

        if speech.size == 0:
            return Prediction(
                NO_SPEECH, None, None, len(probs), 0, duration,
                probs.tolist(), list(flags),
            )

        p = float(speech.mean())
        if p < CONF_LOW:
            label, confidence = self.labels[0], 1.0 - p
        elif p > CONF_HIGH:
            label, confidence = self.labels[1], p
        else:
            label, confidence = UNCERTAIN, None

        return Prediction(
            label, p, confidence, len(probs), int(mask.sum()), duration,
            probs.tolist(), list(flags),
        )


_DEFAULT: GenderClassifier | None = None


def load_default(checkpoint: str | Path = DEFAULT_CHECKPOINT) -> GenderClassifier:
    """Process-wide singleton, so apps load the weights only once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GenderClassifier(checkpoint)
    return _DEFAULT
