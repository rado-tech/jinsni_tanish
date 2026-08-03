"""Audio decoding, log-mel features, voice activity detection and windowing.

The mel spectrogram is computed with ``torch.stft`` against a precomputed
filterbank rather than with librosa, so deployments do not need librosa/scipy/
numba. Both paths agree to within 1e-3 dB; ``scripts/09_export.py`` asserts this
before building a deployment bundle.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

from genderid.config import (
    HOP_LENGTH,
    HOP_SAMPLES,
    MAX_INPUT_SECONDS,
    MEL_FILTERS_FILE,
    N_FFT,
    N_MELS,
    PAD_DB,
    SAMPLE_RATE,
    TOP_DB,
    VAD_DB,
    WINDOW_SAMPLES,
)

_MEL_FILTERS: torch.Tensor | None = None
_WINDOW = torch.hann_window(N_FFT)


def mel_filterbank() -> torch.Tensor:
    """Return the (n_mels, n_fft // 2 + 1) mel filterbank.

    Loaded from the file bundled with the package. If it is missing (fresh
    checkout), it is computed with librosa and cached to disk.
    """
    global _MEL_FILTERS
    if _MEL_FILTERS is not None:
        return _MEL_FILTERS

    if MEL_FILTERS_FILE.exists():
        filters = np.load(MEL_FILTERS_FILE)
    else:
        import librosa  # dev-only dependency

        filters = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS
        ).astype(np.float32)
        MEL_FILTERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(MEL_FILTERS_FILE, filters)

    _MEL_FILTERS = torch.from_numpy(np.asarray(filters, dtype=np.float32))
    return _MEL_FILTERS


def decode_audio(path: str | Path, max_seconds: int = MAX_INPUT_SECONDS) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 at ``SAMPLE_RATE``.

    One ffmpeg call handles container parsing, downmixing, resampling and
    truncation. Required for Telegram voice notes, which arrive as OGG/Opus —
    a format soundfile cannot read.

    Raises:
        FileNotFoundError: ffmpeg is not on PATH.
        subprocess.CalledProcessError: the input could not be decoded.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-t", str(max_seconds),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def resample(y: np.ndarray, sr: int) -> np.ndarray:
    """Resample to ``SAMPLE_RATE`` using torchaudio's sinc interpolation.

    Browser microphones typically deliver 44.1 or 48 kHz. Naive decimation would
    alias energy above 8 kHz back into the speech band.
    """
    y = np.asarray(y, dtype=np.float32)
    if sr == SAMPLE_RATE:
        return y
    import torchaudio.functional as AF

    return AF.resample(torch.from_numpy(y), sr, SAMPLE_RATE).numpy()


def log_mel(waveform: np.ndarray) -> np.ndarray:
    """Waveform -> log-mel spectrogram in dB, shape (n_mels, n_frames).

    Matches ``librosa.power_to_db(melspectrogram(...), ref=np.max)``: the loudest
    bin of each clip becomes 0 dB, which normalises away recording gain.
    """
    spec = torch.stft(
        torch.from_numpy(np.asarray(waveform, dtype=np.float32)),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=_WINDOW,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    power = mel_filterbank() @ (spec.abs() ** 2)
    db = 10.0 * torch.log10(torch.clamp(power, min=1e-10))
    return torch.clamp(db - db.max(), min=-TOP_DB).numpy()


def fit_frames(mel: np.ndarray, n_frames: int) -> np.ndarray:
    """Trim or right-pad a spectrogram to exactly ``n_frames`` columns.

    Padding uses ``PAD_DB`` (silence), not zero — in this dB scale zero means
    *maximum loudness*.
    """
    if mel.shape[1] >= n_frames:
        return mel[:, :n_frames]
    pad = n_frames - mel.shape[1]
    return np.pad(mel, ((0, 0), (0, pad)), constant_values=PAD_DB)


def is_speech(waveform: np.ndarray, threshold_db: float = VAD_DB) -> bool:
    """Energy-based voice activity detection.

    Adequate in quiet conditions and dependency-free. In noisy environments a
    neural VAD (e.g. Silero) is the better choice: loud noise passes this test.
    """
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    return 20.0 * np.log10(rms + 1e-10) > threshold_db


def sliding_windows(waveform: np.ndarray) -> tuple[list[np.ndarray], list[bool]]:
    """Split audio into overlapping 2 s windows.

    Returns:
        (windows, speech_flags) — always at least one window; short input is
        zero-padded (raw audio, so zero is silence here).
    """
    y = np.asarray(waveform, dtype=np.float32)
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))

    windows, flags = [], []
    for start in range(0, len(y), HOP_SAMPLES):
        segment = y[start:start + WINDOW_SAMPLES]
        if len(segment) < WINDOW_SAMPLES:
            segment = np.pad(segment, (0, WINDOW_SAMPLES - len(segment)))
        windows.append(segment)
        flags.append(is_speech(segment))
        if start + WINDOW_SAMPLES >= len(y):
            break
    return windows, flags
