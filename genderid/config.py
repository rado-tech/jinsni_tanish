"""Shared constants.

These values are baked into the trained checkpoint. Changing any of the audio
or feature parameters invalidates existing models — retrain after editing.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# --- Audio -------------------------------------------------------------------
SAMPLE_RATE = 16_000

# --- Log-mel features --------------------------------------------------------
N_FFT = 512            # 32 ms analysis window
HOP_LENGTH = 160       # 10 ms hop -> 100 frames per second
N_MELS = 64
TOP_DB = 80.0          # dynamic range floor; also the padding value in dB space
PAD_DB = -TOP_DB

# --- Model input -------------------------------------------------------------
CROP_FRAMES = 200                       # 2 s at 10 ms per frame
WINDOW_SAMPLES = SAMPLE_RATE * 2        # 2 s of raw audio
HOP_SAMPLES = SAMPLE_RATE // 2          # 0.5 s stride between windows

# --- Decision thresholds -----------------------------------------------------
VAD_DB = -45.0                          # frames quieter than this count as silence
CONF_LOW = 0.35                         # p(male) below this -> female
CONF_HIGH = 0.65                        # p(male) above this -> male; between -> uncertain

# --- Labels ------------------------------------------------------------------
# Index order matters: it is the order the classifier head was trained with.
LABELS = ["Ayol", "Erkak"]              # ["female", "male"] in Uzbek
LABELS_EN = ["female", "male"]

# --- Limits ------------------------------------------------------------------
MAX_INPUT_SECONDS = 120                 # longer audio is truncated on decode

# --- Default paths -----------------------------------------------------------
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "cnn_best.pt"
MEL_FILTERS_FILE = PACKAGE_DIR / "mel_filters.npy"
