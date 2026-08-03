"""Speaker gender identification from Uzbek speech.

Public API:
    from genderid import GenderClassifier, load_default
"""

from genderid.config import LABELS, SAMPLE_RATE
from genderid.inference import GenderClassifier, Prediction, load_default
from genderid.model import GenderCNN

__version__ = "1.0.0"
__all__ = [
    "GenderCNN",
    "GenderClassifier",
    "Prediction",
    "load_default",
    "LABELS",
    "SAMPLE_RATE",
]
