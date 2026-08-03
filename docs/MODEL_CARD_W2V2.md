---
language:
  - uz
license: apache-2.0
library_name: transformers
tags:
  - audio
  - audio-classification
  - speaker-recognition
  - uzbek
  - wav2vec2
datasets:
  - DavronSherbaev/uzbekvoice-filtered
metrics:
  - accuracy
  - f1
base_model: facebook/wav2vec2-base
pipeline_tag: audio-classification
---

# Uzbek Speaker Gender Recognition (wav2vec2-base)

`facebook/wav2vec2-base` fine-tuned to predict speaker gender from Uzbek
speech. 94.6M parameters, 361 MB.

This is the **accuracy-first** variant. For real-time or resource-constrained
use see [rado-tech/uzbek-gender-cnn](https://huggingface.co/rado-tech/uzbek-gender-cnn),
which is 126× faster and 1480× smaller at a 0.65-point accuracy cost.

## Model details

- **Base model:** `facebook/wav2vec2-base`
- **Head:** `Wav2Vec2ForSequenceClassification`, 2 labels
- **Frozen:** the convolutional feature encoder (4.2M parameters). It encodes
  low-level acoustics that transfer unchanged, and freezing it cuts ~30% of
  step time.
- **Trainable:** 90.4M parameters
- **Input:** raw waveform, 2 seconds at 16 kHz (32,000 samples)
- **Output:** two logits; index 0 = `Ayol` (female), 1 = `Erkak` (male)
- **License:** Apache 2.0

## Training

| Setting | Value |
|---|---|
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 3e-5 backbone / 1e-3 classifier head |
| Schedule | linear with 10% warmup |
| Weight decay | 0.01 |
| Precision | mixed (fp16) |
| Hardware | one NVIDIA T4, ~15 minutes |

Discriminative learning rates matter here: a single large rate destroys the
pretrained representations, while a single small rate leaves the randomly
initialised head undertrained.

| Epoch | Train loss / acc | Val loss / acc |
|---|---|---|
| 1 | 0.0709 / 0.9795 | 0.0301 / 0.9920 |
| 2 | 0.0206 / 0.9953 | 0.0187 / 0.9963 |
| 3 | 0.0130 / 0.9972 | 0.0200 / 0.9967 |

Validation accuracy after a single epoch already exceeded what a comparable
from-scratch CNN reached in fifteen.

## Evaluation

Held-out test split: 3,049 clips from 364 speakers, disjoint from training and
validation.

| Metric | wav2vec2 | CNN baseline |
|---|---|---|
| Accuracy | **98.75%** | 98.10% |
| Errors | 38 | 58 |
| Precision (male) | 0.9993 | 0.9940 |
| Recall (male) | 0.9759 | 0.9681 |
| Recall (female) | 0.9993 | 0.9940 |
| F1 | 0.9875 | 0.9809 |
| CPU latency (2 s audio) | 504 ms | 4 ms |

McNemar's test against the CNN: χ² = 16.41 (critical 3.841), so the difference
is statistically significant. Notably the two models disagree asymmetrically —
wav2vec2 was right where the CNN was wrong 21 times, the reverse happened once.

37 of the 38 errors are shared with the CNN, and 34 of those trace to five
contributor accounts with demonstrably wrong labels — one contains recordings
from two different people. Excluding two such accounts the model scores 99.73%,
though the reportable figure remains 98.75%.

## Usage

```python
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

repo = "rado-tech/uzbek-gender-wav2vec2"
extractor = Wav2Vec2FeatureExtractor.from_pretrained(repo)
model = Wav2Vec2ForSequenceClassification.from_pretrained(repo).eval()

# waveform: mono float32 at 16 kHz, ideally a 2-second window
inputs = extractor([waveform], sampling_rate=16_000, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=1)

labels = ["Ayol", "Erkak"]   # female, male
print(labels[probs.argmax(1).item()], probs.max().item())
```

For longer recordings, split into overlapping 2-second windows, drop silent
ones, and average the probabilities — the reference implementation is in the
project repository.

## Intended use

Research and exploratory analysis of Uzbek speech.

**Out of scope.** Do not use this model to make decisions about individuals —
access control, identity verification, profiling, or content moderation. It
predicts an acoustic correlate of vocal tract physiology, not a person's gender
identity.

## Limitations and biases

- **Language.** Trained only on Uzbek; accuracy elsewhere is unverified.
- **Binary labels.** The source dataset labels gender as binary, so the model
  does too. This reflects the data, not the range of human gender identity.
- **Age distribution.** About 75% of training speakers are 18–24.
- **Latency.** 504 ms per 2-second window on CPU makes this unsuitable for
  real-time streaming; use the CNN variant for that.
- **Recording conditions.** Crowdsourced phone and laptop audio. Studio
  recordings and heavy background noise are out of distribution.

## Training data

[DavronSherbaev/uzbekvoice-filtered](https://huggingface.co/datasets/DavronSherbaev/uzbekvoice-filtered),
Apache 2.0. A 30,000-clip subset (34.8 hours) from 3,505 speakers, filtered on
community quality signals, capped at 40 clips per speaker, and split so no
speaker appears in more than one partition.

## Citation

```bibtex
@software{uzbek_gender_wav2vec2,
  title   = {Uzbek Speaker Gender Recognition (wav2vec2)},
  year    = {2026},
  note    = {facebook/wav2vec2-base fine-tuned on uzbekvoice-filtered},
  license = {Apache-2.0}
}
```
