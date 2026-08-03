---
language:
  - uz
license: apache-2.0
library_name: pytorch
tags:
  - audio
  - audio-classification
  - speaker-recognition
  - uzbek
  - cnn
datasets:
  - DavronSherbaev/uzbekvoice-filtered
metrics:
  - accuracy
  - f1
pipeline_tag: audio-classification
---

# Uzbek Speaker Gender Recognition (CNN)

A compact convolutional network that predicts speaker gender from Uzbek speech.
60,706 parameters, 0.2 MB on disk, ~4 ms per 2-second window on a laptop CPU.

## Model details

- **Architecture:** four `Conv2d → BatchNorm2d → ReLU → MaxPool2d` blocks
  (1→16→32→64→64 channels), global average pooling, dropout 0.3, linear head
- **Input:** log-mel spectrogram, 64 mel bands × 200 frames (2 s at 16 kHz),
  normalised with per-band statistics computed on the training split
- **Output:** two logits; index 0 = `Ayol` (female), 1 = `Erkak` (male)
- **Training:** AdamW, lr 3e-3, cosine schedule, weight decay 1e-4, 15 epochs,
  batch size 64, random 2-second crops as augmentation
- **License:** Apache 2.0

## Intended use

Exploratory and research use on Uzbek speech: dataset analysis, demos, and as a
lightweight baseline for voice-attribute classification.

**Out of scope.** Do not use this model to make decisions about individuals —
access control, identity verification, profiling, content moderation, or any
setting where a person is affected by the output. It predicts an acoustic
correlate of vocal tract physiology, not a person's gender identity.

## Evaluation

Held-out test split: 3,049 clips from 364 speakers disjoint from training and
validation.

| Metric | Value |
|---|---|
| Accuracy | 98.10% |
| Precision (male) | 0.9940 |
| Recall (male) | 0.9681 |
| Recall (female) | 0.9940 |
| F1 | 0.9809 |

A fine-tuned wav2vec2-base baseline reaches 98.75% on the same split at 126×
the latency and 1480× the size.

About 30 of the 58 test errors trace to two contributor accounts with
demonstrably wrong labels — one contains recordings from two different people.
Excluding them the model scores 99.07%, though the reportable figure remains
98.10%.

## Limitations and biases

- **Language.** Trained only on Uzbek. Accuracy elsewhere is unverified, though
  the acoustic cues (fundamental frequency, formants) are largely
  language-independent.
- **Binary labels.** The source dataset labels gender as binary, so the model
  does too. This reflects the data, not the range of human gender identity.
  Voices that do not fit either training class will be misclassified or land in
  the uncertain band.
- **Age distribution.** About 75% of training speakers are 18–24. Accuracy was
  measured across age brackets with no significant effect, but coverage of
  older and younger voices is thin.
- **Regional variation.** Namangan-accented speech is the weakest region
  (96.6%) while the transformer baseline handles it perfectly, suggesting the
  small model is more accent-sensitive.
- **Recording conditions.** Trained on crowdsourced phone and laptop
  recordings. Studio audio or heavy background noise are out of distribution.
- **VAD.** The bundled voice-activity detector is energy-based and treats loud
  noise as speech.

## Usage

```python
from genderid import GenderClassifier

clf = GenderClassifier("cnn_best.pt")
result = clf.predict_file("sample.wav")

print(result.label)        # 'Ayol' | 'Erkak' | 'uncertain' | 'no_speech'
print(result.confidence)   # None when uncertain
print(result.probability)  # p(male), averaged over speech windows
```

Long recordings are split into overlapping 2-second windows; silent windows are
dropped and the remaining probabilities averaged. When the average falls between
0.35 and 0.65 the model reports `uncertain` instead of guessing.

Accepts any format ffmpeg can read. For in-memory audio use
`predict_waveform(array, sample_rate)`.

## Training data

[DavronSherbaev/uzbekvoice-filtered](https://huggingface.co/datasets/DavronSherbaev/uzbekvoice-filtered),
Apache 2.0. A 30,000-clip subset (34.8 hours) from 3,505 speakers, filtered on
community quality signals, capped at 40 clips per speaker, and split so that no
speaker appears in more than one partition.

## Citation

```bibtex
@software{uzbek_gender_cnn,
  title  = {Uzbek Speaker Gender Recognition},
  year   = {2026},
  note   = {CNN trained on uzbekvoice-filtered},
  license = {Apache-2.0}
}
```
