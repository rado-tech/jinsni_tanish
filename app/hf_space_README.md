---
title: Uzbek Speaker Gender Recognition
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# Uzbek Speaker Gender Recognition

Upload audio or speak into your microphone; the model predicts the speaker's
gender from voice.

| | |
|---|---|
| Architecture | log-mel spectrogram + 4-block CNN |
| Parameters | 60,706 |
| Test accuracy | 98.1% (speaker-disjoint split) |
| Latency | ~4 ms per 2 s window on CPU |
| Weights | [rado-tech/uzbek-gender-cnn](https://huggingface.co/rado-tech/uzbek-gender-cnn) |
| Training data | [uzbekvoice-filtered](https://huggingface.co/datasets/DavronSherbaev/uzbekvoice-filtered) |

A more accurate but much heavier alternative — 98.75% at 504 ms per window —
is available as [rado-tech/uzbek-gender-wav2vec2](https://huggingface.co/rado-tech/uzbek-gender-wav2vec2).
The small model is used here because real-time streaming needs the latency
headroom.

## How it works

1. Audio is converted to 16 kHz mono
2. Split into 2-second windows with a 0.5-second stride
3. Each window becomes a log-mel spectrogram (64 mel bands, 32 ms frames)
4. Energy-based voice activity detection drops silent windows
5. The CNN scores each remaining window; probabilities are averaged
6. If the averaged probability lands between 0.35 and 0.65, the answer is
   reported as *uncertain* rather than forced into a class

## Limitations

- Trained on Uzbek speech; accuracy on other languages is unverified
- Gender is modelled as binary because the source dataset is labelled that way.
  This reflects the dataset, not the range of human gender identity
- Energy-based VAD treats loud noise as speech
- Predictions are approximate and must not be used to make decisions about
  individuals

Source code and training pipeline: see the project repository.
