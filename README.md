# Uzbek Speaker Gender Recognition

Identify a speaker's gender from their voice — from an uploaded file or a live
microphone stream. Trained on Uzbek speech.

A 60,706-parameter CNN reaches **98.1%** accuracy on a speaker-disjoint test
split and runs in **4 ms** on a laptop CPU. A fine-tuned wav2vec2 baseline
(94.6M parameters) reaches 98.8% but needs 504 ms per prediction — the
comparison, and what it implies for deployment, is the point of this repository.

*Uzbek: [README.uz.md](README.uz.md)*

---

## Results

Held-out test split — 3,049 clips from 364 speakers, none of whom appear in
training or validation.

| | CNN (from scratch) | wav2vec2-base (fine-tuned) |
|---|---|---|
| Accuracy | 98.10% | **98.75%** |
| Errors | 58 | 38 |
| Precision / Recall / F1 | 0.994 / 0.968 / 0.981 | 0.999 / 0.976 / 0.988 |
| CPU latency, 2 s audio | **4.0 ms** | 504 ms |
| Model size | **0.2 MB** | 361 MB |
| Parameters | **60,706** | 94,569,090 |
| Training | 55 min, 4 CPU cores | 15 min, one T4 GPU |

McNemar's test: χ² = 16.41 (critical value 3.84), so the accuracy difference is
statistically real — but the CNN is 126× faster and 1480× smaller, which is why
it is the one deployed for real-time use.

Full analysis, including error slices and a data-quality finding, is in
[docs/RESULTS.md](docs/RESULTS.md).

## How it works

```
audio ──> 16 kHz mono ──> 2 s windows (0.5 s stride) ──> log-mel (64 × 200)
                                    │
                                    ├─ energy VAD drops silent windows
                                    ▼
                          CNN ──> p(male) per window
                                    │
                                    ▼
                    average ──> female / male / uncertain
```

Four convolutional blocks (16→32→64→64 channels), each `Conv2d → BatchNorm →
ReLU → MaxPool`, then global average pooling and a linear layer. GAP instead of
flatten keeps the model small and lets it accept any input length.

When the averaged probability falls between 0.35 and 0.65 the answer is reported
as **uncertain** rather than forced into a class. In testing this correctly
flagged a dataset account that turned out to contain two different speakers.

## Quick start

**1. Install.**

```bash
git clone <repository-url>
cd rado-gender_classification
python -m venv venv && venv/Scripts/activate      # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

This installs the `genderid` package in editable mode along with everything the
pipeline and the interfaces need. Skipping it means `import genderid` will fail.

**2. Get the model weights.** Checkpoints are not stored in git.

```bash
python scripts/00_get_model.py
```

Or train your own — see [Reproducing the models](#reproducing-the-models) below.

**3. Predict.**

```python
from genderid import GenderClassifier

clf = GenderClassifier("models/cnn_best.pt")
result = clf.predict_file("sample.wav")

print(result.label)        # 'Ayol' | 'Erkak' | 'uncertain' | 'no_speech'
print(result.confidence)   # None when uncertain
print(result.probability)  # p(male), averaged over speech windows
```

`predict_file` accepts anything ffmpeg can read (wav, mp3, ogg/opus, m4a) and
needs ffmpeg on PATH. `predict_waveform(array, sample_rate)` works on in-memory
audio with no external dependency.

**4. Or run the interfaces.**

```bash
python -m app.gradio_app
```

Opens the web UI at `http://127.0.0.1:7860` with file upload and live microphone.
For the Telegram bot set `TELEGRAM_BOT_TOKEN` and run `python -m app.telegram_bot`.

## Reproducing the models

Scripts are numbered in execution order. Steps 5–7 need a GPU and are only
required for the transformer baseline.

| Step | Script | What it does | Time |
|---|---|---|---|
| 0 | `00_get_model.py` | Download pretrained weights (skip if training) | 1 min |
| 1 | `01_explore.py` | Inspect dataset schema and balance | 1 min |
| 2 | `02_prepare.py` | Filter, cap per speaker, speaker-disjoint split | 17 min |
| 3 | `03_features.py` | Cache log-mel features | 6 min |
| 4 | `04_train_cnn.py` | Train the CNN | 55 min (CPU) |
| 5 | `05_fetch_audio.py` | Re-fetch the same clips on a GPU machine | 8 min |
| 6 | `06_finetune_w2v2.py` | Fine-tune wav2vec2 | 15 min (T4) |
| 7 | `07_predict_w2v2.py` | Score the test split | 2 min |
| 8 | `08_compare.py` | Metrics, slices, significance, latency | 2 min |
| 9 | `09_export.py` | Verify parity and build deployment bundles | 1 min |

```bash
python scripts/02_prepare.py
python scripts/03_features.py
python scripts/04_train_cnn.py --quick    # smoke test first
python scripts/04_train_cnn.py
```

Every script accepts `--help`. Training scripts support `--quick` for a
two-minute run on a subset before committing to the full job.

## Dataset

[DavronSherbaev/uzbekvoice-filtered](https://huggingface.co/datasets/DavronSherbaev/uzbekvoice-filtered)
— crowdsourced Uzbek speech, Apache 2.0, roughly 500k clips with speaker,
gender, age and region metadata.

This project uses 30,000 clips (34.8 hours) from 3,505 speakers. Three
preparation rules matter more than volume:

- **Quality gate on metadata**, before any audio is decoded — rejects clips with
  community reports or more downvotes than upvotes.
- **40 clips per speaker maximum.** In the raw data the top five contributors
  accounted for 23% of a sample; capping brings that to 0.7%.
- **Speaker-disjoint splitting.** Every clip from one contributor lands in a
  single split, so test accuracy measures generalisation to new voices rather
  than memorisation.

See [docs/DATASET.md](docs/DATASET.md) for details and known data issues.

## Deployment

The `09_export.py` script builds two self-contained bundles and refuses to do so
unless feature and prediction parity checks pass.

- **Hugging Face Space** — Gradio UI, free CPU tier, HTTPS (so the microphone
  works).
- **Railway / Docker** — Telegram bot and web UI in one container.

Deployment inference drops librosa entirely: mel spectrograms are computed with
`torch.stft` against a precomputed filterbank, verified to agree with librosa to
within 0.001 dB. That removes scipy and numba from the image.

Step-by-step instructions: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Project layout

```
genderid/          Reusable package: config, model, features, inference
scripts/           Numbered pipeline, from raw dataset to deployment bundle
app/               Gradio UI, Telegram bot, Dockerfile
docs/              Dataset notes, results, deployment, model card
docs/lessons/      Step-by-step tutorial in Uzbek (how this was built)
models/            Checkpoints and reports (gitignored)
data/              Prepared audio and feature cache (gitignored)
```

## Limitations

- Trained on Uzbek speech; accuracy on other languages is unverified.
- Gender is modelled as binary because the source dataset is labelled that way.
  This reflects the dataset, not the range of human gender identity.
- The energy-based VAD treats loud noise as speech. A neural VAD (e.g. Silero)
  is the right upgrade for noisy environments.
- The training data skews young: about 75% of speakers are 18–24.
- Predictions are approximate. Do not use them to make decisions about
  individuals.

## License

Apache 2.0 — see [LICENSE](LICENSE). The training dataset is Apache 2.0 as well.
