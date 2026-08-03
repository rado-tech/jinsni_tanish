# Results

All numbers come from the held-out test split: 3,049 clips, 364 speakers, none
of whom appear in training or validation. The split was fixed before any model
was trained and used exactly once.

## Headline comparison

| Metric | CNN (from scratch) | wav2vec2-base (fine-tuned) |
|---|---|---|
| Accuracy | 98.10% | 98.75% |
| Error rate | 1.90% | 1.25% |
| Errors | 58 | 38 |
| Precision (male) | 0.9940 | 0.9993 |
| Recall (male) | 0.9681 | 0.9759 |
| Recall (female) | 0.9940 | 0.9993 |
| F1 | 0.9809 | 0.9875 |
| Parameters | 60,706 | 94,569,090 |
| Size on disk | 0.2 MB | 360.8 MB |
| CPU latency (2 s, batch 1) | 4.02 ms (p90 4.49) | 504.42 ms (p90 520.49) |

Confusion matrices (male = positive):

| | TN | FP | FN | TP |
|---|---|---|---|---|
| CNN | 1502 | 9 | 49 | 1489 |
| wav2vec2 | 1510 | 1 | 37 | 1501 |

## Is the difference real?

Both models scored the same clips, so the comparison is paired and McNemar's
test applies. It counts only disagreements:

- CNN correct where wav2vec2 was wrong: **1**
- wav2vec2 correct where CNN was wrong: **21**
- χ² = 16.41, critical value 3.841 → **significant** (p < 0.05)

The asymmetry is the interesting part: wav2vec2's errors are almost a subset of
the CNN's. The transformer rarely fails where the small model succeeds.

Read in error-rate terms rather than accuracy points: 1.90% → 1.25% is a 34%
reduction in mistakes, not a 0.65-point improvement. At high accuracy the
percentage-point gap always understates the difference.

## Error analysis: the errors are mostly label noise

37 clips were misclassified by **both** models. Grouping them by speaker:

| Speaker | Region | Label | Errors | Model confidence |
|---|---|---|---|---|
| `3d3fca02…` | Andijon | Erkak | 24 of 40 clips | both models p(male) ≈ 0.40 |
| `05086718…` | Toshkent | Erkak | 6 of 6 clips | confidently female |
| three others | — | 1 each | | |

**34 of the 37 shared errors come from five speakers.** On those clips
wav2vec2's confidence in the wrong answer had a median of 1.000, with 35 of 37
above 0.95.

That pattern is diagnostic. A model failing on a genuinely hard example is
uncertain (p ≈ 0.5). Two architecturally unrelated models — a 60k-parameter CNN
and a 94M-parameter transformer — being confidently wrong on the *same* clips
points at the labels, not the models.

Manual listening confirmed it: account `3d3fca02` contains recordings from both
a man and a woman, submitted under one `client_id`. Both models split its clips
24 female / 16 male, which is exactly what a shared account would produce.

Excluding those two speakers (46 clips, 1.5% of the test set):

| | Full test set | Excluding two suspect speakers |
|---|---|---|
| CNN | 98.10% (58 errors) | 99.07% (28 errors) |
| wav2vec2 | 98.75% (38 errors) | 99.73% (8 errors) |

**The cleaned figures are diagnostic, not headline results.** Removing test
examples after seeing the errors would invalidate the evaluation. The reportable
numbers remain 98.10% and 98.75%.

## Slices

### Clip duration

| Duration | n | CNN | wav2vec2 |
|---|---|---|---|
| <2 s | 14 | 1.0000 | 1.0000 |
| 2–3 s | 326 | 0.9847 | 0.9969 |
| 3–4 s | 1023 | 0.9863 | 0.9941 |
| 4–5 s | 1008 | 0.9792 | 0.9831 |
| >5 s | 678 | 0.9735 | 0.9794 |

Longer clips look harder, which is counter-intuitive. It is a confound: the
share of clips belonging to the five problem speakers rises with duration (0%
below 2 s, 7.4% above 5 s). After excluding them the curve flattens — wav2vec2
reaches 100% on the 2–4 s buckets and 99.25% above 5 s.

### Age bracket

| Bracket | n | CNN | wav2vec2 |
|---|---|---|---|
| 30–39 | 58 | 0.9655 | 1.0000 |
| 18–24 | 1741 | 0.9736 | 0.9799 |
| 25–34 | 528 | 0.9905 | 1.0000 |
| <19 | 141 | 0.9929 | 0.9929 |
| 12–17 | 149 | 0.9933 | 0.9933 |

A pre-registered hypothesis — that adolescent male voices (12–17) would be the
main failure mode, since the voice has not yet broken — was **not supported**.
That bracket is among the best-served. No age effect was found.

### Region

| Region | n | CNN | wav2vec2 |
|---|---|---|---|
| Andijon | 375 | 0.9333 | 0.9333 |
| Namangan | 177 | 0.9661 | 1.0000 |
| Toshkent shahri | 453 | 0.9779 | 0.9801 |
| Buxoro | 256 | 0.9844 | 0.9961 |
| Farg'ona | 307 | 0.9870 | 1.0000 |

Andijon scoring identically for both models was the clue that led to the label
noise finding above — its errors trace to a single shared account. After
excluding it, Namangan becomes the weakest region for the CNN (96.6%) while
wav2vec2 handles it perfectly, suggesting genuine accent sensitivity in the
smaller model.

## Training behaviour

### CNN — 15 epochs, 4 CPU cores, ~55 minutes

Best validation accuracy 98.50% at epoch 14.

Two things are worth noting from the curves:

- **Epoch 1 shows train 97.5% against val 87.9%.** This is not overfitting; it
  is BatchNorm. In training mode BN uses batch statistics, in eval mode it uses
  running averages that lag while the weights are still moving quickly. It
  resolves by epoch 2.
- **Validation accuracy oscillates** (dips to 94.2% and 95.0%) while training
  accuracy stays smooth. The learning rate of 3e-3 is on the high side for this
  model; the cosine schedule stabilises the last four epochs at 98.2–98.5%.
  A run at 1e-3 would produce a smoother curve.

The final train/val gap is about 1 point, and validation loss keeps falling
(0.345 → 0.054), so random cropping and dropout did their job.

### wav2vec2 — 3 epochs, one T4 GPU, ~15 minutes

| Epoch | Train loss / acc | Val loss / acc |
|---|---|---|
| 1 | 0.0709 / 0.9795 | 0.0301 / 0.9920 |
| 2 | 0.0206 / 0.9953 | 0.0187 / 0.9963 |
| 3 | 0.0130 / 0.9972 | 0.0200 / 0.9967 |

Validation accuracy after a single epoch (99.20%) already exceeded anything the
CNN reached in fifteen. That is the value of pretrained representations.

Validation loss ticks up in epoch 3 while accuracy gains one clip — the earliest
sign of overfitting, and a reasonable place to stop.

## Reproducing

```bash
python scripts/08_compare.py
```

Writes `models/comparison.json` and `models/comparison.png`. Requires
`models/cnn_best.pt`; `models/w2v2_test_preds.csv` and `models/w2v2_best/` are
optional and enable the transformer columns and the latency comparison.
