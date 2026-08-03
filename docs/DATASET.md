# Dataset

## Source

[DavronSherbaev/uzbekvoice-filtered](https://huggingface.co/datasets/DavronSherbaev/uzbekvoice-filtered)
— crowdsourced Uzbek speech under Apache 2.0, roughly 500k clips (~600 hours).

Relevant columns:

| Column | Notes |
|---|---|
| `path` | the audio itself — **not** `audio`, which is the usual name |
| `sentence` | transcription |
| `client_id` | contributor identifier; the basis for splitting |
| `duration` | seconds, present in metadata so filtering needs no decoding |
| `gender` | `Ayol` (female) / `Erkak` (male) |
| `year_of_birth` | age bracket, not a year |
| `accent_region` | 20 regions of Uzbekistan |
| `native_language` | 7 options |
| `upvotes_count`, `downvotes_count`, `reported_count` | community quality signals |

Audio is already 16 kHz mono, so no resampling is needed during preparation.
Clip lengths run 0.47–12.6 s with a mean of 4.33 s.

Gender balance in the source is unusually good: 88,082 female against 90,943
male, with 25 missing — no resampling or class weighting required.

## What this project uses

30,000 clips (34.8 hours) from 3,505 speakers, selected by
`scripts/02_prepare.py`.

| Split | Clips | Share | Female | Male | Speakers |
|---|---|---|---|---|---|
| train | 23,942 | 79.8% | 11,982 | 11,960 | 2,768 |
| val | 3,009 | 10.0% | 1,507 | 1,502 | 373 |
| test | 3,049 | 10.2% | 1,511 | 1,538 | 364 |

Of 133,688 rows scanned, 797 failed the quality gate (0.6%) and 102,891 were
skipped by quota or the per-speaker cap. The bulk of the filtering enforces
balance and diversity, not quality.

## Preparation rules

### Quality gate before decoding

A clip is kept when `gender` is valid, `duration` is 1.5–10 s,
`reported_count == 0`, and `downvotes_count <= upvotes_count`. All four are
metadata checks, so rejected clips never cost an audio decode.

### Per-speaker cap of 40

In a 2,000-row sample the five most prolific contributors produced 23% of the
clips. Without a cap the model would fit a handful of voices. After capping,
the top five account for 0.7% of the prepared set and the median contributor
has 3 clips.

### Speaker-disjoint splitting

Every clip from one `client_id` lands in a single split. Splitting by clip
instead would let the model recognise a voice it had memorised, inflating test
accuracy while telling you nothing about performance on new speakers.

The assignment shuffles speakers with a fixed seed, then greedily fills test,
then val, then train, per gender. Buckets overshoot their 10% target by at most
one speaker's worth of clips. `02_prepare.py` asserts zero speaker overlap
before writing anything.

### Label conflict removal

If one `client_id` appears with both gender labels, all of its clips are
dropped — there is no way to know which label is correct. None were found in
the prepared subset, which is itself useful information.

## Known data issues

### Shared accounts

Test-set error analysis found that account `3d3fca02…` (Andijon, labelled male)
contains recordings from two different people. Both models classify 24 of its
40 clips as female with high confidence; manual listening confirmed a male and a
female voice under one ID. A second account, `05086718…`, has all six of its
clips confidently classified against its label.

Together these two accounts produce 30 of the 58 CNN test errors. See
[RESULTS.md](RESULTS.md) for the full analysis.

If you build on this dataset, run the same check: find clips where two
independent models are confidently wrong in agreement, then group by
`client_id`. Concentration in a few speakers indicates label noise rather than
model weakness.

### Overlapping age brackets

`year_of_birth` values overlap and are inconsistent, presumably because the
collection form changed over time:

| Bracket | Count |
|---|---|
| 18-24 | 134,906 |
| 25-34 | 11,804 |
| 35-... | 9,479 |
| 19-29 | 8,494 |
| 12-17 | 5,811 |
| < 19 | 3,309 |
| 30-39 | 660 |

A speaker could fall into both `18-24` and `19-29`. Any age-prediction work must
harmonise these into non-overlapping bands (for example `<18`, `18-24`, `25-34`,
`35+`) first.

### Age skew

About 75% of clips come from speakers aged 18–24. Gender classification turned
out to be insensitive to this, but tasks that depend on vocal maturity would be
affected.

### Regional imbalance

Toshkent and Andijon dominate. Region or accent classification would need
per-class quotas, class-weighted loss, or both.

## Signal check

Before training anything, it is worth confirming the label carries acoustic
signal. Averaging log-mel energy over 800 clips per class:

| Mel bands | Frequency | Female | Male | Difference |
|---|---|---|---|---|
| 0–8 | 0–383 Hz | −40.3 dB | −33.3 dB | **7.1 dB** |
| 8–16 | 383–766 Hz | −42.3 dB | −40.0 dB | 2.3 dB |
| 16–32 | 766–1731 Hz | −52.4 dB | −50.0 dB | 2.4 dB |

The separation is concentrated in the F0 region, exactly where voice physiology
predicts it: male fundamentals sit around 85–155 Hz against 165–255 Hz for
female speakers, so the lowest mel bands hold more male harmonics.

## Other tasks this dataset supports

The same preparation pipeline transfers with a change of label column:

| Task | Label source | Notes |
|---|---|---|
| Age bracket | `year_of_birth` | harmonise the overlapping brackets first |
| Region / accent | `accent_region` | 20 classes, heavily imbalanced |
| Native language | `native_language` | 7 classes |
| Multi-task | all three | one backbone, three heads |
| Clip quality prediction | vote counts | predicts community rejection |
| Speaker verification | `client_id` | needs a metric-learning objective |
| Speech recognition | `sentence` | ~600 hours; the dataset's original purpose |
