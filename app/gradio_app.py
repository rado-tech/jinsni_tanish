"""Gradio web interface: file upload and live microphone."""

from __future__ import annotations

import numpy as np
import gradio as gr

from genderid import load_default
from genderid.config import SAMPLE_RATE, WINDOW_SAMPLES
from genderid.features import is_speech, resample
from genderid.inference import NO_SPEECH, TOO_SHORT, UNCERTAIN

CLASSIFIER = load_default()

TITLE = "Ovozdan jins aniqlash · Speaker gender from voice"
DESCRIPTION = """
CNN trained on Uzbek speech. **98.1%** test accuracy, 60,706 parameters,
~4 ms per prediction on CPU.

When the model is not confident it says **"aniq emas" (uncertain)** instead of
guessing.
"""
DISCLAIMER = """
---
*Trained on [uzbekvoice-filtered](https://huggingface.co/datasets/DavronSherbaev/uzbekvoice-filtered).
Predictions are approximate. Not suitable for decisions about individuals.*
"""


def _to_mono_float(audio: tuple[int, np.ndarray]) -> tuple[np.ndarray, int]:
    """Gradio hands over (sample_rate, array); normalise to float32 mono."""
    sr, y = audio
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    return y.astype(np.float32), sr


def _bar(p: float, width: int = 24) -> str:
    filled = int(p * width)
    return "█" * filled + "░" * (width - filled)


def analyze_file(audio):
    if audio is None:
        return "Upload or record audio first.", ""

    y, sr = _to_mono_float(audio)
    result = CLASSIFIER.predict_waveform(y, sr)

    if result.label == NO_SPEECH:
        return "### Nutq topilmadi / No speech\nSpeak louder or record somewhere quieter.", ""
    if result.label == TOO_SHORT:
        return "### Juda qisqa / Too short\nAt least one second of audio is needed.", ""

    p = result.probability
    lines = [f"## {result.label}", "", f"`Ayol {_bar(p)} Erkak`", ""]
    if result.label == UNCERTAIN:
        lines.append(
            "The voice sits in the ambiguous band — it may be borderline, or the "
            "recording may contain more than one speaker."
        )
    else:
        lines.append(f"Confidence: **{result.confidence * 100:.1f}%**")
    lines += ["", f"p(Erkak) = {p:.3f}"]

    windows = " ".join(
        f"[{i * 0.5:.1f}s {'speech' if f else '-':6} {pr:.2f}]"
        for i, (pr, f) in enumerate(
            zip(result.window_probabilities, result.window_speech)
        )
    )
    detail = (
        f"{result.duration:.1f} s | {result.n_windows} windows, "
        f"{result.n_speech} with speech\n{windows}"
    )
    return "\n".join(lines), detail


def stream(new_chunk, buffer):
    """Rolling 2-second buffer; re-classified on every incoming chunk."""
    if new_chunk is None:
        return buffer, "Speak into the microphone..."

    y, sr = _to_mono_float(new_chunk)
    y = resample(y, sr)
    buffer = y if buffer is None else np.concatenate([buffer, y])
    buffer = buffer[-WINDOW_SAMPLES:]

    if len(buffer) < SAMPLE_RATE:
        return buffer, "Listening..."

    segment = buffer
    if len(segment) < WINDOW_SAMPLES:
        segment = np.pad(segment, (0, WINDOW_SAMPLES - len(segment)))

    if not is_speech(segment):
        rms_db = 20 * np.log10(np.sqrt(np.mean(segment ** 2)) + 1e-10)
        return buffer, f"No speech detected ({rms_db:.0f} dB)"

    p = CLASSIFIER.predict_window(segment)
    label = (
        CLASSIFIER.labels[1] if p > 0.65
        else CLASSIFIER.labels[0] if p < 0.35
        else UNCERTAIN
    )
    return buffer, f"## {label}\n\n`Ayol {_bar(p)} Erkak`\n\np(Erkak) = {p:.3f}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

        with gr.Tab("Upload / Fayl"):
            audio_in = gr.Audio(
                sources=["upload", "microphone"], type="numpy", label="Audio"
            )
            run = gr.Button("Analyze", variant="primary")
            verdict = gr.Markdown()
            detail = gr.Textbox(label="Per-window detail", lines=3)
            run.click(analyze_file, inputs=audio_in, outputs=[verdict, detail])

        with gr.Tab("Live / Realtime"):
            gr.Markdown("Enable the microphone and talk — the last 2 seconds are analysed.")
            state = gr.State(None)
            mic = gr.Audio(
                sources=["microphone"], streaming=True, type="numpy", label="Microphone"
            )
            live = gr.Markdown("Enable the microphone...")
            mic.stream(stream, inputs=[mic, state], outputs=[state, live])

        gr.Markdown(DISCLAIMER)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
