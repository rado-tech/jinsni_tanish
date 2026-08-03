# Deployment

Only the CNN is deployed. wav2vec2 buys 0.65 accuracy points for a 361 MB model,
~500 MB of RAM per request and 504 ms of latency — a poor trade on
usage-billed infrastructure, and impossible for real-time streaming.

Latency arithmetic for a 2-second window refreshed every 0.5 s:

| Model | Per second of audio | Feasible in real time? |
|---|---|---|
| CNN | 2 × ~7 ms = 14 ms | yes, ~1.4% of one core |
| wav2vec2 | 2 × 504 ms = 1008 ms | no, over 100% of one core |

## Build the bundles

```bash
python scripts/09_export.py
```

Two parity checks run first and abort the build on failure:

1. **Feature parity** — deployment computes mel spectrograms with `torch.stft`
   and a precomputed filterbank instead of librosa (which would drag in scipy
   and numba). Both paths are compared on 60 real clips; the measured difference
   is ~0.0001 dB against a 0.05 dB tolerance.
2. **Prediction parity** — the packaged pipeline must return the same
   probabilities as the training-time model on the same audio.

Silent accuracy loss after an optimisation is the most common deployment
failure. These checks make it loud.

Output:

```
build/spaces/     app.py, requirements.txt, README.md, genderid/, models/
build/railway/    app/, genderid/, models/, Dockerfile, railway.json, requirements.txt
```

## Hugging Face Space

Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space)
with SDK **Gradio** (not Static) and hardware **CPU basic**.

Upload with the CLI — Hugging Face no longer accepts password authentication
over git, and the CLI handles tokens for you:

```bash
pip install -U "huggingface_hub[cli]"
```

```bash
huggingface-cli login
```

```bash
huggingface-cli upload <user>/<space-name> build/spaces . --repo-type=space
```

Create the token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with
the **Write** role. If you prefer git, paste the same token when prompted for a
password.

The Space serves over HTTPS, so microphone capture works. Free Spaces sleep
after 48 hours of inactivity and take about 30 seconds to wake.

**If the free hardware option is greyed out** you have hit the CPU Basic quota,
not a paywall — Gradio Spaces are free. Pause or delete other Spaces under
[settings/spaces](https://huggingface.co/settings/spaces). The quota is known to
lag; if it does not clear, deploy the web UI on Railway instead (below), which
also serves HTTPS.

## Railway (Telegram bot + web UI)

One service runs both interfaces in a single process, sharing one loaded model.

**1. Create the bot.** Message [@BotFather](https://t.me/BotFather) on Telegram,
send `/newbot`, and keep the token private.

**2. Push `build/railway/` to a GitHub repository.**

**3. Railway:** New Project → Deploy from GitHub repo. The `Dockerfile` is
detected automatically.

**4. Variables:**

```
TELEGRAM_BOT_TOKEN = <token from BotFather>
```

Optional: `RUN_WEB=0` disables the web UI and runs the bot alone.

**5. Networking → Generate Domain** for the web UI. Railway issues an HTTPS
`*.up.railway.app` address, so the microphone works there too.

**6. Deploy.** The build takes 4–7 minutes (torch CPU wheel). Logs should show:

```
Gradio 6.x.x starting on port 8080
Web UI listening on port 8080
Connecting to Telegram...
```

### Operational notes

- **One polling instance per token.** Running the bot locally while Railway is
  live gives Telegram `409 Conflict` and breaks both.
- **Port.** Polling needs no inbound port; Gradio does. `app/main.py` reads
  `$PORT`. With `RUN_WEB=0`, Railway's "no port detected" message is expected.
- **Always on.** Unlike a free Space, Railway does not sleep, so it bills
  continuously — roughly 300 MB of RAM at idle.
- **Never commit the token.** It belongs in Railway Variables. If it leaks,
  revoke it immediately with `/revoke` in BotFather.

## Troubleshooting

**`telegram.error.TimedOut` on startup.** Container cold start plus cloud egress
exceeds python-telegram-bot's 5-second default. `app/telegram_bot.py` already
raises all timeouts to 30 s and sets `bootstrap_retries=-1`. If it persists,
switch the Railway region (Settings → Region → Europe West) or move from polling
to webhooks.

**`Blocks.launch() got an unexpected keyword argument`.** Gradio changes
`launch()` parameters across majors. `app/main.py` inspects the signature and
passes only supported names, so this should not occur; if a new Gradio release
breaks something else, pin the version in `requirements.txt`.

**`PermissionError` during `09_export.py`.** Fixed — bundle directories are
overwritten rather than deleted, because a `.git` folder inside them contains
read-only objects that `shutil.rmtree` cannot remove on Windows.

**ffmpeg missing locally.** Only affects local testing of the bot; the Docker
image installs it. Get it from [ffmpeg.org](https://ffmpeg.org/download.html) if
you want to test before deploying.

## Publishing the model to the Hub

To share the weights as a standalone model repository:

```bash
huggingface-cli upload <user>/uzbek-gender-cnn models/cnn_best.pt cnn_best.pt
```

```bash
huggingface-cli upload <user>/uzbek-gender-cnn docs/MODEL_CARD.md README.md
```

The model card in [MODEL_CARD.md](MODEL_CARD.md) already carries the metadata
header the Hub expects.

## Image size

The image is dominated by torch (~900 MB). Exporting the network to ONNX and
serving with `onnxruntime` (~50 MB) would cut it roughly threefold and speed up
cold starts. For a 60,706-parameter CNN the conversion is straightforward; it
has not been done here because torch keeps the deployment identical to the
training code.
