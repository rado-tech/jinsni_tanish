# Container image for Railway / any Docker host.
#
# ffmpeg is required: Telegram delivers voice notes as OGG/Opus, which the
# Python audio stack cannot decode on its own.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Install torch from the CPU index first. A plain `pip install torch` on Linux
# pulls the CUDA build (~2.5 GB), which is dead weight without a GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "app.main"]
