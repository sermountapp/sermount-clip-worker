# Clip render worker for RunPod Serverless (built by RunPod from GitHub).
# Same base as sermount-whisperx-worker, pinned by digest; ffmpeg comes with
# the base image (Ubuntu 22.04 build with h264_nvenc). No torch, no models.
FROM --platform=linux/amd64 runpod/base:1.4.0-cuda1281-ubuntu2204@sha256:6a1a66d92c5a9946c2748415c923321ed816c26f5d331a2389f4646145611965

ENV PYTHONUNBUFFERED=1 \
    RUNPOD_LOG_LEVEL=INFO \
    CAPTION_FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
    PATH=/opt/venv/bin:$PATH

# The caption font is baked in; a worker never downloads anything but the
# source range it is given.
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/* \
 && ffmpeg -hide_banner -encoders 2>/dev/null | grep h264_nvenc >/dev/null \
 && ffmpeg -hide_banner -filters 2>/dev/null | grep -w overlay >/dev/null
# (no grep -q: it exits on the first match while ffmpeg is still writing, and
#  under the build shell's pipefail that SIGPIPE reported the check as failed.)

# deno, the JS runtime yt-dlp solves YouTube's challenges with (without one
# it warns "No supported JavaScript runtime" and some formats go missing).
# 2.9.5 is the version yt-dlp 2026.08.19 pins in its pin-deno extra, beside
# yt-dlp-ejs==0.8.0 (requirements.txt). Baked at build and checked against
# the release's published sha256: a mismatch fails the build. Python unzips
# (the base image need not carry unzip).
ARG DENO_VERSION=2.9.5
ARG DENO_SHA256=8b010a3b1a4a0188a67cdb8a7a27348b2a501af78aec7fc74f2ace167368d530
ENV DENO_INSTALL=/opt/deno \
    PATH=/opt/deno/bin:$PATH
RUN mkdir -p /opt/deno/bin \
 && python3.11 -c "import urllib.request; urllib.request.urlretrieve('https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip', '/tmp/deno.zip')" \
 && echo "${DENO_SHA256}  /tmp/deno.zip" | sha256sum -c - \
 && python3.11 -c "import zipfile; zipfile.ZipFile('/tmp/deno.zip').extract('deno', '/opt/deno/bin')" \
 && chmod 755 /opt/deno/bin/deno \
 && rm /tmp/deno.zip \
 && deno --version | grep "^deno ${DENO_VERSION} "

RUN python3.11 -m venv /opt/venv
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && pip check

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
