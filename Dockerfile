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

RUN python3.11 -m venv /opt/venv
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && pip check

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
