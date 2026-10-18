"""RunPod serverless handler: render one approved sermon clip.

Input (`{"input": ...}`, every URL short-lived and signed; the worker never
holds a database or storage key):

    {"source": {"kind": "youtube", "url": "...", "start": 3721.0, "end": 3812.5}
             | {"kind": "audio", "audio_url": "...", "start": 121.0, "end": 212.5},
     "cache": {"get_url": "<signed GET or null>", "put_url": "<signed PUT or null>", "token": "..."},
     "output": {"put_url": "<signed PUT>", "token": "...", "container": "mp4" | "m4a"},
     "aspect": "9:16", "width": 1080, "height": 1920,
     "captions": {"cues": [{"s": 0.0, "e": 2.1, "words": [{"w": "I", "s": 0.0, "e": 0.2}]}],
                  "position": "bottom" | "middle" | "top"} | null,
     "edits": {"removed": [[12.4, 13.1]]},
     "normalize_audio": true,
     "max_output_seconds": 1200,
     "background": "#0f1115",
     "identity": {"logo_url": "<signed GET or null>", "church_name", "title", "speaker", "date"} | null}
                  (audio sermons in a video format only: the audiogram's identity stack)

Times inside `source` are on that source's own clock (YouTube full-video time
for youtube, stored-audio time for audio). `edits.removed` and caption cues
are relative to `source.start`, on the source clock; the removals are cut
from the timeline (audio and video together) and the cues are mapped onto
the output clock with the same list, so a caption never points at a cut word.

Output (always; the handler never raises to RunPod):

    {"ok": true|false, "reason": null|"input|fetch|cache|render|upload|too_long|timeout|error",
     "bytes", "duration", "fetch_ms", "render_ms", "exec_ms", "cache_hit", "encoder", "image_tag",
     "cause": null|"<one word>", "cause_detail": null|"<redacted, one line, <= 300 chars>",
     "proxy_probe": "ok|auth|unreachable|skipped", "proxy_fp": "<8 hex>"|null, "ports_tried": int|null}
  (v0.2.5: `reason` keeps its eight words, which the app maps; `cause` says
  which kind, e.g. fetch → proxy_config | proxy_auth | proxy_unreachable |
  blocked | unavailable | no_format | fetch_timeout | fetch_other.)

Rules (from SermonVault's measured findings, docs/edit-mode-clips-brief.md):
  - Fetch only the range: cache GET first; on a miss yt-dlp download_ranges
    over [start - 10, end + 10] with force_keyframes_at_cuts, 1080p cap, the
    encoder args in external_downloader_args (NOT postprocessor_args, which
    FFmpegFD ignores) AND merge_output_format mp4, then PUT the range to the
    cache. One fetch at a time per worker process. YouTube needs PROXY_URL
    (no proxy fails fast); three attempts rotate PROXY_PORTS inside a
    FETCH_WALL_SECS clock, each stall bounded by a 20 s socket timeout.
  - Render: trim to the exact range, cut `removed` with select/aselect,
    centre-crop and scale, burn captions (never -c:v copy with a burn),
    loudnorm when asked. h264_nvenc when present, else libx264.
  - Captions are drawn by Pillow into PNG strips (one per spoken-word state)
    and composited with `overlay`, so the burn needs neither libass nor
    freetype in ffmpeg and the layout lives in Python, where the app's page
    mirrors it: bold sans, dark pill, current word in the accent colour, at
    captions.position (bottom | middle | top), two lines split at the space
    nearest the middle. Lines are measured in pixels against the frame's own
    width less a side margin, and the font steps down until two lines fit,
    so a 9:16 frame wraps narrower than a 16:9 one.
  - Logs: counts, timings, encoder and reason only. Never URLs, caption text
    or words. Every temp file is deleted in a `finally`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.parse import unquote, urlparse, urlunparse

import requests

IMAGE_TAG = "v0.2.6"  # bump with every release; the app records it per render

PAD_SECS = 10.0
MAX_SOURCE_BYTES = 4 * 1024 * 1024 * 1024
DOWNLOAD_TIMEOUT = (10, 120)
DOWNLOAD_CHUNK = 1024 * 1024
UPLOAD_TIMEOUT = (10, 600)
ASPECTS = {"16:9", "1:1", "9:16", "audio"}

# Encode (v0.2.4). Every output has a bitrate ceiling and a keyframe every
# 2 s. An audiogram is a near-still picture (identity background, caption
# pill, waveform): uncapped, nvenc spent ~4.5 Mbit/s on it (render 15: 40.8 MB
# for 69.5 s); at cq 30 it still ran AT v0.2.3's 1.5M ceiling (14.65 MB for
# 70.5 s), so the ceiling is what sets its size: 0.8M is ~8.5 MB per 70 s.
# A video clip at cq 23 with no ceiling ran ~6.6 Mbit/s (render 13: 64.3 MB
# for 75.5 s); the user chose the CRF 26 tier, under a 4M guard for heavy
# motion. docs/clip-worker-v0-2-4-encode-brief.md.
AUDIOGRAM_NVENC_CQ = "30"
AUDIOGRAM_X264_CRF = "32"
AUDIOGRAM_MAXRATE = "0.8M"
AUDIOGRAM_BUFSIZE = "1.6M"
VIDEO_NVENC_CQ = "26"
VIDEO_X264_CRF = "26"
VIDEO_MAXRATE = "4M"
VIDEO_BUFSIZE = "8M"
GOP_FRAMES = "60"               # a keyframe every 2 s at 30 fps
AUDIO_SAMPLE_RATE = "48000"     # loudnorm resamples (it passed 96 kHz through); every output is 48 kHz
DEFAULT_BACKGROUND = "#0f1115"
ACCENT = "#B8842E"
CAPTION_TEXT = "#F6F3EC"
CAPTION_PILL = (15, 17, 21, 210)
CAPTION_MAX_CHARS = 84
# Caption position: the caption's BOTTOM edge at this fraction of the frame
# height, from the top. Same literals as the app's app/clip_captions.py
# CAPTION_ANCHORS (and edit-mode.js, which previews it).
CAPTION_ANCHORS = {"bottom": 0.88, "middle": 0.56, "top": 0.24}
CAPTION_SIDE_MARGIN = 0.06     # of the frame width, each side; no line reaches closer to the edge
CAPTION_MIN_FONT_PX = 18
# v0.2.1: ONE caption size per format, a fraction of the frame height (same
# numbers as the app's app/clip_captions.py CAPTION_FONT_FRAC). The app cuts
# cues to fit two lines at this size; a third line is the fallback, and the
# font only shrinks for a single word wider than the frame.
CAPTION_FONT_FRAC = {"9:16": 0.034, "1:1": 0.046, "16:9": 0.046}
CAPTION_MAX_LINES = 3
FONT_CANDIDATES = (
    os.environ.get("CAPTION_FONT", ""),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
)
# The identity stack's quiet face (speaker · date); the display face is the caption font.
FONT_REGULAR_CANDIDATES = (
    os.environ.get("IDENTITY_FONT_REGULAR", ""),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
)
IDENTITY_TEXT_MAX = 200
LOGO_MAX_BYTES = 5 * 1024 * 1024
WAVE_COLOUR = "0xB8842E@0.85"      # the house accent; 0.55 was nearly invisible on render 11
WAVE_HEIGHT_FRAC = 0.12            # of the frame height (16:9, 1:1)
WAVE_HEIGHT_FRAC_TALL = 0.14       # 9:16
# The wave branch only (the audio track is untouched): speech at a pulpit
# mic's level drew a 2-60 px line in a 230 px box (render 14, v0.2.1), since
# showwaves draws amplitude. Levelled first (up to +40 dB, a short window),
# its peaks fill 80-90 % of the box at any source level (measured on a
# -34 dB speech sample and the test sample at -24 dB).
WAVE_LEVEL = "dynaudnorm=f=100:g=7:p=0.95:m=100"
FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")
PROXY_SCHEMES = {"http", "https", "socks4", "socks5", "socks5h"}
FETCH_RETRY_SLEEP = 2.0
FETCH_ATTEMPTS = 3              # attempt n leaves through PROXY_URLS[(n - 1) % len]
YTDLP_MESSAGE_CAP = 300
# One GET through the proxy at boot (v0.2.5): a ~1 KB JSON the vendor serves for
# exactly this. 200 ok, 407 auth, anything else unreachable. It never stops
# the worker: the vendor endpoint can be down while fetches work.
PROXY_PROBE_URL = "https://ip.decodo.com/json"
PROXY_PROBE_TIMEOUT = (5, 10)

def _env_secs(name, default):
    try:
        v = float(os.environ.get(name) or default)
        return v if v > 0 else default
    except ValueError:
        return default


# The fetch's clocks (v0.2.5). Every attempt must finish inside the wall
# clock, checked between attempts; inside an attempt, yt-dlp's socket timeout
# and ffmpeg's -rw_timeout (per-socket inactivity, not total) bound a stall,
# since yt-dlp's FFmpegFD waits on ffmpeg with no timeout of its own.
FETCH_WALL_SECS = _env_secs("FETCH_WALL_SECS", 600.0)
SOCKET_TIMEOUT = 20.0

logging.getLogger("urllib3").setLevel(logging.WARNING)
log = logging.getLogger("clips")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s clips %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)
log.propagate = False

_FETCH_LOCK = threading.Lock()
_ENCODER = None


def proxy_from_env(value):
    """(url or None, "on" | "off" | "invalid"). Same shape check as the app's
    app/proxy.py: a supported scheme, a hostname and a port. The value carries
    credentials, so it is never logged or returned in a result; only the
    state is."""
    if not value or not value.strip():
        return None, "off"
    try:
        parsed = urlparse(value.strip())
        ok = parsed.scheme.lower() in PROXY_SCHEMES and bool(parsed.hostname) and bool(parsed.port)
    except ValueError:
        ok = False
    return (value.strip(), "on") if ok else (None, "invalid")


def proxy_fingerprint(value):
    """The first 8 hex chars of sha256 of the stripped value, or None when
    unset. Not the secret and not derivable from it without the secret: the
    operator hashes the value they believe is set and compares
    (printf '%s' "$PROXY_URL" | shasum -a 256 | cut -c1-8)."""
    if not value or not value.strip():
        return None
    return hashlib.sha256(value.strip().encode()).hexdigest()[:8]


# Port rotation (v0.2.5), ported verbatim from the app's app/proxy.py
# (is_valid_proxy_url, get_proxy_url, _parse_proxy_ports,
# _proxy_url_with_port, get_proxy_urls) so PROXY_PORTS means the same thing
# on both sides: PROXY_URL holds the credentials, PROXY_PORTS (optional,
# comma or space separated) the ports to rotate through.

def is_valid_proxy_url(value: str | None) -> bool:
    """Return True when value is a proxy URL shape yt-dlp can use."""
    if not value or not value.strip():
        return False
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return False
    if parsed.scheme.lower() not in PROXY_SCHEMES:
        return False
    if not parsed.hostname or not parsed.port:
        return False
    return True


def get_proxy_url(value: str | None) -> str | None:
    """Return a stripped proxy URL only if it is syntactically usable."""
    if not is_valid_proxy_url(value):
        return None
    return value.strip()


def _parse_proxy_ports(value: str | None) -> list[int]:
    """Parse comma/space separated proxy ports, dropping invalid entries."""
    if not value or not value.strip():
        return []
    ports: list[int] = []
    for raw in re.split(r"[\s,]+", value.strip()):
        if not raw:
            continue
        try:
            port = int(raw)
        except ValueError:
            log.warning("PROXY_PORTS: ignoring non-integer entry %r", raw)
            continue
        if port < 1 or port > 65535:
            log.warning("PROXY_PORTS: ignoring out-of-range port %r", raw)
            continue
        if port not in ports:
            ports.append(port)
    return ports


def _proxy_url_with_port(proxy_url: str, port: int) -> str:
    """Return proxy_url with port replaced while preserving raw credentials."""
    parsed = urlparse(proxy_url.strip())
    raw_auth = ""
    if "@" in parsed.netloc:
        raw_auth = parsed.netloc.rsplit("@", 1)[0] + "@"
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{raw_auth}{host}:{port}"
    return urlunparse(parsed._replace(netloc=netloc))


def get_proxy_urls(value: str | None, ports_value: str | None = None) -> list[str]:
    """Return one or more syntactically valid proxy URLs for yt-dlp.

    PROXY_URL remains the credential/source of truth. Optional PROXY_PORTS
    lets deployments rotate through provider ports without duplicating
    credentials in multiple env vars.
    """
    base = get_proxy_url(value)
    if not base:
        return []
    ports = _parse_proxy_ports(ports_value)
    if not ports:
        return [base]

    urls: list[str] = []
    for port in ports:
        candidate = _proxy_url_with_port(base, port)
        if is_valid_proxy_url(candidate) and candidate not in urls:
            urls.append(candidate)
    return urls or [base]


PROXY_URL, PROXY_STATE = proxy_from_env(os.environ.get("PROXY_URL"))
PROXY_URLS = get_proxy_urls(PROXY_URL, os.environ.get("PROXY_PORTS")) if PROXY_STATE == "on" else []
PROXY_FP = proxy_fingerprint(os.environ.get("PROXY_URL"))
PROXY_PROBE = None      # ok | auth | unreachable | skipped, set once at boot (None: not probed)
PROXY_PROBE_MS = None


class ClipError(Exception):
    def __init__(self, reason, cause=None, detail=None, ports_tried=None):
        super().__init__(reason)
        self.reason = reason
        self.cause = cause
        self.detail = detail
        self.ports_tried = ports_tried


# ── input ────────────────────────────────────────────────────────────────────

def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _url_ok(u):
    if not isinstance(u, str) or not u:
        return False
    p = urlparse(u)
    return p.scheme in ("https", "http", "file") and bool(p.path or p.netloc)


def validate_input(job_input):
    """The parsed job, or ClipError('input')."""
    if not isinstance(job_input, dict):
        raise ClipError("input")
    src = job_input.get("source")
    if not isinstance(src, dict) or src.get("kind") not in ("youtube", "audio"):
        raise ClipError("input")
    start, end = src.get("start"), src.get("end")
    if not _finite(start) or not _finite(end) or start < 0 or end <= start:
        raise ClipError("input")
    if src["kind"] == "youtube" and not _url_ok(src.get("url")):
        raise ClipError("input")
    if src["kind"] == "audio" and not _url_ok(src.get("audio_url")):
        raise ClipError("input")
    # v0.2.1: the app sends the exact cached window it keyed the cache by,
    # so a cached file's first second is the same for every render sharing
    # the key. It must cover the range; absent, the window is derived as before.
    fetch = None
    if src["kind"] == "youtube" and ("fetch_start" in src or "fetch_end" in src):
        fs, fe = src.get("fetch_start"), src.get("fetch_end")
        if not _finite(fs) or not _finite(fe) or fs < 0 or fs > start or fe < end or fe - fs > (end - start) + 3 * PAD_SECS:
            raise ClipError("input")
        fetch = (float(fs), float(fe))
    out = job_input.get("output")
    if not isinstance(out, dict) or not _url_ok(out.get("put_url")) or out.get("container") not in ("mp4", "m4a"):
        raise ClipError("input")
    aspect = job_input.get("aspect")
    if aspect not in ASPECTS:
        raise ClipError("input")
    if aspect == "audio" and out["container"] != "m4a":
        raise ClipError("input")
    if aspect != "audio" and out["container"] != "mp4":
        raise ClipError("input")
    width, height = job_input.get("width"), job_input.get("height")
    if aspect != "audio":
        if not (isinstance(width, int) and isinstance(height, int)) or isinstance(width, bool) or isinstance(height, bool):
            raise ClipError("input")
        if width < 16 or height < 16 or width % 2 or height % 2:
            raise ClipError("input")
    cache = job_input.get("cache") or {}
    if not isinstance(cache, dict):
        raise ClipError("input")
    for k in ("get_url", "put_url"):
        if cache.get(k) is not None and not _url_ok(cache.get(k)):
            raise ClipError("input")
    edits = job_input.get("edits") or {}
    if not isinstance(edits, dict):
        raise ClipError("input")
    removed = normalize_removed(edits.get("removed"), float(end) - float(start))
    captions = job_input.get("captions")
    cues = None
    if captions is not None:
        if not isinstance(captions, dict) or not isinstance(captions.get("cues"), list):
            raise ClipError("input")
        position = captions.get("position", "bottom")
        if position not in CAPTION_ANCHORS:
            raise ClipError("input")
        cues = []
        for c in captions["cues"]:
            if not isinstance(c, dict) or not _finite(c.get("s")) or not _finite(c.get("e")) or c["e"] <= c["s"]:
                raise ClipError("input")
            words = c.get("words")
            if not isinstance(words, list) or not words:
                raise ClipError("input")
            ws = []
            for w in words:
                if not isinstance(w, dict) or not isinstance(w.get("w"), str) or not _finite(w.get("s")) or not _finite(w.get("e")):
                    raise ClipError("input")
                ws.append({"w": w["w"], "s": float(w["s"]), "e": float(w["e"])})
            cues.append({"s": float(c["s"]), "e": float(c["e"]), "words": ws})
    if captions is None:
        position = "bottom"
    identity = None
    raw_identity = job_input.get("identity")
    if raw_identity is not None and src["kind"] == "audio" and aspect != "audio":
        if not isinstance(raw_identity, dict):
            raise ClipError("input")
        identity = {}
        for k in ("church_name", "title", "speaker", "date"):
            v = raw_identity.get(k)
            if v is not None and not isinstance(v, str):
                raise ClipError("input")
            identity[k] = " ".join((v or "").split())[:IDENTITY_TEXT_MAX] or None
        logo = raw_identity.get("logo_url")
        if logo is not None and not _url_ok(logo):
            raise ClipError("input")
        identity["logo_url"] = logo
    max_out = job_input.get("max_output_seconds", 1200)
    if not _finite(max_out) or max_out <= 0:
        raise ClipError("input")
    background = job_input.get("background") or DEFAULT_BACKGROUND
    if not isinstance(background, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", background):
        raise ClipError("input")
    return {
        "kind": src["kind"],
        "url": src.get("url") if src["kind"] == "youtube" else src.get("audio_url"),
        "start": float(start), "end": float(end), "fetch": fetch,
        "cache_get": cache.get("get_url"), "cache_put": cache.get("put_url"), "cache_token": cache.get("token"),
        "out_put": out["put_url"], "out_token": out.get("token"), "container": out["container"],
        "aspect": aspect, "width": width, "height": height,
        "removed": removed, "cues": cues, "caption_position": position, "identity": identity,
        "normalize_audio": bool(job_input.get("normalize_audio", True)),
        "max_output_seconds": float(max_out),
        "background": background,
    }


def normalize_removed(removed, length):
    """Sorted, merged, clipped to [0, length], relative to the clip start."""
    if removed is None:
        return []
    if not isinstance(removed, list):
        raise ClipError("input")
    pairs = []
    for item in removed:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not _finite(item[0]) or not _finite(item[1]):
            raise ClipError("input")
        a, b = max(0.0, float(item[0])), min(float(length), float(item[1]))
        if b > a:
            pairs.append([a, b])
    pairs.sort()
    out = []
    for a, b in pairs:
        if out and a <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


# ── timeline maths (pure) ────────────────────────────────────────────────────

def output_length(length, removed):
    return max(0.0, length - sum(b - a for a, b in removed))


def output_time(t, removed):
    """Source-relative time → output-timeline time (removed time before t collapses)."""
    cut = 0.0
    for a, b in removed:
        if a >= t:
            break
        cut += min(b, t) - a
    return t - cut


def in_removed(t, removed):
    return any(a <= t < b for a, b in removed)


def keep_expr(removed):
    """select/aselect expression keeping every frame outside the removed ranges."""
    if not removed:
        return None
    return "*".join(f"not(between(t,{a:.3f},{b:.3f}))" for a, b in removed)


def crop_filter(width, height):
    """Centre crop to the target aspect, then scale to width×height."""
    return (f"crop='min(iw,ih*{width}/{height})':'min(ih,iw*{height}/{width})',"
            f"scale={width}:{height}:flags=lanczos")


# ── captions (pure layout; Pillow only at draw time) ─────────────────────────

def split_lines(text, max_chars=CAPTION_MAX_CHARS):
    """Up to two lines, split at the space nearest the middle (the page does the same)."""
    text = " ".join(text.split())
    if len(text) <= max_chars // 2:
        return [text] if text else []
    mid = len(text) // 2
    best = None
    for i, ch in enumerate(text):
        if ch == " " and (best is None or abs(i - mid) < abs(best - mid)):
            best = i
    if best is None:
        return [text]
    return [text[:best].strip(), text[best + 1:].strip()]


def caption_states(cues, removed, out_len):
    """One state per spoken-word change, on the OUTPUT clock:
    [{"s", "e", "text", "current"}] with `current` the index of the lit word
    (-1 before the first word). Words inside a removed range are dropped;
    consecutive identical states merge; states are clipped to [0, out_len]."""
    states = []
    for cue in cues or []:
        words = [w for w in cue["words"] if not in_removed(w["s"], removed) and w["w"].strip()]
        if not words:
            continue
        text = " ".join(w["w"].strip() for w in words)
        cs, ce = output_time(cue["s"], removed), output_time(cue["e"], removed)
        ce = min(ce, out_len)
        if ce <= cs:
            continue
        bounds = [cs] + [max(cs, output_time(w["s"], removed)) for w in words] + [ce]
        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            if e <= s:
                continue
            state = {"s": s, "e": e, "text": text, "current": k - 1}
            if states and states[-1]["text"] == text and states[-1]["current"] == state["current"] and abs(states[-1]["e"] - s) < 1e-6:
                states[-1]["e"] = e
            else:
                states.append(state)
    return [st for st in states if st["e"] > st["s"]]


def _font(size, regular=False):
    from PIL import ImageFont
    for path in (FONT_REGULAR_CANDIDATES if regular else FONT_CANDIDATES):
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _line_width(font, words):
    return sum(font.getlength(w) for w in words) + font.getlength(" ") * max(0, len(words) - 1)


def _wrap(words, font, max_w):
    lines, cur = [], []
    for w in words:
        if cur and _line_width(font, cur + [w]) > max_w:
            lines.append(cur)
            cur = [w]
        else:
            cur = cur + [w]
    if cur:
        lines.append(cur)
    return lines


def caption_format(width, height):
    """The output format of a frame size: 9:16, 1:1 or 16:9."""
    if height > width * 1.2:
        return "9:16"
    return "1:1" if width <= height * 1.2 else "16:9"


def caption_font_px(width, height):
    """The format's one caption size."""
    return max(CAPTION_MIN_FONT_PX, int(round(height * CAPTION_FONT_FRAC[caption_format(width, height)])))


def layout_caption(text, width, height):
    """(font_px, font, lines, line_widths) for one caption on a frame of this
    size, at the format's fixed size: the house split (two lines at the
    space nearest the middle) when both halves fit inside width − 2 × side
    margin − the pill's padding, otherwise a greedy wrap, which may take a
    third line. The size only steps down when a single word is wider than
    the frame (or a cue the app did not cut would need a fourth line). v0.2.0
    shrank every long cue, so short cues read large and long ones small."""
    words = text.split()
    margin = int(round(width * CAPTION_SIDE_MARGIN))
    font_px = caption_font_px(width, height)
    while True:
        font = _font(font_px)
        max_w = width - 2 * margin - 2 * int(font_px * 0.6)
        lines = [ln.split() for ln in split_lines(text)]
        if not all(_line_width(font, ln) <= max_w for ln in lines):
            lines = _wrap(words, font, max_w)
        fits = len(lines) <= CAPTION_MAX_LINES and all(_line_width(font, ln) <= max_w for ln in lines)
        if fits or font_px <= CAPTION_MIN_FONT_PX:
            return font_px, font, [" ".join(ln) for ln in lines], [_line_width(font, ln) for ln in lines]
        font_px = max(CAPTION_MIN_FONT_PX, int(font_px * 0.92))


def caption_strip_height(width, height):
    """Every strip is the tallest pill the format can draw (CAPTION_MAX_LINES
    at its fixed size), so the strips make one constant-size video track and
    every caption sits on the same baseline (the strip's bottom)."""
    font_px = caption_font_px(width, height)
    pad_y, gap, line_h = int(font_px * 0.35), int(font_px * 0.22), int(font_px * 1.25)
    return pad_y * 2 + line_h * CAPTION_MAX_LINES + gap * (CAPTION_MAX_LINES - 1)


def draw_caption_png(path, text, current, width, height):
    """Draw one caption state: the pill with up to two centred lines, the
    current word in the accent colour, bottom-aligned in a strip of
    caption_strip_height(). The compositor puts the strip's bottom edge at
    the position's anchor."""
    from PIL import Image, ImageDraw
    font_px, font, lines, line_widths = layout_caption(text, width, height)
    pad_x, pad_y, gap = int(font_px * 0.6), int(font_px * 0.35), int(font_px * 0.22)
    space_w = font.getlength(" ")
    line_h = int(font_px * 1.25)
    strip_h = caption_strip_height(width, height)
    pill_h = pad_y * 2 + line_h * len(lines) + gap * (len(lines) - 1)
    img = Image.new("RGBA", (width, strip_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pill_w = min(width - 4, int(max(line_widths) if line_widths else 0) + pad_x * 2)
    x0 = (width - pill_w) // 2
    top = strip_h - pill_h
    draw.rounded_rectangle([x0, top, x0 + pill_w, strip_h - 1], radius=int(font_px * 0.4), fill=CAPTION_PILL)
    idx = 0
    y = top + pad_y
    for ln, lw in zip(lines, line_widths):
        x = (width - lw) / 2
        for w in ln.split():
            colour = ACCENT if idx == current else CAPTION_TEXT
            draw.text((x, y), w, font=font, fill=colour)
            x += font.getlength(w) + space_w
            idx += 1
        y += line_h + gap
    img.save(path, "PNG")
    return img.size


def write_caption_track(tmp, states, out_len, width, height):
    """The caption track as a dense 30 fps alpha video (qtrle .mov, strip
    height only) built from one PNG per state through the concat demuxer.
    A dense track overlays cleanly; the sparse PNG list used straight as an
    overlay input made ffmpeg end the output early once a cut was present
    (measured 5.1 s of 7). Returns the .mov path, or None with no states."""
    if not states:
        return None
    from PIL import Image
    cap_dir = os.path.join(tmp, "cap")
    os.makedirs(cap_dir, exist_ok=True)
    strip_h = caption_strip_height(width, height)
    blank = os.path.join(cap_dir, "blank.png")
    Image.new("RGBA", (width, strip_h), (0, 0, 0, 0)).save(blank, "PNG")
    entries = []
    cursor = 0.0
    rendered = {}
    for st in states:
        if st["s"] > cursor + 1e-4:
            entries.append((blank, st["s"] - cursor))
        key = (st["text"], st["current"])
        if key not in rendered:
            p = os.path.join(cap_dir, f"s{len(rendered):05d}.png")
            draw_caption_png(p, st["text"], st["current"], width, height)
            rendered[key] = p
        entries.append((rendered[key], st["e"] - max(st["s"], cursor)))
        cursor = st["e"]
    if cursor < out_len:
        entries.append((blank, out_len - cursor))
    lines = ["ffconcat version 1.0"]
    for p, d in entries:
        lines.append(f"file '{p}'")
        lines.append(f"duration {max(0.001, d):.3f}")
    lines.append(f"file '{entries[-1][0]}'")
    list_path = os.path.join(cap_dir, "captions.ffconcat")
    with open(list_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    track = os.path.join(cap_dir, "captions.mov")
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", list_path,
           "-vf", "fps=30,format=argb", "-c:v", "qtrle", "-t", f"{out_len:.3f}", track]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(600, out_len * 10))
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise ClipError("render", cause="caption_track", detail=type(exc).__name__)
    if proc.returncode != 0 or not os.path.exists(track):
        log.info("caption track failed rc=%s", proc.returncode)
        raise ClipError("render", cause="caption_track", detail=_stderr_tail(proc.stderr))
    return track


# ── ffmpeg command (pure) ────────────────────────────────────────────────────

def pick_encoder():
    """h264_nvenc when the ffmpeg here lists it, else libx264. Probed once."""
    global _ENCODER
    if _ENCODER:
        return _ENCODER
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        out = ""
    _ENCODER = "h264_nvenc" if re.search(r"\bh264_nvenc\b", out) else "libx264"
    return _ENCODER


def video_codec_args(encoder, audiogram=False):
    """A quality target under a bitrate ceiling, 2 s keyframes: the
    audiogram's (v0.2.3, ceiling v0.2.4) or the video clip's (v0.2.4)."""
    if audiogram:
        cq, crf, maxrate, bufsize = AUDIOGRAM_NVENC_CQ, AUDIOGRAM_X264_CRF, AUDIOGRAM_MAXRATE, AUDIOGRAM_BUFSIZE
    else:
        cq, crf, maxrate, bufsize = VIDEO_NVENC_CQ, VIDEO_X264_CRF, VIDEO_MAXRATE, VIDEO_BUFSIZE
    cap = ["-maxrate", maxrate, "-bufsize", bufsize, "-profile:v", "high", "-g", GOP_FRAMES]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", cq, "-b:v", "0"] + cap
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", crf] + cap


# ── audiogram identity stack ─────────────────────────────────────────────────
# An audio sermon has no picture, so its video formats draw who and what
# this is on the house background: the church's logo (or its name set in the
# display face), the sermon title, and speaker · date, plus a waveform from
# the clip's audio. The stack never enters the caption band at any of the
# three caption positions (layout below is pure, so the tests can check
# every aspect x position); captions stay the only element with a pill.

def _free(lo, hi, blocked):
    """[lo, hi] minus the blocked (y0, y1) spans, as a list of (y0, y1)."""
    spans = [(lo, hi)]
    for b0, b1 in sorted(blocked):
        nxt = []
        for a0, a1 in spans:
            if b1 <= a0 or b0 >= a1:
                nxt.append((a0, a1))
                continue
            if b0 > a0:
                nxt.append((a0, b0))
            if b1 < a1:
                nxt.append((b1, a1))
        spans = nxt
    return [s for s in spans if s[1] - s[0] > 0]


def _fit_text(text, max_w, start_px, min_px, max_lines, regular=False):
    """(font_px, font, lines): the largest size from start_px down to min_px
    at which the text wraps into max_lines within max_w. One line never
    wraps (max_lines=1) and shrinks instead; at min_px whatever does not fit
    on the last line is cut with an ellipsis rather than overflow."""
    words = (text or "").split()
    if not words:
        return start_px, _font(start_px, regular), []
    px = start_px
    while True:
        font = _font(px, regular)
        lines = [" ".join(ln) for ln in _wrap(words, font, max_w)]
        if len(lines) <= max_lines and all(font.getlength(ln) <= max_w for ln in lines):
            return px, font, lines
        if px <= min_px:
            lines = lines[:max_lines]
            last = lines[-1]
            while last and font.getlength(last + "…") > max_w:
                last = last[:-1].rstrip()
            lines[-1] = last + "…"
            return px, font, lines
        px = max(min_px, int(px * 0.92))


def _text_block(lines, font, px):
    line_h = int(px * 1.22)
    return line_h * len(lines), line_h


def audiogram_layout(width, height, identity, caption_position, captions_on, logo_size=None):
    """Boxes (x0, y0, x1, y1) for one audiogram frame: logo or name, title,
    meta (speaker · date), wave, and the caption band it must avoid. Text
    boxes carry their font and lines for the drawing pass. Per aspect:
      9:16  one centred column (logo, title, meta) from 12 % down;
      1:1   logo top-left, title and meta to its right;
      16:9  logo top-left, title and meta as a lower-left block;
    and the wave (v0.2.2) against the caption band, under it or over it
    where there is room, else across the largest free span that is left.
    When the captions sit where the stack would go, the stack moves past them."""
    identity = identity or {}
    W, H = int(width), int(height)
    side = int(round(W * CAPTION_SIDE_MARGIN))
    tall = H > W * 1.2
    top_safe = int(round(H * (0.12 if tall else 0.07)))
    bottom_safe = int(round(H * 0.07))
    gap = int(round(H * 0.02))
    band = None
    blocked = []
    wave_h = int(H * (WAVE_HEIGHT_FRAC_TALL if tall else WAVE_HEIGHT_FRAC))
    wave_h -= wave_h % 2
    wave_w = (W - 2 * side) - (W - 2 * side) % 2
    wave = None
    if captions_on:
        y1 = int(round(H * CAPTION_ANCHORS.get(caption_position, CAPTION_ANCHORS["bottom"])))
        band = (max(0, y1 - caption_strip_height(W, H)), y1)
        blocked.append((band[0] - gap, band[1] + gap))
        # v0.2.2: the wave hugs the captions, full strip width, a 2 % gap from
        # the band: directly under it where there is room (Top, Middle), else
        # directly over it (Bottom). It is placed before the stack, which then
        # goes round both. v0.2.1 centred it in the largest free span, apart.
        below, above = band[1] + gap, band[0] - gap - wave_h
        wy = below if below + wave_h <= H - bottom_safe else (above if above >= top_safe else None)
        if wy is not None:
            wy -= wy % 2
            wave = (side, wy, side + wave_w, wy + wave_h)
            blocked.append((wy - gap, wy + wave_h + gap))
    meta_text = " · ".join(x for x in (identity.get("speaker"), identity.get("date")) if x)
    title_text = identity.get("title") or ""
    out = {"band": band, "logo": None, "name": None, "title": None, "meta": None, "wave": wave}

    def logo_box(max_w, max_h):
        if not logo_size:
            return None
        lw, lh = logo_size
        scale = min(max_w / lw, max_h / lh, 2.0)
        return max(1, int(lw * scale)), max(1, int(lh * scale))

    def name_block(max_w, px):
        npx, nfont, nlines = _fit_text(identity.get("church_name") or "", max_w, px, max(14, int(px * 0.5)), 2)
        return (npx, nfont, nlines) if nlines else None

    def place(height_needed, from_bottom=False):
        spans = _free(top_safe, H - bottom_safe, blocked)
        spans = [s for s in spans if s[1] - s[0] >= height_needed]
        if not spans:
            return None
        s = spans[-1] if from_bottom else spans[0]
        return s[1] - height_needed if from_bottom else s[0]

    if tall:  # 9:16: one centred column
        col_w = W - 2 * side
        parts = []
        lb = logo_box(W * 0.20, H * 0.09)
        if lb:
            parts.append(("logo", lb[1], lb))
        else:
            nb = name_block(col_w, int(H * 0.034))
            if nb:
                parts.append(("name", _text_block(nb[2], nb[1], nb[0])[0], nb))
        tb = _fit_text(title_text, col_w, int(H * 0.040), int(H * 0.022), 2)
        if tb[2]:
            parts.append(("title", _text_block(tb[2], tb[1], tb[0])[0], tb))
        mb = _fit_text(meta_text, col_w, int(H * 0.022), int(H * 0.014), 1, regular=True)
        if mb[2]:
            parts.append(("meta", _text_block(mb[2], mb[1], mb[0])[0], mb))
        inner = int(H * 0.018)
        total = sum(p[1] for p in parts) + inner * max(0, len(parts) - 1)
        y = place(total)
        if y is None:
            y = top_safe
        for key, h, data in parts:
            if key == "logo":
                out["logo"] = ((W - data[0]) // 2, y, (W - data[0]) // 2 + data[0], y + data[1])
            else:
                width_used = int(max(data[1].getlength(ln) for ln in data[2]))
                out[key] = {"box": ((W - width_used) // 2, y, (W + width_used) // 2, y + h),
                            "px": data[0], "lines": data[2], "align": "center", "regular": key == "meta"}
            y += h + inner
    else:
        square = W <= H * 1.2
        lb = logo_box(W * (0.18 if square else 0.12), H * (0.14 if square else 0.12))
        logo_w = lb[0] if lb else 0
        if square:  # 1:1: one row, logo then the text to its right
            text_x = side + (logo_w + int(W * 0.03) if lb else 0)
            text_w = W - side - text_x
            tb = _fit_text(title_text, text_w, int(H * 0.048), int(H * 0.026), 2)
            mb = _fit_text(meta_text, text_w, int(H * 0.028), int(H * 0.016), 1, regular=True)
            nb = None if lb else name_block(text_w, int(H * 0.040))
            blocks = [b for b in (("name", nb), ("title", tb), ("meta", mb)) if b[1] and b[1][2]]
            inner = int(H * 0.012)
            text_h = sum(_text_block(b[1][2], b[1][1], b[1][0])[0] for b in blocks) + inner * max(0, len(blocks) - 1)
            row_h = max(lb[1] if lb else 0, text_h)
            y = place(row_h)
            if y is None:
                y = top_safe
            if lb:
                out["logo"] = (side, y + (row_h - lb[1]) // 2, side + lb[0], y + (row_h - lb[1]) // 2 + lb[1])
            ty = y + (row_h - text_h) // 2
            for key, data in blocks:
                h = _text_block(data[2], data[1], data[0])[0]
                width_used = int(max(data[1].getlength(ln) for ln in data[2]))
                out[key] = {"box": (text_x, ty, text_x + width_used, ty + h), "px": data[0], "lines": data[2],
                            "align": "left", "regular": key == "meta"}
                ty += h + inner
        else:  # 16:9: logo (or name) top-left, title and meta lower-left
            if lb:
                y = place(lb[1])
                y = top_safe if y is None else y
                out["logo"] = (side, y, side + lb[0], y + lb[1])
                blocked.append((y - gap, y + lb[1] + gap))
            else:
                nb = name_block(int(W * 0.45), int(H * 0.050))
                if nb:
                    h = _text_block(nb[2], nb[1], nb[0])[0]
                    y = place(h)
                    y = top_safe if y is None else y
                    width_used = int(max(nb[1].getlength(ln) for ln in nb[2]))
                    out["name"] = {"box": (side, y, side + width_used, y + h), "px": nb[0], "lines": nb[2],
                                   "align": "left", "regular": False}
                    blocked.append((y - gap, y + h + gap))
            text_w = int(W * 0.6)
            tb = _fit_text(title_text, text_w, int(H * 0.055), int(H * 0.030), 2)
            mb = _fit_text(meta_text, text_w, int(H * 0.032), int(H * 0.018), 1, regular=True)
            blocks = [b for b in (("title", tb), ("meta", mb)) if b[1][2]]
            inner = int(H * 0.012)
            text_h = sum(_text_block(b[1][2], b[1][1], b[1][0])[0] for b in blocks) + inner * max(0, len(blocks) - 1)
            y = place(text_h, from_bottom=True)
            y = (H - bottom_safe - text_h) if y is None else y
            for key, data in blocks:
                h = _text_block(data[2], data[1], data[0])[0]
                width_used = int(max(data[1].getlength(ln) for ln in data[2]))
                out[key] = {"box": (side, y, side + width_used, y + h), "px": data[0], "lines": data[2],
                            "align": "left", "regular": key == "meta"}
                y += h + inner

    if out["wave"]:
        return out
    # No captions (or no room by them): the wave across the largest free span left.
    taken = list(blocked)
    for key in ("logo", "name", "title", "meta"):
        v = out[key]
        if v:
            b = v if key == "logo" else v["box"]
            taken.append((b[1] - gap, b[3] + gap))
    spans = sorted(_free(top_safe, H - bottom_safe, taken), key=lambda s: s[1] - s[0], reverse=True)
    if spans and spans[0][1] - spans[0][0] >= int(H * 0.05):
        s0, s1 = spans[0]
        wh = min(wave_h, s1 - s0)
        wh -= wh % 2
        wy = s0 + (s1 - s0 - wh) // 2
        wy -= wy % 2
        out["wave"] = (side, wy, side + wave_w, wy + wh)
    return out


def fetch_logo(url, tmp):
    """The org's logo to a local file, or None. A logo that cannot be had
    (expired link, not an image, too large) never fails the render: the
    church name takes its place."""
    if not url:
        return None
    path = os.path.join(tmp, "logo.img")
    try:
        download(url, path)
        if os.path.getsize(path) > LOGO_MAX_BYTES:
            return None
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return path
    except Exception as exc:
        log.info("logo skipped type=%s", type(exc).__name__)
        return None


def draw_identity_background(path, width, height, background, identity, caption_position, captions_on, logo_path=None):
    """The audiogram's still frame: house colour, a soft radial glow, then
    the identity stack from audiogram_layout. Returns the layout (its wave
    box goes to ffmpeg)."""
    from PIL import Image, ImageDraw, ImageFilter
    W, H = int(width), int(height)
    rgb = tuple(int(background.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    img = Image.new("RGB", (W, H), rgb)
    # A soft radial glow, drawn small and scaled up: cheap and band-free enough.
    small = Image.new("RGB", (max(8, W // 8), max(8, H // 8)), rgb)
    d = ImageDraw.Draw(small)
    lift = tuple(min(255, c + 22) for c in rgb)
    sw, sh = small.size
    d.ellipse([sw * 0.05, sh * 0.25, sw * 0.95, sh * 0.75], fill=lift)
    small = small.filter(ImageFilter.GaussianBlur(radius=max(2, sw // 5)))
    img = small.resize((W, H), Image.BICUBIC)
    logo = None
    logo_size = None
    if logo_path:
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo_size = logo.size
        except Exception:
            logo = None
    layout = audiogram_layout(W, H, identity, caption_position, captions_on, logo_size)
    canvas = img.convert("RGBA")
    if logo is not None and layout["logo"]:
        x0, y0, x1, y1 = layout["logo"]
        canvas.alpha_composite(logo.resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))
    draw = ImageDraw.Draw(canvas)
    muted = (246, 243, 236, 150)
    for key in ("name", "title", "meta"):
        block = layout[key]
        if not block:
            continue
        font = _font(block["px"], block["regular"])
        line_h = int(block["px"] * 1.22)
        x0, y0, x1, _ = block["box"]
        for i, ln in enumerate(block["lines"]):
            lw = font.getlength(ln)
            x = x0 + (x1 - x0 - lw) / 2 if block["align"] == "center" else x0
            draw.text((x, y0 + i * line_h), ln, font=font, fill=muted if key == "meta" else CAPTION_TEXT)
    canvas.convert("RGB").save(path, "PNG")
    return layout


def caption_y(position, width, height):
    """The strip's top row: its bottom edge (the pill's bottom) on the anchor."""
    anchor = CAPTION_ANCHORS.get(position, CAPTION_ANCHORS["bottom"])
    return max(0, int(round(height * anchor)) - caption_strip_height(width, height))


def build_ffmpeg_cmd(job, source_path, file_origin, out_path, caption_track, encoder):
    """The full argv. `file_origin` is the source clock time at the file's
    first sample (the padded range start for a fetched YouTube range, 0 for
    the stored audio file).

    The range is cut on the INPUT side (-ss / -t before -i): with a
    re-encode that is frame-accurate, and the timestamps restart at 0, which
    is what the sparse caption track is timed against. A `trim` +
    `setpts=PTS-STARTPTS` in the graph instead made framesync (overlay) end
    the output early (measured 5.1 s of 7 with audio mapped), so it is not
    used here."""
    start, end = job["start"], job["end"]
    off = start - file_origin
    length = end - start
    removed = job["removed"]
    out_len = output_length(length, removed)
    keep = keep_expr(removed)
    cut_chain = [f"aselect='{keep}'", "asetpts=N/SR/TB"] if keep else []
    level_chain = ["loudnorm=I=-16:TP=-1.5:LRA=11"] if job["normalize_audio"] else []
    audiogram = job["kind"] == "audio" and job["aspect"] != "audio"
    audio_src = "1:a" if audiogram else "0:a"
    wave = job.get("wave_box") if audiogram else None
    if wave:
        # The waveform reads the cut audio before loudnorm (which resamples).
        graph = [f"[{audio_src}]" + ",".join(cut_chain + ["asplit=2"]) + "[ap][aw]",
                 "[ap]" + ",".join(level_chain or ["anull"]) + "[a]",
                 f"[aw]{WAVE_LEVEL},showwaves=s={wave[2] - wave[0]}x{wave[3] - wave[1]}:mode=cline:scale=sqrt:rate=30:colors={WAVE_COLOUR},format=rgba[wave]"]
    else:
        graph = [f"[{audio_src}]" + ",".join(cut_chain + level_chain or ["anull"]) + "[a]"]
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-nostdin"]
    source_in = ["-ss", f"{off:.3f}", "-t", f"{length:.3f}", "-i", source_path]

    if job["aspect"] == "audio":
        cmd += source_in + ["-filter_complex", ";".join(graph), "-map", "[a]", "-vn",
                            "-c:a", "aac", "-b:a", "128k", "-ar", AUDIO_SAMPLE_RATE, "-movflags", "+faststart", "-t", f"{out_len:.3f}", out_path]
        return cmd

    width, height = job["width"], job["height"]
    if audiogram:
        # Audiogram: the identity background (house colour, gradient, logo,
        # title, speaker · date) held for the length of the output, or a
        # solid colour when there is none; the waveform goes over it.
        if job.get("identity_bg"):
            cmd += ["-loop", "1", "-framerate", "30", "-t", f"{out_len:.3f}", "-i", job["identity_bg"]] + source_in
        else:
            colour = "0x" + job["background"].lstrip("#")
            cmd += ["-f", "lavfi", "-i", f"color=c={colour}:s={width}x{height}:r=30:d={out_len:.3f}"] + source_in
        video_chain = ["format=rgba"] if wave else ["format=yuv420p"]
        cap_input_index = 2
    else:
        cmd += source_in
        video_chain = []
        if keep:
            video_chain += [f"select='{keep}'", "setpts=N/FRAME_RATE/TB"]
        video_chain += [crop_filter(width, height), "format=yuv420p"]
        cap_input_index = 1
    base = "[0:v]" + ",".join(video_chain)
    if wave:
        base = f"{base}[bg0];[bg0][wave]overlay=x={wave[0]}:y={wave[1]}:eof_action=pass,format=yuv420p"
    if caption_track:
        cmd += ["-i", caption_track]
        cap_y = caption_y(job.get("caption_position", "bottom"), width, height)
        graph.append(base + "[vb]")
        graph.append(f"[{cap_input_index}:v]format=rgba[cap]")
        # yuv420 out of the overlay, then an explicit format: libx264 refuses
        # the alpha-carrying format the overlay would otherwise pick.
        # eof_action=pass, never shortest=1; -t caps the length.
        graph.append(f"[vb][cap]overlay=x=(W-w)/2:y={cap_y}:format=yuv420:eof_action=pass,format=yuv420p[v]")
    else:
        graph.append(base + "[v]")
    cmd += ["-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]"]
    cmd += video_codec_args(encoder, audiogram)
    cmd += ["-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", AUDIO_SAMPLE_RATE, "-movflags", "+faststart",
            "-t", f"{out_len:.3f}", out_path]
    return cmd


# ── transfers ────────────────────────────────────────────────────────────────

def _with_token(url, token):
    if not token or "token=" in url:
        return url
    return url + ("&" if "?" in url else "?") + "token=" + token


def download(url, dest, timeout=DOWNLOAD_TIMEOUT):
    """GET to a file (file:// is for local tests only). Raises ClipError('fetch')."""
    p = urlparse(url)
    if p.scheme == "file":
        try:
            shutil.copyfile(p.path, dest)
            size = os.path.getsize(dest)
        except OSError as exc:
            raise ClipError("fetch", cause="source_network", detail=type(exc).__name__)
        if size == 0:
            raise ClipError("fetch", cause="source_empty")
        return size
    total = 0
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                raise ClipError("fetch", cause="source_http", detail=f"HTTP {resp.status_code}")
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(DOWNLOAD_CHUNK):
                    total += len(chunk)
                    if total > MAX_SOURCE_BYTES:
                        raise ClipError("fetch", cause="source_too_big")
                    fh.write(chunk)
    except requests.RequestException as exc:
        # the class only: the message quotes the signed URL
        raise ClipError("fetch", cause="source_network", detail=type(exc).__name__)
    if total == 0:
        raise ClipError("fetch", cause="source_empty")
    return total


def upload(url, token, path, content_type, reason):
    """PUT a file to a signed upload URL (Supabase: token in the query, x-upsert)."""
    p = urlparse(url)
    if p.scheme == "file":
        try:
            shutil.copyfile(path, p.path)
            return
        except OSError as exc:
            raise ClipError(reason, cause="upload_network", detail=type(exc).__name__)
    try:
        with open(path, "rb") as fh:
            resp = requests.put(_with_token(url, token), data=fh, timeout=UPLOAD_TIMEOUT,
                                headers={"Content-Type": content_type, "x-upsert": "true"})
    except requests.RequestException as exc:
        raise ClipError(reason, cause="upload_network", detail=type(exc).__name__)
    if resp.status_code >= 300:
        raise ClipError(reason, cause="upload_http", detail=f"HTTP {resp.status_code}")


# ── fetch ────────────────────────────────────────────────────────────────────

def padded_range(start, end, pad=PAD_SECS):
    """The range fetched from YouTube: ±pad, never below 0, rounded outward
    to whole seconds so the cache key the app computes lines up."""
    return math.floor(max(0.0, start - pad)), math.ceil(end + pad)


def proxy_secrets(proxy_url):
    """Every fragment of the proxy URL a message could quote: the URL, its
    username, its password (as written and percent-decoded) and its host.
    Longest first, so the full URL is replaced before its parts."""
    if not proxy_url:
        return ()
    parsed = urlparse(proxy_url)
    parts = {proxy_url, parsed.hostname or ""}
    for part in (parsed.username, parsed.password):
        if part:
            parts.update({part, unquote(part)})
    return tuple(sorted((p for p in parts if p), key=len, reverse=True))


def proxy_secrets_all(urls):
    """proxy_secrets over every rotation URL: the user, password and host are
    the same for all of them; only each full URL (its port) adds an entry."""
    parts = set()
    for u in urls:
        parts.update(proxy_secrets(u))
    return tuple(sorted(parts, key=len, reverse=True))


def redact(msg, secrets):
    """One line, no proxy fragment, at most YTDLP_MESSAGE_CAP characters."""
    text = " ".join(str(msg).split())
    for secret in secrets:
        text = text.replace(secret, "[proxy]")
    return text[:YTDLP_MESSAGE_CAP]


def _stderr_tail(stderr):
    """ffmpeg's last stderr line (it runs at -loglevel error; the inputs are
    local temp files, captions are pictures), one line, capped; or None."""
    lines = [ln for ln in (stderr or "").splitlines() if ln.strip()]
    return redact(lines[-1], ()) if lines else None


def _ffreport_path(out_path, attempt):
    """Where attempt n's ffmpeg report goes (v0.2.6): the job's temp dir, beside
    the range. None when the path holds a character FFREPORT reads as syntax
    (':' separates its options; quotes and backslashes escape), so a strange
    temp dir costs the report, never a mis-parsed one."""
    path = os.path.join(os.path.dirname(os.path.abspath(out_path)), f"ffmpeg-fetch-{attempt}.log")
    return None if any(c in path for c in ":,'\\") else path


# A URL in an ffmpeg line keeps its scheme and host; the path and query go.
# ffmpeg 4.4 (the image's) writes the input URL first ("<url>: Connection
# refused"), and a signed googlevideo URL is longer than YTDLP_MESSAGE_CAP: kept
# whole, the cap would cut the error and leave the URL, whose numbers
# (dur=407.2) could also match a cause pattern.
_URL_TAIL = re.compile(r"\b([a-z][a-z0-9+.-]*://[^/\s?#]+)\S*?(?=:\s|:$|\s|$)", re.IGNORECASE)


def _report_tail(path, secrets):
    """The last non-empty line of an ffmpeg report, URLs cut to their host,
    redacted; or None. The report opens with "Command line:" and the argv on
    the next line, which carries -http_proxy and the media URL: everything up
    to it is dropped, so a report with no error line has no tail rather than
    the argv."""
    try:
        with open(path, errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    if "Command line:" in lines:
        lines = lines[lines.index("Command line:") + 2:]
    lines = [ln for ln in lines if ln.strip()]
    return redact(_URL_TAIL.sub(r"\1", lines[-1]), secrets) if lines else None


def _with_tail(message, tail):
    """message | ffmpeg: tail, within YTDLP_MESSAGE_CAP; the message is cut
    first, so ffmpeg's own line survives."""
    suffix = f" | ffmpeg: {tail}"[:YTDLP_MESSAGE_CAP]
    return message[:max(0, YTDLP_MESSAGE_CAP - len(suffix))] + suffix


# Fetch causes (v0.2.5), first match wins, on the REDACTED message: the most
# specific first, so a wording change downgrades to fetch_other, never to a
# wrong word. "Names the proxy" means the message says proxy (a quoted URL
# reads [proxy] by then).
_AGE = r"confirm your age|age[- ]restricted"
FETCH_CAUSES = (
    ("proxy_auth", r"\b407\b"),
    ("proxy_unreachable", r"unable to connect to proxy|tunnel connection failed|proxyerror"
                          r"|(?=.*proxy).*(connection refused|connection reset|timed out)"),
    ("blocked", r"sign in to confirm(?! your age)|not a bot|http error 403|http error 429"),
    ("unavailable", r"private video|video unavailable|this video has been removed|not available in your country"
                    r"|" + _AGE + r"|this live event|premieres"),
    ("no_format", r"requested format is not available|no video formats found|signature solving failed"
                  r"|n challenge solving failed"),
    ("fetch_timeout", r"timed out"),
)


def classify_fetch(message):
    """The cause word for a failed YouTube fetch from its redacted message."""
    low = (message or "").lower()
    for cause, pattern in FETCH_CAUSES:
        if re.search(pattern, low):
            return cause
    return "fetch_other"


class _RedactingYdlLogger:
    """yt-dlp's own messages never reach stderr unredacted: a bot-wall line or
    a proxy 407 can quote the proxy URL, which carries credentials. Warnings
    and errors are logged with every proxy fragment replaced by [proxy];
    debug and info are dropped."""

    def __init__(self, secrets):
        self.secrets = secrets

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        log.info("yt-dlp warning: %s", redact(msg, self.secrets))

    def error(self, msg):
        log.info("yt-dlp error: %s", redact(msg, self.secrets))


def ytdlp_options(out_path, range_start, range_end, proxy=None, socket_timeout=SOCKET_TIMEOUT):
    """yt-dlp options for one range. The encoder arguments go in
    external_downloader_args (FFmpegFD reads those; postprocessor_args is
    ignored with download_ranges) AND merge_output_format pins mp4: without
    the pin the container defaults to webm, which rejects H.264 and forces a
    VP9 software transcode of every fetch (SermonVault §5). `proxy` is this
    attempt's proxy URL (PROXY_URL when not given)."""
    import yt_dlp
    proxy = proxy or PROXY_URL
    opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best",
        "download_ranges": yt_dlp.utils.download_range_func(None, [(range_start, range_end)]),
        "force_keyframes_at_cuts": True,
        "external_downloader_args": {"ffmpeg_o": ["-c:v", "libx264", "-crf", "19", "-preset", "veryfast", "-c:a", "aac"]},
        "merge_output_format": "mp4",
        "outtmpl": out_path,
        "quiet": True, "no_warnings": True, "noprogress": True,
        "retries": 3, "fragment_retries": 3,
        "noplaylist": True,
        "socket_timeout": socket_timeout,
        "logger": _RedactingYdlLogger(proxy_secrets_all([proxy] + list(PROXY_URLS))),
    }
    # Before every -i: a media socket idle this long ends ffmpeg (microseconds).
    ffmpeg_i = ["-rw_timeout", str(int(socket_timeout * 1_000_000))]
    if proxy:
        opts["proxy"] = proxy
        # With download_ranges FFmpegFD fetches the media; yt-dlp hands it
        # the proxy only as env vars, so pass it as ffmpeg's own input option
        # too (placed before every -i). The media URL is bound to the IP that
        # resolved it, so ffmpeg must leave through the same proxy.
        ffmpeg_i += ["-http_proxy", proxy]
    opts["external_downloader_args"]["ffmpeg_i"] = ffmpeg_i
    return opts


def _now():
    return time.monotonic()


def fetch_youtube_range(url, out_path, range_start, range_end):
    """Download the range; returns ports_tried (distinct proxy ports used).
    Attempt n leaves through urls[(n - 1) % len(urls)]: yt-dlp's requests
    and ffmpeg's -http_proxy change together, since the media URL is bound
    to the IP that resolved it."""
    import yt_dlp
    urls = PROXY_URLS or [PROXY_URL]
    secrets = proxy_secrets_all(urls)
    deadline = _now() + FETCH_WALL_SECS
    used = []
    last = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        remaining = deadline - _now()
        if remaining <= 0:
            # Cut between attempts. The last message (if any) stays as the detail.
            raise ClipError("fetch", cause="fetch_timeout", ports_tried=len(used),
                            detail=last or f"no file within FETCH_WALL_SECS={FETCH_WALL_SECS:g}")
        proxy = urls[(attempt - 1) % len(urls)]
        if proxy not in used:
            used.append(proxy)
        # an attempt's own stall budget never runs past the wall clock
        opts = ytdlp_options(out_path, range_start, range_end, proxy=proxy,
                             socket_timeout=max(1.0, min(SOCKET_TIMEOUT, remaining)))
        # v0.2.6: FFmpegFD runs ffmpeg at -loglevel quiet with our environment,
        # so FFREPORT is the only way to hear why the media download died. The
        # worker takes one job at a time and fetches under _FETCH_LOCK, so no
        # other ffmpeg (a render above all) can see the variable while it is set.
        report = _ffreport_path(out_path, attempt)
        try:
            if report:
                os.environ["FFREPORT"] = f"file={report}:level=16"   # 16: errors only
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
            finally:
                os.environ.pop("FFREPORT", None)
            break
        except Exception as exc:
            # redacted first (it can quote the proxy URL), classified on what the app may see
            last = redact(exc, secrets)
            # ffmpeg's own error line goes in BEFORE classification: its "407"
            # or "Connection refused" is the cause "ffmpeg exited with code 1" hides
            tail = _report_tail(report, secrets) if report else None
            if tail:
                last = _with_tail(last, tail)
            cause = classify_fetch(last)
            # type and cause only: a DownloadError message can quote the proxy URL
            log.info("yt-dlp failed type=%s attempt=%d port=%s cause=%s", type(exc).__name__, attempt,
                     urlparse(proxy).port if proxy else "-", cause)
            if attempt == FETCH_ATTEMPTS:
                raise ClipError("fetch", cause=cause, detail=last, ports_tried=len(used))
            if os.path.exists(out_path):
                os.remove(out_path)  # a half-written file would be skipped as "already downloaded"
            if cause == "proxy_auth":
                # v0.2.6: every port checks the same login, so the next one is refused too
                raise ClipError("fetch", cause=cause, detail=last, ports_tried=len(used))
            time.sleep(min(FETCH_RETRY_SLEEP, max(0.0, deadline - _now())))
        finally:
            # the report echoes the argv, proxy URL included: never left on disk
            if report and os.path.exists(report):
                os.remove(report)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise ClipError("fetch", cause="fetch_other", detail="yt-dlp wrote no file", ports_tried=len(used))
    return len(used)


def fetch_source(job, tmp):
    """(path, file_origin, cache_hit, bytes, ports_tried). YouTube: cache GET,
    else yt-dlp then cache PUT (ports_tried 0 on a cache hit). Audio: the
    signed stored-audio URL (its own clock, origin 0; ports_tried None)."""
    if job["kind"] == "audio":
        path = os.path.join(tmp, "source.audio")
        size = download(job["url"], path)
        return path, 0.0, False, size, None
    if PROXY_STATE != "on":
        # v0.2.5: no proxy is not a configuration (YouTube walls datacenter
        # IPs). Fail before the cache GET too, so a misconfigured endpoint
        # shows on the first job, not on the first cache miss.
        raise ClipError("fetch", cause="proxy_config",
                        detail="PROXY_URL is not set" if PROXY_STATE == "off" else "PROXY_URL is invalid")
    rs, re_ = job.get("fetch") or padded_range(job["start"], job["end"])
    path = os.path.join(tmp, "source.mp4")
    if job["cache_get"]:
        try:
            size = download(job["cache_get"], path)
            return path, float(rs), True, size, 0
        except ClipError:
            log.info("cache miss")
    with _FETCH_LOCK:
        ports_tried = fetch_youtube_range(job["url"], path, rs, re_)
    size = os.path.getsize(path)
    if job["cache_put"]:
        try:
            upload(job["cache_put"], job["cache_token"], path, "video/mp4", "cache")
        except ClipError:
            log.info("cache put failed; rendering anyway")
    return path, float(rs), False, size, ports_tried


# ── render ───────────────────────────────────────────────────────────────────

def probe_duration(path):
    try:
        out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return round(float(out), 3)
    except Exception:
        return None


def render(job, source_path, file_origin, tmp):
    """Returns (out_path, encoder). Raises ClipError('render' | 'too_long')."""
    length = job["end"] - job["start"]
    out_len = output_length(length, job["removed"])
    if out_len > job["max_output_seconds"] + 1e-6:
        raise ClipError("too_long")
    if out_len < 0.5:
        raise ClipError("input", cause="too_short")
    caption_track = None
    if job["cues"] and job["aspect"] != "audio":
        states = caption_states(job["cues"], job["removed"], out_len)
        caption_track = write_caption_track(tmp, states, out_len, job["width"], job["height"])
        log.info("captions cues=%d states=%d", len(job["cues"]), len(states))
    if job["kind"] == "audio" and job["aspect"] != "audio" and job.get("identity") is not None:
        try:
            bg = os.path.join(tmp, "identity.png")
            layout = draw_identity_background(bg, job["width"], job["height"], job["background"], job["identity"],
                                              job["caption_position"], bool(caption_track),
                                              fetch_logo(job["identity"].get("logo_url"), tmp))
            job = dict(job, identity_bg=bg, wave_box=layout["wave"])
        except Exception as exc:
            # The plain background is still a render; the stack is not worth a failure.
            log.warning("identity skipped type=%s", type(exc).__name__)
    encoder = "none" if job["aspect"] == "audio" else pick_encoder()
    out_path = os.path.join(tmp, "out." + job["container"])
    cmd = build_ffmpeg_cmd(job, source_path, file_origin, out_path, caption_track, encoder)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=job["max_output_seconds"] * 6 + 600)
    except subprocess.TimeoutExpired:
        raise ClipError("timeout", cause="encode_timeout")
    except OSError as exc:
        raise ClipError("render", cause="encode", detail=type(exc).__name__)
    if proc.returncode != 0 and encoder == "h264_nvenc":
        # A GPU encoder that is listed but cannot open (driver, session cap)
        # falls back to software for this job only.
        log.warning("nvenc failed rc=%s; retrying with libx264", proc.returncode)
        encoder = "libx264"
        cmd = build_ffmpeg_cmd(job, source_path, file_origin, out_path, caption_track, encoder)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=job["max_output_seconds"] * 6 + 600)
        except subprocess.TimeoutExpired:
            raise ClipError("timeout", cause="encode_timeout")
    if proc.returncode != 0:
        log.info("ffmpeg failed rc=%s stderr_lines=%d", proc.returncode, len((proc.stderr or "").splitlines()))
        raise ClipError("render", cause="encode", detail=_stderr_tail(proc.stderr))
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        log.info("ffmpeg wrote nothing rc=0")
        raise ClipError("render", cause="encode_empty")
    return out_path, encoder


# ── the job ──────────────────────────────────────────────────────────────────

def _diagnostics(cause=None, cause_detail=None):
    """v0.2.5: on every result. `reason` keeps its eight words (the app maps
    them); these are new fields the app ignores until it reads them."""
    return {"cause": cause, "cause_detail": cause_detail, "proxy_probe": PROXY_PROBE, "proxy_fp": PROXY_FP}


def _failure(reason, started, cause=None, cause_detail=None, **extra):
    out = {"ok": False, "reason": reason, "bytes": None, "duration": None, "fetch_ms": None,
           "render_ms": None, "exec_ms": int((time.monotonic() - started) * 1000), "cache_hit": None,
           "encoder": None, "image_tag": IMAGE_TAG}
    out.update(_diagnostics(cause, cause_detail))
    out.update(extra)
    return out


def run(job_input):
    started = time.monotonic()
    tmp = None
    fetch_ms = render_ms = None
    cache_hit = ports_tried = None
    try:
        try:
            job = validate_input(job_input)
        except ClipError as exc:
            exc.cause = exc.cause or "input"
            raise
        tmp = tempfile.mkdtemp(prefix="clip-")
        t0 = time.monotonic()
        source_path, origin, cache_hit, src_bytes, ports_tried = fetch_source(job, tmp)
        fetch_ms = int((time.monotonic() - t0) * 1000)
        t1 = time.monotonic()
        out_path, encoder = render(job, source_path, origin, tmp)
        render_ms = int((time.monotonic() - t1) * 1000)
        size = os.path.getsize(out_path)
        duration = probe_duration(out_path)
        content_type = "audio/mp4" if job["container"] == "m4a" else "video/mp4"
        upload(job["out_put"], job["out_token"], out_path, content_type, "upload")
        exec_ms = int((time.monotonic() - started) * 1000)
        log.info("rendered kind=%s aspect=%s cache_hit=%s source_bytes=%d out_bytes=%d duration=%s removed=%d "
                 "captions=%s encoder=%s fetch_ms=%d render_ms=%d exec_ms=%d ports_tried=%s",
                 job["kind"], job["aspect"], cache_hit, src_bytes, size, duration, len(job["removed"]),
                 bool(job["cues"]), encoder, fetch_ms, render_ms, exec_ms, ports_tried)
        return {"ok": True, "reason": None, "bytes": size, "duration": duration, "fetch_ms": fetch_ms,
                "render_ms": render_ms, "exec_ms": exec_ms, "cache_hit": cache_hit, "encoder": encoder,
                "image_tag": IMAGE_TAG, **_diagnostics(), "ports_tried": ports_tried}
    except ClipError as exc:
        if exc.ports_tried is not None:
            ports_tried = exc.ports_tried
        log.info("not rendered reason=%s cause=%s", exc.reason, exc.cause)
        return _failure(exc.reason, started, exc.cause, exc.detail,
                        fetch_ms=fetch_ms, render_ms=render_ms, cache_hit=cache_hit, ports_tried=ports_tried)
    except Exception as exc:  # never raise to RunPod
        log.warning("not rendered reason=error type=%s", type(exc).__name__)
        return _failure("error", started, type(exc).__name__,
                        fetch_ms=fetch_ms, render_ms=render_ms, cache_hit=cache_hit, ports_tried=ports_tried)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def probe_proxy(proxy_url, state):
    """(result, ms): one GET of PROXY_PROBE_URL through the proxy, no retries.
    An HTTPS GET through an HTTP proxy meets a 407 at the CONNECT, so requests
    raises ProxyError ("Tunnel connection failed: 407 ...") instead of
    returning the status: both shapes are read. Nothing about the exception
    is logged or kept beyond the word."""
    if state != "on":
        return "skipped", None
    t0 = time.monotonic()
    try:
        with requests.get(PROXY_PROBE_URL, proxies={"http": proxy_url, "https": proxy_url},
                          timeout=PROXY_PROBE_TIMEOUT) as resp:
            status = resp.status_code
        result = "ok" if status == 200 else "auth" if status == 407 else "unreachable"
    except requests.exceptions.ProxyError as exc:
        result = "auth" if re.search(r"\b407\b", str(exc)) else "unreachable"
    except Exception:
        result = "unreachable"
    return result, int((time.monotonic() - t0) * 1000)


def _first_line(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""
    return (out.splitlines() or [""])[0]


def tool_versions():
    """{ytdlp, ffmpeg, runpod, js_runtime}: what this image actually holds."""
    try:
        import yt_dlp.version
        ytdlp = yt_dlp.version.__version__
    except Exception:
        ytdlp = "none"
    try:
        import runpod
        rp = getattr(runpod, "__version__", "unknown")
    except Exception:
        rp = "none"
    ff = _first_line([FFMPEG, "-version"]).split()          # "ffmpeg version 6.1.1 Copyright ..."
    deno = _first_line(["deno", "--version"]).split()       # "deno 2.9.5 (stable, release, ...)"
    return {"ytdlp": ytdlp, "ffmpeg": ff[2] if len(ff) > 2 and ff[1] == "version" else "none", "runpod": rp,
            "js_runtime": " ".join(deno[:2]) if len(deno) >= 2 and deno[0] == "deno" else "none"}


def boot_line(encoder, versions, proxy_urls, state, fp, probe, probe_ms):
    """The one boot line. Host and user are not secret (the README's
    standard: the username shows in the vendor's dashboard); the password
    and the URL never appear, only the fingerprint. proxy_host is the first
    rotation port; proxy_ports how many there are."""
    host = user = "-"
    if state == "on" and proxy_urls:
        parsed = urlparse(proxy_urls[0])
        host = f"{parsed.hostname}:{parsed.port}"
        user = parsed.username or "-"
    return (f"clip worker {IMAGE_TAG} encoder={encoder} ytdlp={versions['ytdlp']} ffmpeg={versions['ffmpeg']} "
            f"runpod={versions['runpod']} js_runtime={versions['js_runtime']} proxy={state} proxy_host={host} "
            f"proxy_ports={len(proxy_urls) if state == 'on' else 0} proxy_user={user} proxy_fp={fp or '-'} proxy_probe={probe or '-'} "
            f"probe_ms={'-' if probe_ms is None else probe_ms}")


def log_startup():
    """Probe the proxy (once; every result carries the word), then log the boot line."""
    global PROXY_PROBE, PROXY_PROBE_MS
    urls = PROXY_URLS or ([PROXY_URL] if PROXY_URL else [])
    PROXY_PROBE, PROXY_PROBE_MS = probe_proxy(urls[0] if urls else None, PROXY_STATE)
    log.info("%s", boot_line(_ENCODER, tool_versions(), urls, PROXY_STATE, PROXY_FP, PROXY_PROBE, PROXY_PROBE_MS))


def handler(job):
    return run((job or {}).get("input"))


if __name__ == "__main__":
    import runpod

    pick_encoder()  # probe before taking jobs
    log_startup()
    runpod.serverless.start({"handler": handler})
