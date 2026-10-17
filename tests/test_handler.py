"""Handler shape tests. The network, yt-dlp's process spawn and ffmpeg are
faked; `-m real` runs one real libx264 render of samples/sample-10s.mp4."""

import json
import os
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import handler  # noqa: E402

SAMPLE = os.path.join(ROOT, "samples", "sample-10s.mp4")
SECRET_PROXY = "http://user-sp1:s3cr3t%2Bpw@gate.example.test:10001"


@pytest.fixture(autouse=True)
def proxy_on(monkeypatch):
    """v0.2.5: a YouTube job without a valid PROXY_URL fails fast, so every
    test runs as a configured endpoint unless it says otherwise. The proxy
    is fictitious; no test reaches it (fetches are faked or file://)."""
    monkeypatch.setattr(handler, "PROXY_URL", SECRET_PROXY)
    monkeypatch.setattr(handler, "PROXY_STATE", "on")
    monkeypatch.setattr(handler, "PROXY_FP", handler.proxy_fingerprint(SECRET_PROXY))
    monkeypatch.setattr(handler, "PROXY_PROBE", "ok")
    monkeypatch.setattr(handler, "PROXY_PROBE_MS", 120)


def _job(**over):
    job = {
        "source": {"kind": "youtube", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "start": 100.0, "end": 108.0},
        "cache": {"get_url": None, "put_url": None, "token": None},
        "output": {"put_url": "https://supabase.test/storage/v1/object/upload/sign/clips/1/2/3/4.mp4?token=T", "token": "T", "container": "mp4"},
        "aspect": "9:16", "width": 1080, "height": 1920,
        "captions": {"cues": [
            {"s": 0.0, "e": 2.4, "words": [{"w": "This", "s": 0.0, "e": 0.3}, {"w": "is", "s": 0.3, "e": 0.5},
                                           {"w": "a", "s": 0.5, "e": 0.6}, {"w": "sample", "s": 0.6, "e": 1.1}, {"w": "render.", "s": 1.1, "e": 1.6}]},
            {"s": 2.6, "e": 5.9, "words": [{"w": "John", "s": 2.6, "e": 2.9}, {"w": "3:16", "s": 2.9, "e": 3.4}, {"w": "stays", "s": 3.4, "e": 3.7},
                                           {"w": "on", "s": 3.7, "e": 3.8}, {"w": "one", "s": 3.8, "e": 4.0}, {"w": "cue.", "s": 4.0, "e": 4.4}]},
        ]},
        "edits": {"removed": [[6.0, 7.0]]},
        "normalize_audio": True,
        "max_output_seconds": 1200,
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(job.get(k), dict):
            job[k] = {**job[k], **v}
        else:
            job[k] = v
    return job


# ── input ─────────────────────────────────────────────────────────────────────

def test_validate_accepts_the_contract():
    job = handler.validate_input(_job())
    assert job["kind"] == "youtube" and job["start"] == 100.0 and job["end"] == 108.0
    assert job["removed"] == [[6.0, 7.0]] and len(job["cues"]) == 2 and job["container"] == "mp4"


@pytest.mark.parametrize("bad", [
    None, {}, {"source": {"kind": "vimeo"}},
    _job(source={"kind": "youtube", "url": "https://x/y", "start": 10, "end": 10}),
    _job(source={"kind": "audio", "start": 1, "end": 5}),
    _job(output={"container": "mkv"}),
    _job(aspect="4:3"),
    _job(aspect="audio"),                         # mp4 container with audio aspect
    _job(width=1081),                             # odd
    _job(edits={"removed": [[1, "x"]]}),
    _job(captions={"cues": [{"s": 1, "e": 0, "words": [{"w": "a", "s": 0, "e": 1}]}]}),
    _job(max_output_seconds=0),
    _job(background="navy"),
])
def test_validate_rejects_bad_input(bad):
    with pytest.raises(handler.ClipError) as exc:
        handler.validate_input(bad)
    assert exc.value.reason == "input"


def test_removed_is_merged_sorted_and_clipped():
    assert handler.normalize_removed([[5, 6], [1, 2.5], [2, 3], [-1, 0.5], [7, 99]], 8.0) == [[0.0, 0.5], [1.0, 3.0], [5.0, 6.0], [7.0, 8.0]]


# ── timeline maths ────────────────────────────────────────────────────────────

def test_output_time_collapses_removed_ranges():
    removed = [[1.0, 2.0], [4.0, 4.5]]
    assert handler.output_time(0.5, removed) == 0.5
    assert handler.output_time(2.0, removed) == 1.0
    assert handler.output_time(3.0, removed) == 2.0
    assert handler.output_time(5.0, removed) == 3.5
    assert handler.output_length(8.0, removed) == 6.5


def test_keep_expression_and_crop():
    assert handler.keep_expr([[6.0, 7.0], [7.5, 7.75]]) == "not(between(t,6.000,7.000))*not(between(t,7.500,7.750))"
    assert handler.keep_expr([]) is None
    assert handler.crop_filter(1080, 1920).startswith("crop='min(iw,ih*1080/1920)':'min(ih,iw*1920/1080)',scale=1080:1920")


def test_caption_states_follow_the_spoken_word_on_the_output_clock():
    job = handler.validate_input(_job(edits={"removed": [[0.4, 0.55]]}))
    states = handler.caption_states(job["cues"], job["removed"], 7.85)
    # "a" starts at 0.5, inside the cut [0.4, 0.55) → dropped from the text;
    # "is" (0.3) and "sample" (0.6) stay.
    assert states[0]["text"] == "This is sample render."
    assert [s["current"] for s in states[:4]] == [0, 1, 2, 3]
    # "sample" starts at 0.6 on the source clock = 0.45 on the output clock.
    assert states[2]["s"] == pytest.approx(handler.output_time(0.6, job["removed"])) == pytest.approx(0.45)
    assert all(s["e"] > s["s"] for s in states)
    assert any(s["text"].startswith("John 3:16") for s in states)


def test_split_lines_breaks_near_the_middle():
    assert handler.split_lines("short") == ["short"]
    lines = handler.split_lines("For God so loved the world that he gave his only Son see John 3:16 and more")
    assert len(lines) == 2 and abs(len(lines[0]) - len(lines[1])) < 12


# ── yt-dlp: the arguments must reach the process ─────────────────────────────

def test_ytdlp_options_carry_both_halves_of_the_fix():
    opts = handler.ytdlp_options("/tmp/x.mp4", 90, 118)
    assert opts["external_downloader_args"]["ffmpeg_o"] == ["-c:v", "libx264", "-crf", "19", "-preset", "veryfast", "-c:a", "aac"]
    assert opts["merge_output_format"] == "mp4"
    assert opts["force_keyframes_at_cuts"] is True
    assert opts["format"] == "bestvideo[height<=1080]+bestaudio/best"
    assert "postprocessor_args" not in opts
    assert handler.padded_range(100.3, 108.2) == (90, 119)
    assert handler.padded_range(4.0, 20.0) == (0, 30)



@pytest.mark.parametrize("value,state", [
    (SECRET_PROXY, "on"), ("socks5h://u:p@gate.example.test:7000", "on"),
    (None, "off"), ("  ", "off"),
    ("gate.example.test:10001", "invalid"), ("ftp://u:p@gate.example.test:21", "invalid"),
    ("http://u:p@gate.example.test", "invalid"), ("http://u:p@gate.example.test:notaport", "invalid"),
])
def test_ytdlp_options_carry_the_proxy_only_when_valid(monkeypatch, value, state):
    url, got = handler.proxy_from_env(value)
    assert got == state
    monkeypatch.setattr(handler, "PROXY_URL", url)
    opts = handler.ytdlp_options("/tmp/x.mp4", 90, 118)
    if state == "on":
        assert opts["proxy"] == value
        assert opts["external_downloader_args"]["ffmpeg_i"] == ["-rw_timeout", "20000000", "-http_proxy", value]
    else:
        assert "proxy" not in opts
        assert opts["external_downloader_args"]["ffmpeg_i"] == ["-rw_timeout", "20000000"]
    # the fix's two halves are untouched either way
    assert opts["external_downloader_args"]["ffmpeg_o"] == ["-c:v", "libx264", "-crf", "19", "-preset", "veryfast", "-c:a", "aac"]
    assert opts["merge_output_format"] == "mp4"


def test_ytdlp_messages_are_logged_redacted_and_capped(monkeypatch, capsys):
    import logging
    lines = []

    class Grab(logging.Handler):
        def emit(self, record):
            lines.append(self.format(record))

    monkeypatch.setattr(handler, "PROXY_URL", SECRET_PROXY)
    logger = handler.ytdlp_options("/tmp/x.mp4", 90, 118)["logger"]
    grab = Grab()
    handler.log.addHandler(grab)
    try:
        logger.warning(f"WARNING: url={SECRET_PROXY}\n pw=s3cr3t%2Bpw decoded=s3cr3t+pw user=user-sp1 host=gate.example.test")
        logger.error("ERROR: " + "x" * 2048)
        logger.info(f"[download] via {SECRET_PROXY}")
        logger.debug(f"[debug] via {SECRET_PROXY}")
    finally:
        handler.log.removeHandler(grab)
    assert lines[0] == ("yt-dlp warning: WARNING: url=[proxy] pw=[proxy] decoded=[proxy] "
                        "user=[proxy] host=[proxy]")
    assert lines[1] == "yt-dlp error: " + ("ERROR: " + "x" * 2048)[:300]
    assert len(lines) == 2
    for part in ("s3cr3t", "user-sp1", "gate.example.test"):
        assert part not in "\n".join(lines)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    # no proxy: nothing to redact, the cap still applies
    assert handler.redact("a\n  b " + "y" * 500, handler.proxy_secrets(None)) == ("a b " + "y" * 500)[:300]


def _fake_youtubedl(failures, calls):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def download(self, urls):
            calls.append(dict(self.opts))
            if len(calls) <= failures:
                import yt_dlp
                self.opts["logger"].error(f"ERROR: [youtube] x: Sign in to confirm you're not a bot (via {self.opts.get('proxy')})")
                raise yt_dlp.utils.DownloadError(f"ERROR: Sign in to confirm you're not a bot (via {self.opts.get('proxy')})")
            shutil.copyfile(SAMPLE, self.opts["outtmpl"])
    return FakeYDL


def test_fetch_retries_once_then_reports_fetch(monkeypatch, tmp_path):
    import yt_dlp
    monkeypatch.setattr(handler, "FETCH_RETRY_SLEEP", 0)
    monkeypatch.setattr(handler, "PROXY_URL", SECRET_PROXY)

    calls = []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _fake_youtubedl(1, calls))
    out = tmp_path / "once.mp4"
    handler.fetch_youtube_range("https://www.youtube.com/watch?v=dQw4w9WgXcQ", str(out), 90, 118)
    assert len(calls) == 2 and out.stat().st_size > 0
    assert calls[1]["proxy"] == SECRET_PROXY  # same options on the retry

    calls = []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _fake_youtubedl(2, calls))
    with pytest.raises(handler.ClipError) as err:
        handler.fetch_youtube_range("https://www.youtube.com/watch?v=dQw4w9WgXcQ", str(tmp_path / "twice.mp4"), 90, 118)
    assert err.value.reason == "fetch" and len(calls) == 2


def test_proxy_url_never_reaches_a_log_line(monkeypatch, tmp_path, capsys):
    import logging
    import yt_dlp
    lines = []

    class Grab(logging.Handler):
        def emit(self, record):
            lines.append(self.format(record))

    grab = Grab()
    handler.log.addHandler(grab)
    try:
        monkeypatch.setattr(handler, "FETCH_RETRY_SLEEP", 0)
        monkeypatch.setattr(handler, "PROXY_URL", SECRET_PROXY)
        monkeypatch.setattr(handler, "PROXY_STATE", "on")
        monkeypatch.setattr(handler, "_ENCODER", "libx264")
        monkeypatch.setattr(handler, "probe_proxy", lambda url, state: ("auth", 41))
        handler.log_startup()
        monkeypatch.setattr(yt_dlp, "YoutubeDL", _fake_youtubedl(2, []))
        result = handler.run(_job())
    finally:
        handler.log.removeHandler(grab)
    boot = lines.pop(0)
    assert boot.startswith(f"clip worker {handler.IMAGE_TAG} encoder=libx264 ") and " proxy=on " in boot
    # host and user are shown in the boot line ONLY (not secret); never the password or the URL
    assert "proxy_host=gate.example.test:10001 proxy_user=user-sp1 " in boot and "proxy_probe=auth probe_ms=41" in boot
    assert "s3cr3t" not in boot and SECRET_PROXY not in boot
    assert any("yt-dlp failed type=DownloadError attempt=1" in ln for ln in lines)
    assert any("yt-dlp failed type=DownloadError attempt=2" in ln for ln in lines)
    assert result["ok"] is False and result["reason"] == "fetch"
    assert any("yt-dlp error: ERROR: [youtube] x: Sign in to confirm you're not a bot (via [proxy])" in ln for ln in lines)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    blob = "\n".join(lines) + json.dumps(result)
    for part in ("s3cr3t", "user-sp1", "gate.example.test", SECRET_PROXY):
        assert part not in blob


def test_real_youtubedl_routes_its_messages_to_our_logger(monkeypatch, capsys):
    """The real YoutubeDL, not a fake: its error, warning and screen output
    must go through the logger option and leave stderr and stdout empty."""
    import logging
    import yt_dlp
    lines = []

    class Grab(logging.Handler):
        def emit(self, record):
            lines.append(self.format(record))

    monkeypatch.setattr(handler, "PROXY_URL", SECRET_PROXY)
    opts = handler.ytdlp_options("/tmp/x.mp4", 90, 118)
    assert "logger" in opts and opts["proxy"] == SECRET_PROXY
    grab = Grab()
    handler.log.addHandler(grab)
    try:
        ydl = yt_dlp.YoutubeDL(opts)
        ydl.to_screen(f"[youtube] fetching via {SECRET_PROXY}")
        ydl.report_warning(f"proxy {SECRET_PROXY} slow")
        with pytest.raises(yt_dlp.utils.DownloadError):
            ydl.report_error(f"407 Proxy Authentication Required {SECRET_PROXY}")
    finally:
        handler.log.removeHandler(grab)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert lines == ["yt-dlp warning: proxy [proxy] slow",
                     "yt-dlp error: ERROR: 407 Proxy Authentication Required [proxy]"]


@pytest.mark.parametrize("proxy", [None, SECRET_PROXY])
def test_ffmpegfd_spawns_our_encoder_args_and_an_mp4(tmp_path, monkeypatch, proxy):
    """The real FFmpegFD builds its argv from our options; only the spawn is
    faked. This is the SermonVault §5 lesson: a value reaching the process
    is not a value taking effect, so read the args back out of the argv."""
    import yt_dlp
    from yt_dlp.downloader.external import FFmpegFD

    spawned = {}

    class FakePopen:
        def __init__(self, args, **kw):
            spawned["args"] = list(args)
            self.stdin = None
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def wait(self): return 0

    monkeypatch.setattr(handler, "PROXY_URL", proxy)
    opts = handler.ytdlp_options(str(tmp_path / "source.%(ext)s"), 90, 119)
    ydl = yt_dlp.YoutubeDL(opts)
    fd = FFmpegFD(ydl, ydl.params)
    info = {"url": "https://example.test/video.webm", "ext": "mp4", "protocol": "https",
            "section_start": 90, "section_end": 119,
            "requested_formats": [{"url": "https://example.test/v", "ext": "mp4", "protocol": "https"},
                                  {"url": "https://example.test/a", "ext": "m4a", "protocol": "https"}]}
    with patch("yt_dlp.downloader.external.Popen", FakePopen):
        fd._call_downloader(str(tmp_path / "source.mp4"), info)
    args = spawned["args"]
    joined = " ".join(args)
    assert "-c:v libx264 -crf 19 -preset veryfast -c:a aac" in joined, joined
    assert "-f mp4" in joined, joined
    assert args[-1].endswith("source.mp4")
    assert "-c copy" not in joined  # force_keyframes_at_cuts re-encodes at the cut
    if proxy:
        # before every input: the video and the audio URL are both IP-bound
        inputs = [i for i, a in enumerate(args) if a == "-i"]
        assert len(inputs) == 2
        for i in inputs:
            assert args[i - 2:i] == ["-http_proxy", proxy], args
    else:
        assert "-http_proxy" not in args
    # v0.2.5: a stalled media socket ends ffmpeg after 20 s, on every input
    for i in [i for i, a in enumerate(args) if a == "-i"]:
        assert "-rw_timeout" in args[:i] and args[args.index("-rw_timeout") + 1] == "20000000"
    assert args.count("-rw_timeout") == args.count("-i")


# ── ffmpeg command ────────────────────────────────────────────────────────────

def test_video_command_trims_cuts_crops_and_never_stream_copies():
    job = handler.validate_input(_job())
    cmd = handler.build_ffmpeg_cmd(job, "/t/source.mp4", 90.0, "/t/out.mp4", "/t/cap/captions.mov", "libx264")
    graph = cmd[cmd.index("-filter_complex") + 1]
    i = cmd.index("-i")
    assert cmd[i - 4:i] == ["-ss", "10.000", "-t", "8.000"], cmd  # input-side, frame-accurate on a re-encode
    assert "trim=" not in graph
    assert "select='not(between(t,6.000,7.000))'" in graph and "aselect='not(between(t,6.000,7.000))'" in graph
    assert "crop='min(iw,ih*1080/1920)'" in graph and "scale=1080:1920" in graph
    # bottom: the strip's bottom edge 12 % above the frame's (1690 − the three-line strip at 65 px)
    assert f"overlay=x=(W-w)/2:y={1690 - handler.caption_strip_height(1080, 1920)}:format=yuv420:eof_action=pass" in graph and "shortest" not in graph and "loudnorm" in graph
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "libx264" and "copy" not in cmd
    assert cmd[-2] == "7.000" and cmd[-3] == "-t"
    assert "/t/cap/captions.mov" in cmd and "concat" not in cmd


def test_nvenc_args_when_chosen():
    job = handler.validate_input(_job(normalize_audio=False, captions=None))
    cmd = handler.build_ffmpeg_cmd(job, "/t/source.mp4", 90.0, "/t/out.mp4", None, "h264_nvenc")
    assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
    assert "loudnorm" not in " ".join(cmd) and "overlay" not in " ".join(cmd)


def test_audio_aspect_writes_m4a_without_video():
    job = handler.validate_input(_job(aspect="audio", output={"container": "m4a"}, width=None, height=None))
    cmd = handler.build_ffmpeg_cmd(job, "/t/source.mp4", 90.0, "/t/out.m4a", None, "none")
    assert "-vn" in cmd and cmd[-1].endswith(".m4a") and "-c:v" not in cmd


def test_audiogram_uses_a_colour_source_for_an_audio_sermon():
    job = handler.validate_input(_job(source={"kind": "audio", "audio_url": "https://supabase.test/a.mp3", "start": 100.0, "end": 108.0}, aspect="1:1", width=1080, height=1080))
    cmd = handler.build_ffmpeg_cmd(job, "/t/source.audio", 0.0, "/t/out.mp4", None, "libx264")
    assert "lavfi" in cmd and "color=c=0x0f1115:s=1080x1080:r=30:d=7.000" in cmd
    i = cmd.index("/t/source.audio")
    assert cmd[i - 5:i] == ["-ss", "100.000", "-t", "8.000", "-i"]
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.startswith("[1:a]aselect=")


# ── run(): outcomes, temp dir, reasons ───────────────────────────────────────

@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(handler, "pick_encoder", lambda: "libx264")
    monkeypatch.setattr(handler, "probe_duration", lambda p: 7.0)
    made = []
    real_mkdtemp = handler.tempfile.mkdtemp
    def mk(prefix="clip-"):
        d = real_mkdtemp(prefix=prefix); made.append(d); return d
    monkeypatch.setattr(handler.tempfile, "mkdtemp", mk)
    return made


def _fake_ffmpeg_ok(cmd, **kw):
    with open(cmd[-1], "wb") as fh:
        fh.write(b"\x00" * 1000)
    return subprocess.CompletedProcess(cmd, 0, "", "")


def test_cache_hit_render_upload_and_cleanup(quiet, monkeypatch):
    calls = {"download": [], "upload": []}
    def fake_download(url, dest, timeout=None):
        calls["download"].append(url); shutil.copyfile(SAMPLE, dest); return os.path.getsize(dest)
    def fake_upload(url, token, path, content_type, reason):
        calls["upload"].append((url, token, content_type, reason, os.path.getsize(path)))
    monkeypatch.setattr(handler, "download", fake_download)
    monkeypatch.setattr(handler, "upload", fake_upload)
    monkeypatch.setattr(handler.subprocess, "run", _fake_ffmpeg_ok)
    monkeypatch.setattr(handler, "fetch_youtube_range", lambda *a: pytest.fail("cache hit must not fetch"))

    out = handler.run(_job(cache={"get_url": "https://supabase.test/cache/x.mp4?token=C"}))
    assert out["ok"] is True and out["reason"] is None
    assert out["cache_hit"] is True and out["encoder"] == "libx264" and out["duration"] == 7.0 and out["bytes"] == 1000
    assert out["image_tag"] == handler.IMAGE_TAG and out["fetch_ms"] is not None and out["render_ms"] is not None
    assert calls["download"] == ["https://supabase.test/cache/x.mp4?token=C"]
    assert calls["upload"][0][2:4] == ("video/mp4", "upload")
    assert not os.path.exists(quiet[0])


def test_cache_miss_fetches_once_then_puts_the_range(quiet, monkeypatch):
    events = []
    def fake_download(url, dest, timeout=None):
        events.append(("get", url)); raise handler.ClipError("fetch")
    def fake_fetch(url, out_path, rs, re_):
        events.append(("ytdlp", url, rs, re_)); shutil.copyfile(SAMPLE, out_path)
    def fake_upload(url, token, path, content_type, reason):
        events.append(("put", reason, content_type))
    monkeypatch.setattr(handler, "download", fake_download)
    monkeypatch.setattr(handler, "fetch_youtube_range", fake_fetch)
    monkeypatch.setattr(handler, "upload", fake_upload)
    monkeypatch.setattr(handler.subprocess, "run", _fake_ffmpeg_ok)

    out = handler.run(_job(cache={"get_url": "https://supabase.test/cache/x.mp4?token=C", "put_url": "https://supabase.test/cache/put", "token": "P"}))
    assert out["ok"] is True and out["cache_hit"] is False
    assert events == [("get", "https://supabase.test/cache/x.mp4?token=C"),
                      ("ytdlp", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", 90, 118),
                      ("put", "cache", "video/mp4"), ("put", "upload", "video/mp4")]


@pytest.mark.parametrize("stage,reason", [("fetch", "fetch"), ("render", "render"), ("upload", "upload")])
def test_failures_report_their_reason_and_clean_up(quiet, monkeypatch, stage, reason):
    def fake_download(url, dest, timeout=None):
        if stage == "fetch": raise handler.ClipError("fetch")
        shutil.copyfile(SAMPLE, dest); return 1
    def fake_ffmpeg(cmd, **kw):
        if stage == "render": return subprocess.CompletedProcess(cmd, 1, "", "boom")
        return _fake_ffmpeg_ok(cmd, **kw)
    def fake_upload(url, token, path, content_type, r):
        if stage == "upload": raise handler.ClipError("upload")
    monkeypatch.setattr(handler, "download", fake_download)
    monkeypatch.setattr(handler.subprocess, "run", fake_ffmpeg)
    monkeypatch.setattr(handler, "upload", fake_upload)
    out = handler.run(_job(cache={"get_url": "https://supabase.test/c.mp4"}, source={"kind": "audio", "audio_url": "https://supabase.test/a.mp3", "start": 1.0, "end": 9.0}))
    assert out["ok"] is False and out["reason"] == reason
    assert not os.path.exists(quiet[0])


def test_too_long_is_refused_before_ffmpeg(quiet, monkeypatch):
    monkeypatch.setattr(handler, "download", lambda url, dest, timeout=None: (shutil.copyfile(SAMPLE, dest), 1)[1])
    monkeypatch.setattr(handler.subprocess, "run", lambda *a, **k: pytest.fail("must not render"))
    out = handler.run(_job(cache={"get_url": "https://supabase.test/c.mp4"}, max_output_seconds=5))
    assert out["ok"] is False and out["reason"] == "too_long"
    assert not os.path.exists(quiet[0])


def test_unexpected_exception_never_escapes(quiet, monkeypatch):
    monkeypatch.setattr(handler, "download", lambda *a, **k: 1 / 0)
    out = handler.run(_job(cache={"get_url": "https://supabase.test/c.mp4"}))
    assert out == {**out, "ok": False, "reason": "error"}
    assert not os.path.exists(quiet[0])


def test_bad_input_is_input_and_makes_no_temp_dir(quiet):
    out = handler.run({"source": {"kind": "youtube"}})
    assert out["ok"] is False and out["reason"] == "input" and quiet == []


def test_handler_wraps_run():
    assert handler.handler({"input": None})["reason"] == "input"
    assert handler.handler(None)["reason"] == "input"


# ── one real render ───────────────────────────────────────────────────────────

@pytest.mark.real
@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg needed")
def test_real_libx264_render_of_the_sample(tmp_path, monkeypatch):
    """8 s of the 10 s sample as a 9:16 captioned mp4 with one 1 s cut
    (libx264), through the real fetch (file:// cache hit), render and upload
    (file:// target). Output kept under out/ for inspection."""
    monkeypatch.setattr(handler, "pick_encoder", lambda: "libx264")
    out_dir = os.path.join(ROOT, "out")
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "sample-9x16.mp4")
    if os.path.exists(target):
        os.remove(target)
    job = _job(source={"kind": "youtube", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "start": 1.0, "end": 9.0},
               cache={"get_url": "file://" + SAMPLE, "put_url": None, "token": None},
               output={"put_url": "file://" + target, "token": None, "container": "mp4"})
    out = handler.run(job)
    assert out["ok"] is True, out
    assert out["encoder"] == "libx264" and out["cache_hit"] is True
    assert os.path.exists(target) and out["bytes"] == os.path.getsize(target) > 10_000
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,codec_name:format=duration", "-of", "json", target],
                           capture_output=True, text=True).stdout
    info = json.loads(probe)
    assert info["streams"][0]["width"] == 1080 and info["streams"][0]["height"] == 1920
    assert info["streams"][0]["codec_name"] == "h264"
    assert abs(float(info["format"]["duration"]) - 7.0) < 0.3


# ── caption position and layout (v0.2.0) ────────────────────────────────────

LONG_CUE = "Draw near to God and he will draw near to you, that we are to humble ourselves before God"


def test_caption_anchors_match_the_app():
    assert handler.CAPTION_ANCHORS == {"bottom": 0.88, "middle": 0.56, "top": 0.24}
    app_captions = os.path.join(os.path.dirname(ROOT), "sermount-web", "app", "clip_captions.py")
    if os.path.exists(app_captions):  # the sibling checkout, when there is one
        assert 'CAPTION_ANCHORS = {"bottom": 0.88, "middle": 0.56, "top": 0.24}' in open(app_captions).read()


@pytest.mark.parametrize("position,anchor", [("bottom", 0.88), ("middle", 0.56), ("top", 0.24)])
@pytest.mark.parametrize("aspect,width,height,kind", [("9:16", 1080, 1920, "youtube"), ("1:1", 1080, 1080, "youtube"),
                                                      ("16:9", 1920, 1080, "youtube"), ("9:16", 1080, 1920, "audio")])
def test_caption_strip_sits_on_the_position_anchor_for_video_and_audiogram(position, anchor, aspect, width, height, kind):
    source = ({"kind": "youtube", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "start": 100.0, "end": 108.0} if kind == "youtube"
              else {"kind": "audio", "audio_url": "https://supabase.test/a.mp3?token=T", "start": 100.0, "end": 108.0})
    job = handler.validate_input(_job(source=source, aspect=aspect, width=width, height=height,
                                      captions={"cues": _job()["captions"]["cues"], "position": position}))
    assert job["caption_position"] == position
    cmd = handler.build_ffmpeg_cmd(job, "/tmp/src.mp4", 90.0, "/tmp/out.mp4", "/tmp/cap.mov", "libx264")
    graph = cmd[cmd.index("-filter_complex") + 1]
    y = int(round(height * anchor)) - handler.caption_strip_height(width, height)
    assert f"[cap]overlay=x=(W-w)/2:y={y}:" in graph
    assert 0 <= y and y + handler.caption_strip_height(width, height) <= height


def test_caption_position_is_validated_and_defaults_to_bottom():
    assert handler.validate_input(_job())["caption_position"] == "bottom"
    with pytest.raises(handler.ClipError):
        handler.validate_input(_job(captions={"cues": _job()["captions"]["cues"], "position": "left"}))


@pytest.mark.parametrize("width,height", [(1080, 1920), (1080, 1080), (1920, 1080)])
@pytest.mark.parametrize("text", [LONG_CUE, "short line", "Supercalifragilisticexpialidocious words everywhere " * 2])
def test_no_caption_line_exceeds_the_frame_less_the_margin(width, height, text):
    """The 9:16 overflow (render 7, 2026-10-12): every laid-out line fits the
    frame width minus both side margins, in at most three lines (two is the
    app's target; three the fallback)."""
    font_px, font, lines, widths = handler.layout_caption(" ".join(text.split()), width, height)
    margin = int(round(width * handler.CAPTION_SIDE_MARGIN))
    assert 1 <= len(lines) <= 3
    assert all(w <= width - 2 * margin for w in widths), (widths, width)
    assert " ".join(lines).split() == text.split()   # nothing dropped


def test_drawn_caption_pill_stays_inside_the_9x16_frame(tmp_path):
    from PIL import Image
    png = tmp_path / "cap.png"
    w, h = handler.draw_caption_png(str(png), LONG_CUE, 3, 1080, 1920)
    alpha = Image.open(png).split()[-1]
    box = alpha.getbbox()
    margin = int(round(1080 * handler.CAPTION_SIDE_MARGIN))
    assert box[0] >= margin - 1 and box[2] <= 1080 - margin + 1, box


# ── audiogram identity stack (v0.2.0) ────────────────────────────────────────

IDENTITY = {"logo_url": None, "church_name": "Grace Community Church",
            "title": "Draw Near: Humility and the Presence of God in an Anxious Week",
            "speaker": "Pastor Ann Lee", "date": "October 5, 2026"}


def _boxes(layout):
    out = {}
    for key in ("logo", "name", "title", "meta", "wave"):
        v = layout[key]
        if v:
            out[key] = v if key in ("logo", "wave") else v["box"]
    return out


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@pytest.mark.parametrize("width,height", [(1080, 1920), (1080, 1080), (1920, 1080)])
@pytest.mark.parametrize("position", ["bottom", "middle", "top"])
@pytest.mark.parametrize("logo_size", [(600, 300), (300, 900), None])
def test_identity_stack_stays_in_the_frame_and_out_of_the_caption_band(width, height, position, logo_size):
    layout = handler.audiogram_layout(width, height, IDENTITY, position, True, logo_size)
    boxes = _boxes(layout)
    assert ("logo" in boxes) == bool(logo_size) and ("name" in boxes) == (not logo_size)
    assert "title" in boxes and "meta" in boxes and "wave" in boxes
    band = layout["band"]
    for key, b in boxes.items():
        assert 0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height, (key, b)
        assert not (b[1] < band[1] and band[0] < b[3]), (key, b, band)       # never in the caption band
    keys = list(boxes)
    for i, a in enumerate(keys):
        for c in keys[i + 1:]:
            assert not _overlap(boxes[a], boxes[c]), (a, boxes[a], c, boxes[c])
    if logo_size:  # never stretched
        x0, y0, x1, y1 = boxes["logo"]
        assert abs((x1 - x0) / (y1 - y0) - logo_size[0] / logo_size[1]) < 0.02
    if height > width * 1.2 and position != "top":
        assert boxes.get("logo", boxes.get("name"))[1] >= int(round(height * 0.12))   # clear of platform overlays
    meta = layout["meta"]
    assert len(meta["lines"]) == 1   # speaker · date never wraps
    assert len(layout["title"]["lines"]) <= 2


def test_identity_without_captions_and_without_text_still_lays_out():
    layout = handler.audiogram_layout(1080, 1920, {"church_name": None, "title": None, "speaker": None, "date": None},
                                      "bottom", False, None)
    assert layout["band"] is None and layout["title"] is None and layout["wave"]


def test_identity_is_validated_and_only_for_audiograms():
    audio = {"kind": "audio", "audio_url": "https://supabase.test/a.mp3?token=T", "start": 1.0, "end": 8.0}
    job = handler.validate_input(_job(source=audio, identity=dict(IDENTITY, logo_url="https://supabase.test/logos/1/logo.png?token=T")))
    assert job["identity"]["title"] == IDENTITY["title"] and job["identity"]["logo_url"].startswith("https://")
    assert handler.validate_input(_job(identity=IDENTITY))["identity"] is None       # a YouTube render ignores it
    with pytest.raises(handler.ClipError):
        handler.validate_input(_job(source=audio, identity=dict(IDENTITY, title=7)))
    with pytest.raises(handler.ClipError):
        handler.validate_input(_job(source=audio, identity=dict(IDENTITY, logo_url="ftp://x/logo.png")))


def test_audiogram_command_holds_the_identity_frame_and_overlays_the_waveform():
    audio = {"kind": "audio", "audio_url": "https://supabase.test/a.mp3?token=T", "start": 1.0, "end": 8.0}
    job = dict(handler.validate_input(_job(source=audio, identity=IDENTITY)), identity_bg="/tmp/identity.png",
               wave_box=(64, 900, 1016, 1168))
    cmd = handler.build_ffmpeg_cmd(job, "/tmp/a.m4a", 0.0, "/tmp/out.mp4", "/tmp/cap.mov", "libx264")
    i = cmd.index("/tmp/identity.png")
    assert cmd[i - 7:i + 1] == ["-loop", "1", "-framerate", "30", "-t", "6.000", "-i", "/tmp/identity.png"]  # 7 s less the 1 s cut
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "asplit=2[ap][aw]" in graph and "showwaves=s=952x268:mode=cline" in graph
    assert "[bg0][wave]overlay=x=64:y=900" in graph and "lavfi" not in cmd
    # the waveform reads the audio before loudnorm; the mapped audio is still levelled
    assert graph.index("asplit") < graph.index("loudnorm") and "[ap]loudnorm" in graph


@pytest.mark.real
def test_real_audiogram_render_draws_the_logo_and_the_title(tmp_path, monkeypatch):
    """A 9:16 audiogram from the sample's audio with a generated logo, then a
    frame read back: brass pixels where the layout put the logo, light text
    pixels where it put the title, the house colour elsewhere."""
    from PIL import Image, ImageDraw
    monkeypatch.setattr(handler, "pick_encoder", lambda: "libx264")
    audio = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SAMPLE, "-vn", "-c:a", "aac", str(audio)], check=True)
    logo = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    ImageDraw.Draw(logo).ellipse([0, 0, 399, 399], fill=(184, 132, 46, 255))
    logo_path = tmp_path / "logo.png"
    logo.save(logo_path)
    target = tmp_path / "ag.mp4"
    job = _job(source={"kind": "audio", "audio_url": "file://" + str(audio), "start": 1.0, "end": 8.0},
               output={"put_url": "file://" + str(target), "token": None, "container": "mp4"},
               identity=dict(IDENTITY, logo_url="file://" + str(logo_path)))
    out = handler.run(job)
    assert out["ok"] is True, out
    frame = tmp_path / "f.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "2", "-i", str(target), "-frames:v", "1", str(frame)], check=True)
    img = Image.open(frame).convert("RGB")
    layout = handler.audiogram_layout(1080, 1920, IDENTITY, "bottom", True, (400, 400))
    x0, y0, x1, y1 = layout["logo"]
    r, g, b = img.getpixel(((x0 + x1) // 2, (y0 + y1) // 2))
    assert r > 150 and 100 < g < 170 and b < 90, (r, g, b)      # brass logo centre
    tx0, ty0, tx1, ty1 = layout["title"]["box"]
    strip = img.crop((tx0, ty0, tx1, ty1)).convert("L")
    assert max(strip.getdata()) > 200                          # the title's light text
    assert img.getpixel((5, 5))[0] < 40                        # the house colour in the corner



def test_caption_size_is_fixed_per_format_long_and_short_cues_alike():
    """v0.2.1 (renders 10 / 11): one size per format; a long cue wraps to
    more lines instead of shrinking. Only a word wider than the frame
    shrinks it."""
    for (w, h), frac in (((1080, 1920), 0.034), ((1080, 1080), 0.046), ((1920, 1080), 0.046)):
        size = round(h * frac)
        short = handler.layout_caption("Grace and peace", w, h)
        long_ = handler.layout_caption("Draw near to God and he will draw near to you, humble yourselves", w, h)
        assert short[0] == long_[0] == size, (w, h, short[0], long_[0])
        assert len(long_[2]) >= 2
    wide = handler.layout_caption("Pneumonoultramicroscopicsilicovolcanoconiosis", 1080, 1920)
    assert wide[0] < round(1920 * 0.034) and wide[3][0] <= 1080 - 2 * round(1080 * 0.06)


def test_caption_fractions_match_the_app():
    assert handler.CAPTION_FONT_FRAC == {"9:16": 0.034, "1:1": 0.046, "16:9": 0.046}
    app_captions = os.path.join(os.path.dirname(ROOT), "sermount-web", "app", "clip_captions.py")
    if os.path.exists(app_captions):
        assert 'CAPTION_FONT_FRAC = {"9:16": 0.034, "1:1": 0.046, "16:9": 0.046}' in open(app_captions).read()


def test_waveform_box_is_fourteen_percent_at_9x16_and_twelve_otherwise():
    for (w, h), frac in (((1080, 1920), 0.14), ((1080, 1080), 0.12), ((1920, 1080), 0.12)):
        for position in ("bottom", "middle", "top"):
            x0, y0, x1, y1 = handler.audiogram_layout(w, h, IDENTITY, position, True, (600, 300))["wave"]
            assert y1 - y0 == int(h * frac) - int(h * frac) % 2, (w, h, position)
            side = int(round(w * handler.CAPTION_SIDE_MARGIN))
            assert x0 == side and x1 - x0 == (w - 2 * side) - (w - 2 * side) % 2      # the full strip width
    assert handler.WAVE_COLOUR == "0xB8842E@0.85"


@pytest.mark.parametrize("width,height", [(1080, 1920), (1080, 1080), (1920, 1080)])
@pytest.mark.parametrize("position", ["bottom", "middle", "top"])
def test_waveform_hugs_the_caption_band(width, height, position):
    """v0.2.2 (render 14): the wave sits against the captions with a 2 % gap,
    under the band where there is room, else over it; v0.2.1 centred it in
    the largest free span, apart from everything."""
    layout = handler.audiogram_layout(width, height, IDENTITY, position, True, None)
    band, (x0, y0, x1, y1) = layout["band"], layout["wave"]
    gap = int(round(height * 0.02))
    under, over = y0 - band[1], band[0] - y1
    assert (0 <= under <= gap) or (0 <= over <= gap + 1), (band, layout["wave"])
    if height > width * 1.2:
        assert (under >= 0) == (position in ("middle", "top"))                   # Bottom has no room under it at 9:16
    for key in ("name", "title", "meta"):
        b = layout[key]["box"]
        assert not (b[1] < y1 and y0 < b[3]), (key, b, layout["wave"])          # the stack goes round it


def test_waveform_without_captions_takes_the_largest_free_span():
    layout = handler.audiogram_layout(1080, 1920, IDENTITY, "bottom", False, None)
    assert layout["band"] is None and layout["wave"][3] - layout["wave"][1] == int(1920 * 0.14) - int(1920 * 0.14) % 2


def test_waveform_branch_is_levelled_and_the_audio_track_is_not():
    layout = handler.audiogram_layout(1080, 1920, IDENTITY, "middle", True, None)
    audio = {"kind": "audio", "audio_url": "https://supabase.test/a.mp3?token=T", "start": 1.0, "end": 8.0}
    job = dict(handler.validate_input(_job(source=audio, identity=IDENTITY)), identity_bg="/tmp/i.png", wave_box=layout["wave"])
    graph = handler.build_ffmpeg_cmd(job, "/tmp/a.m4a", 0.0, "/tmp/o.mp4", None, "libx264")
    graph = graph[graph.index("-filter_complex") + 1]
    assert f"[aw]{handler.WAVE_LEVEL},showwaves=" in graph and "mode=cline:scale=sqrt" in graph
    assert handler.WAVE_LEVEL.startswith("dynaudnorm=")
    ap = graph[graph.index("[ap]"):graph.index("[a]")]
    assert "dynaudnorm" not in ap                                                  # the heard audio is untouched


@pytest.mark.real
@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg needed")
def test_real_quiet_audiogram_draws_a_wave_that_fills_its_box(tmp_path, monkeypatch):
    """Render 14: a pulpit-level source drew a ~3 % line in a 12 % box. The
    sample's audio at -24 dB, rendered for real at 9:16 Middle, must draw at
    least 75 % of the wave box's height (peaks; quiet syllables draw less)."""
    from PIL import Image
    monkeypatch.setattr(handler, "pick_encoder", lambda: "libx264")
    quiet = str(tmp_path / "quiet.m4a")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SAMPLE, "-vn", "-af", "volume=-24dB", "-c:a", "aac", quiet], check=True)
    target = str(tmp_path / "ag.mp4")
    words = "Draw near to God and he will draw near to you".split()
    cue = {"s": 0.2, "e": 7.0, "words": [{"w": w, "s": 0.2 + i * 0.6, "e": 0.7 + i * 0.6} for i, w in enumerate(words)]}
    out = handler.run(_job(source={"kind": "audio", "audio_url": "file://" + quiet, "start": 0.5, "end": 8.5},
                           output={"put_url": "file://" + target, "token": None, "container": "mp4"},
                           captions={"cues": [cue], "position": "middle"}, identity=IDENTITY))
    assert out["ok"] is True, out
    layout = handler.audiogram_layout(1080, 1920, IDENTITY, "middle", True, None)
    x0, y0, x1, y1 = layout["wave"]
    lo = hi = None
    for t in (1.5, 3.0, 4.5, 6.0):
        png = str(tmp_path / f"f{t}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", target, "-frames:v", "1", png], check=True)
        px = Image.open(png).convert("RGB").load()
        for y in range(y0, y1):
            if any(abs(px[x, y][0] - 0xB8) < 40 and abs(px[x, y][1] - 0x84) < 40 and abs(px[x, y][2] - 0x2E) < 40
                   for x in range(x0, x1, 4)):
                lo = y if lo is None else min(lo, y)
                hi = y if hi is None else max(hi, y)
    assert lo is not None and (hi - lo + 1) >= 0.75 * (y1 - y0), (lo, hi, y0, y1)



def test_the_fetch_window_from_the_app_is_the_cached_files_origin(quiet, monkeypatch):
    """v0.2.1: the app's cache key and the worker's file origin come from the
    same window. Before, the worker used floor(start − 10) of each job's own
    start, so two renders sharing a key (a nudge, or the v1.1 lead-in on a
    range cached before it) could place the clip up to a second off."""
    events = []
    monkeypatch.setattr(handler, "fetch_youtube_range", lambda url, out, rs, re_: (events.append((rs, re_)), shutil.copyfile(SAMPLE, out)))
    monkeypatch.setattr(handler, "download", lambda *a, **k: (_ for _ in ()).throw(handler.ClipError("fetch")))
    job = handler.validate_input(_job(source={"kind": "youtube", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                              "start": 189.75, "end": 194.25, "fetch_start": 180, "fetch_end": 205}))
    path, origin, hit, size = handler.fetch_source(job, str(handler.tempfile.mkdtemp()))
    assert events == [(180.0, 205.0)] and origin == 180.0
    # without a window (an older app) the derived one, as before
    job = handler.validate_input(_job(source={"kind": "youtube", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "start": 189.75, "end": 194.25}))
    handler.fetch_source(job, str(handler.tempfile.mkdtemp()))
    assert events[-1] == (179, 205)
    for bad in ({"fetch_start": 190, "fetch_end": 205}, {"fetch_start": 180, "fetch_end": 194}, {"fetch_start": "x", "fetch_end": 205},
                {"fetch_start": 100, "fetch_end": 205}):
        with pytest.raises(handler.ClipError):
            handler.validate_input(_job(source={"kind": "youtube", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                                "start": 189.75, "end": 194.25, **bad}))


# ── v0.2.4: every output capped, 2 s keyframes, 48 kHz audio ─────────────────

V024_ARGS = {
    ("h264_nvenc", True): ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "30", "-b:v", "0",
                           "-maxrate", "0.8M", "-bufsize", "1.6M", "-profile:v", "high", "-g", "60"],
    ("libx264", True): ["-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
                        "-maxrate", "0.8M", "-bufsize", "1.6M", "-profile:v", "high", "-g", "60"],
    ("h264_nvenc", False): ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "26", "-b:v", "0",
                            "-maxrate", "4M", "-bufsize", "8M", "-profile:v", "high", "-g", "60"],
    ("libx264", False): ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                         "-maxrate", "4M", "-bufsize", "8M", "-profile:v", "high", "-g", "60"],
}


def _codec_slice(cmd):
    i = cmd.index("-c:v")
    return cmd[i:cmd.index("-pix_fmt")]


def _audio_job(**over):
    return _job(source={"kind": "audio", "audio_url": "https://supabase.test/a.mp3", "start": 100.0, "end": 108.0}, **over)


@pytest.mark.parametrize("encoder", ["h264_nvenc", "libx264"])
def test_an_audiogram_has_the_v024_arguments_exactly(encoder):
    job = handler.validate_input(_audio_job())
    cmd = handler.build_ffmpeg_cmd(job, "/t/source.audio", 0.0, "/t/out.mp4", None, encoder)
    assert _codec_slice(cmd) == V024_ARGS[(encoder, True)]
    assert handler.video_codec_args(encoder, True) == V024_ARGS[(encoder, True)]
    assert cmd[cmd.index("-ar") + 1] == "48000" and cmd[cmd.index("-b:a") + 1] == "160k"


@pytest.mark.parametrize("encoder", ["h264_nvenc", "libx264"])
def test_a_video_clip_has_the_v024_arguments_exactly(encoder):
    job = handler.validate_input(_job())
    cmd = handler.build_ffmpeg_cmd(job, "/t/source.mp4", 90.0, "/t/out.mp4", "/t/cap/captions.mov", encoder)
    assert _codec_slice(cmd) == V024_ARGS[(encoder, False)]
    assert handler.video_codec_args(encoder) == V024_ARGS[(encoder, False)]
    assert cmd[cmd.index("-ar") + 1] == "48000" and cmd[cmd.index("-b:a") + 1] == "160k"


def test_the_encode_constants_are_named():
    assert (handler.AUDIOGRAM_NVENC_CQ, handler.AUDIOGRAM_X264_CRF, handler.AUDIOGRAM_MAXRATE, handler.AUDIOGRAM_BUFSIZE) == ("30", "32", "0.8M", "1.6M")
    assert (handler.VIDEO_NVENC_CQ, handler.VIDEO_X264_CRF, handler.VIDEO_MAXRATE, handler.VIDEO_BUFSIZE) == ("26", "26", "4M", "8M")
    assert handler.GOP_FRAMES == "60" and handler.AUDIO_SAMPLE_RATE == "48000"
    assert handler.IMAGE_TAG == "v0.2.4"


def test_the_audio_only_output_is_48k_with_no_video_arguments():
    job = handler.validate_input(_job(aspect="audio", output={"container": "m4a"}, width=None, height=None))
    cmd = handler.build_ffmpeg_cmd(job, "/t/source.mp4", 90.0, "/t/out.m4a", None, "none")
    assert cmd[cmd.index("-ar") + 1] == "48000" and cmd[cmd.index("-b:a") + 1] == "128k"
    assert "-c:v" not in cmd and "-maxrate" not in cmd and "-g" not in cmd


def _probe(path, *args):
    out = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return [t.strip(",") for t in out.split()]


def _keyframes(path):
    return [float(t) for t in _probe(path, "-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries", "frame=pts_time")]


def _render_both_ways(tmp_path, monkeypatch, make_job):
    """The same job with the args before v0.2.3 (cq 23 / crf 20, no cap,
    default GOP) and with v0.2.4's."""
    monkeypatch.setattr(handler, "pick_encoder", lambda: "libx264")
    new = tmp_path / "new.mp4"
    assert handler.run(make_job(new))["ok"] is True
    v024 = handler.video_codec_args
    monkeypatch.setattr(handler, "video_codec_args", lambda encoder, audiogram=False: ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high"])
    old = tmp_path / "old.mp4"
    assert handler.run(make_job(old))["ok"] is True
    monkeypatch.setattr(handler, "video_codec_args", v024)
    return new, old


@pytest.mark.real
def test_real_audiogram_is_smaller_48k_and_keyframes_every_two_seconds(tmp_path, monkeypatch):
    audio = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SAMPLE, "-vn", "-c:a", "aac", "-ar", "96000", str(audio)], check=True)
    new, old = _render_both_ways(tmp_path, monkeypatch, lambda target: _job(
        source={"kind": "audio", "audio_url": "file://" + str(audio), "start": 0.5, "end": 9.5},
        output={"put_url": "file://" + str(target), "token": None, "container": "mp4"}, identity=IDENTITY, edits={"removed": []}))
    assert _probe(new, "-select_streams", "a:0", "-show_entries", "stream=sample_rate") == ["48000"]
    keys = _keyframes(new)
    assert keys[0] == 0.0 and {round(b - a, 2) for a, b in zip(keys, keys[1:])} == {2.0}, keys
    print(f"audiogram 9 s: before {old.stat().st_size} bytes, v0.2.4 {new.stat().st_size} bytes, keyframes {keys}")
    assert new.stat().st_size < old.stat().st_size


@pytest.mark.real
def test_real_video_clip_is_smaller_48k_and_keyframes_at_most_two_seconds_apart(tmp_path, monkeypatch):
    new, old = _render_both_ways(tmp_path, monkeypatch, lambda target: _job(
        source={"kind": "youtube", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "start": 0.5, "end": 9.5},
        cache={"get_url": "file://" + SAMPLE, "put_url": None, "token": None},
        output={"put_url": "file://" + str(target), "token": None, "container": "mp4"}, edits={"removed": []}))
    assert _probe(new, "-select_streams", "a:0", "-show_entries", "stream=sample_rate") == ["48000"]
    keys = _keyframes(new)
    # -g is the longest gap; the encoder may add a keyframe at a scene cut (the sample has some)
    assert keys[0] == 0.0 and max(b - a for a, b in zip(keys, keys[1:])) <= 2.01, keys
    print(f"video 9 s: before {old.stat().st_size} bytes, v0.2.4 {new.stat().st_size} bytes, keyframes {keys}")
    assert new.stat().st_size < old.stat().st_size


# ── v0.2.5: boot line, proxy probe, fail-fast config ─────────────────────────

FAKE_VERSIONS = {"ytdlp": "2026.08.19", "ffmpeg": "6.1.1", "runpod": "1.12.0", "js_runtime": "deno 2.9.5"}


def test_boot_line_has_every_field_in_order():
    fp = handler.proxy_fingerprint(SECRET_PROXY)
    line = handler.boot_line("h264_nvenc", FAKE_VERSIONS, SECRET_PROXY, "on", fp, "ok", 212)
    assert line == (f"clip worker {handler.IMAGE_TAG} encoder=h264_nvenc ytdlp=2026.08.19 ffmpeg=6.1.1 runpod=1.12.0 "
                    f"js_runtime=deno 2.9.5 proxy=on proxy_host=gate.example.test:10001 proxy_user=user-sp1 "
                    f"proxy_fp={fp} proxy_probe=ok probe_ms=212")
    assert "s3cr3t" not in line
    off = handler.boot_line("libx264", dict(FAKE_VERSIONS, js_runtime="none"), None, "off", None, "skipped", None)
    assert off.endswith("js_runtime=none proxy=off proxy_host=- proxy_user=- proxy_fp=- proxy_probe=skipped probe_ms=-")
    # invalid: the fingerprint of what IS set still shows, so the operator can tell which value it holds
    bad = "gate.example.test:10001"
    inv = handler.boot_line("libx264", FAKE_VERSIONS, None, "invalid", handler.proxy_fingerprint(bad), "skipped", None)
    assert f"proxy=invalid proxy_host=- proxy_user=- proxy_fp={handler.proxy_fingerprint(bad)} proxy_probe=skipped" in inv


def test_proxy_fingerprint_matches_the_readme_recipe():
    fp = handler.proxy_fingerprint("  " + SECRET_PROXY + "\n")
    shell = subprocess.run(["sh", "-c", 'printf "%s" "$V" | shasum -a 256 | cut -c1-8'], env={**os.environ, "V": SECRET_PROXY},
                           capture_output=True, text=True, check=True).stdout.strip()
    assert fp == shell and len(fp) == 8 and int(fp, 16) >= 0
    assert handler.proxy_fingerprint(None) is None and handler.proxy_fingerprint("  ") is None


def test_tool_versions_read_this_machine():
    v = handler.tool_versions()
    import yt_dlp.version
    assert v["ytdlp"] == yt_dlp.version.__version__ and v["runpod"] != "none"
    if shutil.which("ffmpeg"):
        assert v["ffmpeg"][0].isdigit(), v
    assert v["js_runtime"] == "none" or v["js_runtime"].startswith("deno "), v


def test_tool_versions_parse_first_lines(monkeypatch):
    lines = {"deno": "deno 2.9.5 (stable, release, x86_64-unknown-linux-gnu)", handler.FFMPEG: "ffmpeg version 6.1.1 Copyright (c) 2000-2023"}
    monkeypatch.setattr(handler, "_first_line", lambda cmd: lines.get(cmd[0], ""))
    v = handler.tool_versions()
    assert v["js_runtime"] == "deno 2.9.5" and v["ffmpeg"] == "6.1.1"
    monkeypatch.setattr(handler, "_first_line", lambda cmd: "")
    v = handler.tool_versions()
    assert v["js_runtime"] == "none" and v["ffmpeg"] == "none"


class _Resp:
    def __init__(self, status): self.status_code = status
    def __enter__(self): return self
    def __exit__(self, *a): return False


@pytest.mark.parametrize("status,word", [(200, "ok"), (407, "auth"), (502, "unreachable"), (403, "unreachable")])
def test_probe_maps_the_status(monkeypatch, status, word):
    seen = {}
    def fake_get(url, proxies=None, timeout=None):
        seen.update(url=url, proxies=proxies, timeout=timeout); return _Resp(status)
    monkeypatch.setattr(handler.requests, "get", fake_get)
    result, ms = handler.probe_proxy(SECRET_PROXY, "on")
    assert result == word and isinstance(ms, int)
    assert seen == {"url": handler.PROXY_PROBE_URL, "proxies": {"http": SECRET_PROXY, "https": SECRET_PROXY}, "timeout": (5, 10)}


@pytest.mark.parametrize("state", ["off", "invalid"])
def test_probe_is_skipped_without_a_valid_proxy(monkeypatch, state):
    monkeypatch.setattr(handler.requests, "get", lambda *a, **k: pytest.fail("must not probe"))
    assert handler.probe_proxy(None, state) == ("skipped", None)


def _local_proxy(reply):
    """A one-thread TCP server that answers every request (a CONNECT, for
    an https URL) with `reply`, or never answers when reply is None."""
    import socket
    import threading
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    held = []

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            try:
                conn.recv(65536)
                if reply is None:
                    held.append(conn)       # stall: keep it open, say nothing
                    continue
                conn.sendall(reply)
            except OSError:
                pass
            conn.close()
    threading.Thread(target=serve, daemon=True).start()
    return srv, srv.getsockname()[1]


def test_probe_reads_a_real_407_at_the_connect():
    """requests raises ProxyError for a 407 on CONNECT rather than returning
    the status; the probe must still say auth. Real requests, local proxy."""
    srv, port = _local_proxy(b"HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm=x\r\n"
                             b"Content-Length: 0\r\nConnection: close\r\n\r\n")
    try:
        assert handler.probe_proxy(f"http://user-sp1:s3cr3t@127.0.0.1:{port}", "on")[0] == "auth"
    finally:
        srv.close()


def test_probe_reports_a_dead_proxy_unreachable():
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    assert handler.probe_proxy(f"http://u:p@127.0.0.1:{port}", "on")[0] == "unreachable"


def test_a_failed_probe_never_stops_the_worker(monkeypatch):
    monkeypatch.setattr(handler, "probe_proxy", lambda url, state: ("unreachable", 10001))
    monkeypatch.setattr(handler, "tool_versions", lambda: FAKE_VERSIONS)
    handler.log_startup()
    assert handler.PROXY_PROBE == "unreachable" and handler.PROXY_PROBE_MS == 10001


@pytest.mark.parametrize("state,detail", [("off", "PROXY_URL is not set"), ("invalid", "PROXY_URL is invalid")])
def test_youtube_job_fails_fast_without_a_valid_proxy(quiet, monkeypatch, state, detail):
    monkeypatch.setattr(handler, "PROXY_URL", None)
    monkeypatch.setattr(handler, "PROXY_STATE", state)
    monkeypatch.setattr(handler, "download", lambda *a, **k: pytest.fail("no cache GET, no download"))
    monkeypatch.setattr(handler, "fetch_youtube_range", lambda *a: pytest.fail("no yt-dlp"))
    out = handler.run(_job(cache={"get_url": "https://supabase.test/cache/x.mp4?token=C"}))
    assert out["ok"] is False and out["reason"] == "fetch"
    assert out["cause"] == "proxy_config" and out["cause_detail"] == detail
    assert out["exec_ms"] < 1000 and not os.path.exists(quiet[0])


@pytest.mark.parametrize("state", ["off", "invalid"])
def test_audio_job_does_not_need_the_proxy(quiet, monkeypatch, state):
    monkeypatch.setattr(handler, "PROXY_URL", None)
    monkeypatch.setattr(handler, "PROXY_STATE", state)
    monkeypatch.setattr(handler, "download", lambda url, dest, timeout=None: (shutil.copyfile(SAMPLE, dest), 1)[1])
    monkeypatch.setattr(handler, "upload", lambda *a: None)
    monkeypatch.setattr(handler.subprocess, "run", _fake_ffmpeg_ok)
    out = handler.run(_audio_job())
    assert out["ok"] is True and out["cause"] is None


# ── v0.2.5: causes ───────────────────────────────────────────────────────────

P = "[proxy]"
CAUSE_TABLE = [
    ("proxy_auth", f"ERROR: [youtube] x: Unable to download API page: ('Unable to connect to proxy', OSError('Tunnel connection failed: 407 Proxy Authentication Required'))"),
    ("proxy_auth", "ERROR: 407 Proxy Authentication Required"),
    ("proxy_unreachable", f"ERROR: [youtube] x: Unable to download API page: ('Unable to connect to proxy', NewConnectionError(\"HTTPSConnection(host='{P}', port=10001): Failed to establish a new connection: [Errno 111] Connection refused\"))"),
    ("proxy_unreachable", "ERROR: Tunnel connection failed: 502 Bad Gateway"),
    ("proxy_unreachable", f"ERROR: Read timed out talking to {P}:10001 (ProxyError)"),
    ("proxy_unreachable", f"ERROR: connection reset by peer via proxy {P}"),
    ("blocked", "ERROR: [youtube] x: Sign in to confirm you're not a bot. Use --cookies-from-browser"),
    ("blocked", "ERROR: unable to download video data: HTTP Error 403: Forbidden"),
    ("blocked", "ERROR: [youtube] x: HTTP Error 429: Too Many Requests"),
    ("unavailable", "ERROR: [youtube] x: Private video. Sign in if you've been granted access to this video"),
    ("unavailable", "ERROR: [youtube] x: Video unavailable. This video has been removed by the uploader"),
    ("unavailable", "ERROR: [youtube] x: The uploader has not made this video available in your country; Video unavailable"),
    ("unavailable", "ERROR: [youtube] x: Sign in to confirm your age. This video may be inappropriate for some users."),
    ("unavailable", "ERROR: [youtube] x: This live event will begin in 3 hours."),
    ("unavailable", "ERROR: [youtube] x: Premieres in 2 days"),
    ("no_format", "ERROR: [youtube] x: Requested format is not available. Use --list-formats for a list of available formats"),
    ("no_format", "ERROR: [youtube] x: No video formats found!"),
    ("no_format", "WARNING: [youtube] x: Signature solving failed: Some formats may be missing."),
    ("no_format", "WARNING: [youtube] x: n challenge solving failed: Some formats may be missing."),
    ("fetch_timeout", "ERROR: unable to download video data: The read operation timed out"),
    ("fetch_timeout", "ERROR: Read timed out. (read timeout=20)"),
    ("fetch_other", "ERROR: ffmpeg exited with code 1"),
    ("fetch_other", ""),
]


@pytest.mark.parametrize("cause,message", CAUSE_TABLE, ids=[f"{c}-{i}" for i, (c, _) in enumerate(CAUSE_TABLE)])
def test_fetch_cause_classification(cause, message):
    assert handler.classify_fetch(message) == cause


def _raising_youtubedl(message, calls):
    class FakeYDL:
        def __init__(self, opts): self.opts = opts
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def download(self, urls):
            import yt_dlp
            calls.append(dict(self.opts))
            raise yt_dlp.utils.DownloadError(message.format(proxy=self.opts.get("proxy")))
    return FakeYDL


def test_a_407_quoting_the_proxy_is_redacted_then_classified(quiet, monkeypatch):
    """The message quotes the full proxy URL (password included): the cause
    is read from the redacted text, and the detail the app gets never holds
    the password, the user or the host."""
    import yt_dlp
    monkeypatch.setattr(handler, "FETCH_RETRY_SLEEP", 0)
    msg = ("ERROR: [youtube] x: Unable to download API page: ('Unable to connect to proxy', "
           "OSError('Tunnel connection failed: 407 Proxy Authentication Required')) proxy={proxy} " + "z" * 400)
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _raising_youtubedl(msg, []))
    out = handler.run(_job())
    assert out["ok"] is False and out["reason"] == "fetch" and out["cause"] == "proxy_auth"
    d = out["cause_detail"]
    assert "proxy=[proxy]" in d and len(d) <= 300 and "\n" not in d
    for part in ("s3cr3t", "user-sp1", "gate.example.test"):
        assert part not in d and part not in json.dumps(out)


def test_real_youtubedl_messages_classify_against_a_local_proxy(monkeypatch):
    """Real yt-dlp, real requests, a local proxy (no internet): a 407 at the
    CONNECT reads proxy_auth and a closed port proxy_unreachable, so the
    wording the classifier keys on is yt-dlp 2026.08.19's own."""
    import socket
    import yt_dlp
    monkeypatch.setattr(handler, "FETCH_RETRY_SLEEP", 0)
    srv, port = _local_proxy(b"HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm=x\r\n"
                             b"Content-Length: 0\r\nConnection: close\r\n\r\n")
    s = socket.socket(); s.bind(("127.0.0.1", 0)); dead = s.getsockname()[1]; s.close()
    try:
        for p, want in ((port, "proxy_auth"), (dead, "proxy_unreachable")):
            url = f"http://user-sp1:s3cr3t@127.0.0.1:{p}"
            monkeypatch.setattr(handler, "PROXY_URL", url)
            with pytest.raises(handler.ClipError) as err:
                handler.fetch_youtube_range("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "/nonexistent/x.mp4", 90, 118)
            assert err.value.reason == "fetch" and err.value.cause == want, err.value.detail
            assert "s3cr3t" not in err.value.detail and "user-sp1" not in err.value.detail
    finally:
        srv.close()


def test_a_success_carries_null_cause_and_the_boot_values(quiet, monkeypatch):
    monkeypatch.setattr(handler, "download", lambda url, dest, timeout=None: (shutil.copyfile(SAMPLE, dest), 1)[1])
    monkeypatch.setattr(handler, "upload", lambda *a: None)
    monkeypatch.setattr(handler.subprocess, "run", _fake_ffmpeg_ok)
    out = handler.run(_job(cache={"get_url": "https://supabase.test/c.mp4"}))
    assert out["ok"] is True
    assert (out["cause"], out["cause_detail"], out["proxy_probe"], out["proxy_fp"]) == (None, None, "ok", handler.proxy_fingerprint(SECRET_PROXY))


class _HttpResp:
    def __init__(self, status, chunks=()):
        self.status_code, self.chunks = status, chunks
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_content(self, n): return iter(self.chunks)


@pytest.mark.parametrize("behaviour,cause,detail", [
    (lambda: _HttpResp(403), "source_http", "HTTP 403"),
    (lambda: _HttpResp(200, []), "source_empty", None),
    (lambda: (_ for _ in ()).throw(__import__("requests").ConnectionError("https://supabase.test/a.mp3?token=SECRETTOKEN")), "source_network", "ConnectionError"),
])
def test_audio_fetch_causes(quiet, monkeypatch, behaviour, cause, detail):
    monkeypatch.setattr(handler.requests, "get", lambda *a, **k: behaviour())
    out = handler.run(_audio_job())
    assert (out["reason"], out["cause"], out["cause_detail"]) == ("fetch", cause, detail)
    assert "SECRETTOKEN" not in json.dumps(out)


def test_audio_fetch_too_big(quiet, monkeypatch):
    monkeypatch.setattr(handler, "MAX_SOURCE_BYTES", 10)
    monkeypatch.setattr(handler.requests, "get", lambda *a, **k: _HttpResp(200, [b"x" * 8, b"x" * 8]))
    out = handler.run(_audio_job())
    assert (out["reason"], out["cause"]) == ("fetch", "source_too_big")


def _cached(monkeypatch):
    monkeypatch.setattr(handler, "download", lambda url, dest, timeout=None: (shutil.copyfile(SAMPLE, dest), 1)[1])


def test_render_causes(quiet, monkeypatch):
    _cached(monkeypatch)
    monkeypatch.setattr(handler, "upload", lambda *a: None)
    job = _job(cache={"get_url": "https://supabase.test/c.mp4"}, captions=None)
    monkeypatch.setattr(handler.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "first\nConversion failed!\n"))
    out = handler.run(job)
    assert (out["reason"], out["cause"], out["cause_detail"]) == ("render", "encode", "Conversion failed!")
    monkeypatch.setattr(handler.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    out = handler.run(job)
    assert (out["reason"], out["cause"]) == ("render", "encode_empty")
    def slow(cmd, **kw): raise subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(handler.subprocess, "run", slow)
    out = handler.run(job)
    assert (out["reason"], out["cause"]) == ("timeout", "encode_timeout")


def test_caption_track_cause(quiet, monkeypatch):
    _cached(monkeypatch)
    monkeypatch.setattr(handler.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "bad concat\n"))
    out = handler.run(_job(cache={"get_url": "https://supabase.test/c.mp4"}))
    assert (out["reason"], out["cause"], out["cause_detail"]) == ("render", "caption_track", "bad concat")


@pytest.mark.parametrize("put,cause,detail", [
    (lambda *a, **k: type("R", (), {"status_code": 413})(), "upload_http", "HTTP 413"),
    (lambda *a, **k: (_ for _ in ()).throw(__import__("requests").Timeout("https://supabase.test/x?token=SECRETTOKEN")), "upload_network", "Timeout"),
])
def test_upload_causes(quiet, monkeypatch, put, cause, detail):
    _cached(monkeypatch)
    monkeypatch.setattr(handler.subprocess, "run", _fake_ffmpeg_ok)
    monkeypatch.setattr(handler.requests, "put", put)
    out = handler.run(_job(cache={"get_url": "https://supabase.test/c.mp4"}))
    assert (out["reason"], out["cause"], out["cause_detail"]) == ("upload", cause, detail)
    assert "SECRETTOKEN" not in json.dumps(out)


def test_input_too_long_and_error_causes(quiet, monkeypatch):
    assert handler.run({"source": {"kind": "youtube"}})["cause"] == "input"
    _cached(monkeypatch)
    out = handler.run(_job(cache={"get_url": "https://supabase.test/c.mp4"}, edits={"removed": [[0.0, 7.8]]}))
    assert (out["reason"], out["cause"]) == ("input", "too_short")
    out = handler.run(_job(cache={"get_url": "https://supabase.test/c.mp4"}, max_output_seconds=5))
    assert (out["reason"], out["cause"]) == ("too_long", None)
    monkeypatch.setattr(handler, "download", lambda *a, **k: 1 / 0)
    out = handler.run(_job(cache={"get_url": "https://supabase.test/c.mp4"}))
    assert (out["reason"], out["cause"], out["cause_detail"]) == ("error", "ZeroDivisionError", None)


def test_reason_keeps_its_eight_words():
    """The app maps exactly these (app/clips.py _WORKER_REASONS); a cause
    never leaks into reason."""
    src = open(os.path.join(ROOT, "handler.py")).read()
    reasons = set(__import__("re").findall(r'ClipError\(\s*"(\w+)"', src)) | {"error"}
    assert reasons <= {"input", "fetch", "cache", "render", "upload", "too_long", "timeout", "error"}, reasons


# ── v0.2.5: the fetch's clocks ───────────────────────────────────────────────

def test_socket_timeout_and_rw_timeout_are_explicit():
    opts = handler.ytdlp_options("/tmp/x.mp4", 90, 118)
    assert handler.SOCKET_TIMEOUT == 20.0 and opts["socket_timeout"] == 20.0
    assert opts["external_downloader_args"]["ffmpeg_i"][:2] == ["-rw_timeout", "20000000"]
    assert handler.ytdlp_options("/tmp/x.mp4", 90, 118, socket_timeout=7.5)["external_downloader_args"]["ffmpeg_i"][:2] == ["-rw_timeout", "7500000"]


@pytest.mark.parametrize("value,want", [(None, 600.0), ("", 600.0), ("90", 90.0), ("x", 600.0), ("-5", 600.0)])
def test_fetch_wall_secs_is_env_overridable(monkeypatch, value, want):
    if value is None:
        monkeypatch.delenv("FETCH_WALL_SECS", raising=False)
    else:
        monkeypatch.setenv("FETCH_WALL_SECS", value)
    assert handler._env_secs("FETCH_WALL_SECS", 600.0) == want


class _Clock:
    def __init__(self): self.t = 1000.0
    def now(self): return self.t
    def sleep(self, s): self.t += s


def _stalling_youtubedl(clock, stall, calls):
    """Each attempt stalls for its whole socket timeout (capped at `stall`), then fails as yt-dlp would."""
    class FakeYDL:
        def __init__(self, opts): self.opts = opts
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def download(self, urls):
            import yt_dlp
            calls.append((clock.t, self.opts["socket_timeout"]))
            clock.t += min(stall, self.opts["socket_timeout"])
            raise yt_dlp.utils.DownloadError("ERROR: unable to download video data: The read operation timed out")
    return FakeYDL


def test_a_stalled_fetch_ends_on_the_wall_clock_with_fetch_timeout(monkeypatch, tmp_path):
    import yt_dlp
    clock = _Clock()
    monkeypatch.setattr(handler, "_now", clock.now)
    monkeypatch.setattr(handler.time, "sleep", clock.sleep)
    monkeypatch.setattr(handler, "FETCH_WALL_SECS", 30.0)
    calls = []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _stalling_youtubedl(clock, 1e9, calls))
    start = clock.t
    with pytest.raises(handler.ClipError) as err:
        handler.fetch_youtube_range("https://www.youtube.com/watch?v=dQw4w9WgXcQ", str(tmp_path / "s.mp4"), 90, 118)
    assert err.value.reason == "fetch" and err.value.cause == "fetch_timeout"
    assert "timed out" in err.value.detail                              # the last attempt's message is kept
    # attempt 1 gets 20 s; after it and the 2 s sleep only 8 s are left, and attempt 2's budget is capped there
    assert calls == [(start, 20.0), (start + 22.0, 8.0)]
    assert clock.t - start <= handler.FETCH_WALL_SECS + 2 * handler.SOCKET_TIMEOUT


def test_a_two_attempt_stall_ends_within_the_bound_at_the_default_clock(monkeypatch, tmp_path):
    """At the default 600 s wall clock a stall is bounded by the socket
    timeout per attempt, long before the wall clock: FETCH_WALL_SECS +
    2 × socket_timeout is the ceiling the brief asks to verify."""
    import yt_dlp
    clock = _Clock()
    monkeypatch.setattr(handler, "_now", clock.now)
    monkeypatch.setattr(handler.time, "sleep", clock.sleep)
    calls = []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _stalling_youtubedl(clock, 1e9, calls))
    start = clock.t
    with pytest.raises(handler.ClipError) as err:
        handler.fetch_youtube_range("https://www.youtube.com/watch?v=dQw4w9WgXcQ", str(tmp_path / "s.mp4"), 90, 118)
    assert err.value.cause == "fetch_timeout"                         # classified from "timed out"
    assert clock.t - start <= handler.FETCH_WALL_SECS + 2 * handler.SOCKET_TIMEOUT


@pytest.mark.real
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg needed")
@pytest.mark.parametrize("via_proxy", [False, True])
def test_real_ffmpeg_rw_timeout_ends_a_stalled_input(via_proxy):
    """Real ffmpeg against a local server that accepts and never answers
    (directly, and as the -http_proxy): -rw_timeout 2 s ends it in ~2 s.
    Without it ffmpeg would wait forever, and so would FFmpegFD."""
    import time as _t
    srv, port = _local_proxy(None)
    try:
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-rw_timeout", "2000000"]
        if via_proxy:
            args += ["-http_proxy", f"http://u:p@127.0.0.1:{port}", "-i", "http://media.example.test/v.mp4"]
        else:
            args += ["-i", f"http://127.0.0.1:{port}/v.mp4"]
        t0 = _t.monotonic()
        proc = subprocess.run(args + ["-f", "null", "-"], capture_output=True, text=True, timeout=30)
        took = _t.monotonic() - t0
    finally:
        srv.close()
    assert proc.returncode != 0 and 1.5 < took < 10, (took, proc.stderr)
