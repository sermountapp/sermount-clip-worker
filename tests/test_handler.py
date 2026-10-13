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
    assert opts["external_downloader_args"] == {"ffmpeg_o": ["-c:v", "libx264", "-crf", "19", "-preset", "veryfast", "-c:a", "aac"]}
    assert opts["merge_output_format"] == "mp4"
    assert opts["force_keyframes_at_cuts"] is True
    assert opts["format"] == "bestvideo[height<=1080]+bestaudio/best"
    assert "postprocessor_args" not in opts
    assert handler.padded_range(100.3, 108.2) == (90, 119)
    assert handler.padded_range(4.0, 20.0) == (0, 30)


SECRET_PROXY = "http://user-sp1:s3cr3t%2Bpw@gate.example.test:10001"


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
        assert opts["external_downloader_args"]["ffmpeg_i"] == ["-http_proxy", value]
    else:
        assert "proxy" not in opts
        assert "ffmpeg_i" not in opts["external_downloader_args"]
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
        handler.log_startup()
        monkeypatch.setattr(yt_dlp, "YoutubeDL", _fake_youtubedl(2, []))
        result = handler.run(_job())
    finally:
        handler.log.removeHandler(grab)
    assert lines[0] == f"clip worker {handler.IMAGE_TAG} encoder=libx264 proxy=on"
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
    # bottom: the strip's bottom edge 12 % above the frame's (1920 - 248 strip - 230), as before v0.2.0
    assert "overlay=x=(W-w)/2:y=1442:format=yuv420:eof_action=pass" in graph and "shortest" not in graph and "loudnorm" in graph
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
    y = int(round(height * anchor)) - handler.caption_strip_height(height)
    assert f"[cap]overlay=x=(W-w)/2:y={y}:" in graph
    assert 0 <= y and y + handler.caption_strip_height(height) <= height


def test_caption_position_is_validated_and_defaults_to_bottom():
    assert handler.validate_input(_job())["caption_position"] == "bottom"
    with pytest.raises(handler.ClipError):
        handler.validate_input(_job(captions={"cues": _job()["captions"]["cues"], "position": "left"}))


@pytest.mark.parametrize("width,height", [(1080, 1920), (1080, 1080), (1920, 1080)])
@pytest.mark.parametrize("text", [LONG_CUE, "short line", "Supercalifragilisticexpialidocious words everywhere " * 2])
def test_no_caption_line_exceeds_the_frame_less_the_margin(width, height, text):
    """The 9:16 overflow (render 7, 2026-10-12): every laid-out line fits the
    frame width minus both side margins, in at most two lines."""
    font_px, font, lines, widths = handler.layout_caption(" ".join(text.split()), width, height)
    margin = int(round(width * handler.CAPTION_SIDE_MARGIN))
    assert 1 <= len(lines) <= 2
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
