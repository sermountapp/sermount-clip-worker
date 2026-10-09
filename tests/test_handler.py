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


def test_ffmpegfd_spawns_our_encoder_args_and_an_mp4(tmp_path):
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
    assert "overlay=x=(W-w)/2:y=H-h-230:format=yuv420:eof_action=pass" in graph and "shortest" not in graph and "loudnorm" in graph
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
