# sermount-clip-worker

RunPod Serverless worker that renders one approved sermon clip: it fetches **only the clip's
range** from the church's YouTube video (or takes the stored audio), cuts the hidden words, crops
to the chosen format, burns captions, normalises the audio and uploads the file. Nothing else.
No model, no torch, no database or storage keys: every URL in the job is signed and short-lived.

RunPod builds this repo from GitHub. Nothing is built or pushed from a laptop.

## Contract

Input (`POST /v2/{endpoint_id}/run`, body `{"input": ...}`):

```json
{"source": {"kind": "youtube", "url": "https://www.youtube.com/watch?v=...", "start": 3721.0, "end": 3812.5},
 "cache": {"get_url": "<signed GET or null>", "put_url": "<signed PUT or null>", "token": "..."},
 "output": {"put_url": "<signed PUT>", "token": "...", "container": "mp4"},
 "aspect": "9:16", "width": 1080, "height": 1920,
 "captions": {"cues": [{"s": 0.0, "e": 2.1, "words": [{"w": "I", "s": 0.0, "e": 0.2}]}], "position": "bottom"},
 "edits": {"removed": [[12.4, 13.1]]},
 "normalize_audio": true,
 "max_output_seconds": 1200,
 "background": "#0f1115"}
```

- `source.kind` is `youtube` (`url`) or `audio` (`audio_url`, a signed URL to the stored
  file). `start` / `end` are on **that source's own clock**: YouTube full-video time for
  youtube, stored-audio time for audio. The app converts; the worker never adds an offset.
- `edits.removed` and every caption time are **relative to `source.start`**, on the source
  clock. The removals are cut from the timeline (audio and video together); the cues are
  mapped onto the output clock with the same list, so a caption never points at a cut word.
- `aspect` is `16:9`, `1:1`, `9:16` or `audio`. `audio` writes an `.m4a` and skips video
  entirely (`container` must be `m4a`); the other three need `width` and `height`
  (1920×1080, 1080×1080, 1080×1920) and `container: mp4`. An audio source with a video aspect
  renders an **audiogram**: a solid `background` (Sermount dark `#0f1115` by default) with the
  captions on it.
- `captions` is `null` for no burn. `captions.position` is `bottom` (default), `middle` or `top`: the
  caption's bottom edge sits at 88 %, 56 % or 24 % of the frame height. Lines wrap to the frame's own
  width less a 6 % margin each side, and the font steps down until the cue fits in two lines.
- `cache` URLs may be `null` (no cache).

Output (always; the handler never raises):

```json
{"ok": true, "reason": null, "bytes": 18214422, "duration": 90.2,
 "fetch_ms": 41200, "render_ms": 9800, "exec_ms": 51300,
 "cache_hit": false, "encoder": "h264_nvenc", "image_tag": "v0.1.0"}
```

`reason` on failure, one word: `input`, `fetch` (the range could not be downloaded), `cache`,
`render`, `upload`, `too_long` (output after removals over `max_output_seconds`), `timeout`,
`error`.

## How it renders (the rules it follows)

1. **Fetch only the range.** Cache GET first. On a miss, `yt-dlp` with `download_ranges` over
   `[start − 10, end + 10]` (rounded outward to whole seconds), `force_keyframes_at_cuts`, format
   `bestvideo[height<=1080]+bestaudio/best`, the encoder arguments in **`external_downloader_args`**
   (`{'ffmpeg_o': ['-c:v', 'libx264', '-crf', '19', '-preset', 'veryfast', '-c:a', 'aac']}`) **and**
   `merge_output_format: 'mp4'`. Both together: with `download_ranges` yt-dlp hands the download
   to FFmpegFD, which reads `external_downloader_args` (not `postprocessor_args`), and without the
   container pin it defaults to WebM, which rejects H.264 and forces a VP9 software transcode of
   every fetch (SermonVault measured 485–574 s per 120 s range before the fix, 40–74 s after).
   The test suite captures the argv FFmpegFD actually spawns and asserts both. Then the range is
   PUT to the cache. One fetch at a time per worker process.
2. **Render.** `trim` to the exact range, `select` / `aselect` cut the removed ranges, centre-crop
   to the aspect, scale, captions composited with `overlay`, `loudnorm` when asked. Never
   `-c:v copy` with a burn. `h264_nvenc` when the ffmpeg lists it, else `libx264` (chosen at
   runtime by probing `ffmpeg -encoders`, so the worker runs on a laptop without a GPU; a listed
   nvenc that fails to open falls back to libx264 for that job).
3. **Captions** are drawn by Pillow into PNG strips, one per spoken-word state (the current word
   in the accent colour), and placed in the bottom third. The layout rules (bold sans, dark pill,
   two lines split at the space nearest the middle, ~84 characters per cue) are the same ones the
   app's Edit Mode page draws in its preview.
4. **Upload** with the signed PUT (`x-upsert: true`, the token in the query). Every temp file is
   deleted in a `finally`, on success and on every failure.

## Privacy

Logs carry counts, timings, the encoder and the one-word reason. Never URLs, caption text or
words. `urllib3` is pinned at WARNING (a signed URL's path carries its token) and the RunPod SDK
runs at `RUNPOD_LOG_LEVEL=INFO`. RunPod keeps a job's input and output for a while after it
finishes (`/status`): the input holds the cue text and expiring signed URLs, the output holds
numbers only.

## Endpoint settings (RunPod console)

New Endpoint → Serverless → **GitHub repo** `sermountapp/sermount-clip-worker`, branch `main`,
Dockerfile path `Dockerfile`.

| Setting | Value |
|---|---|
| GPU | cheapest 16 GB or 24 GB tier offered |
| Min workers (active) | 0 |
| Max workers | 2 |
| Idle timeout | default |
| Execution timeout | 5400 s (fetch time is dominated by YouTube's checks, not length) |
| FlashBoot | on |
| Network volume | none |
| Environment variables | `PROXY_URL`: required for YouTube sermons (RunPod datacenter IPs get YouTube's "Sign in to confirm you're not a bot" wall without it). The same value the Render staging worker uses. Set it as an endpoint secret, never in the Dockerfile or a job input. Audio sermons do not use it. |

The worker also hands `PROXY_URL` to ffmpeg (`-http_proxy`) for the range fetch, so the proxy port must map to one fixed IP for the duration of a fetch: the media URL is bound to the IP that resolved it.

After the endpoint exists, create an API key **scoped to this endpoint** (keys are per endpoint;
the alignment key returns 403 here). It goes into the Render **worker** service only, as
`RUNPOD_CLIP_API_KEY` with `RUNPOD_CLIP_ENDPOINT_ID` (`sync: false`). Never into the web service,
a file, or a chat.

## Shipping a new version

**Every push to `main` builds and deploys** (verified 2026-10-08 on the whisperx repo: a
README-only push started a build and workers rolled onto it). So `main` is production for the
endpoint; batch README changes with code changes, and push only what passed the local checks.

1. Bump `IMAGE_TAG` in `handler.py` (the app records it per render) and commit.
2. Push to `main`. RunPod console → endpoint → Builds: wait for Completed; workers roll over.
3. Run the console test (below) and check `image_tag` in the output.
4. GitHub → Releases → Draft a new release with the same tag (the version label; it does not
   trigger or gate a build).

**Live tag:** none yet (the endpoint has not been created).

## Rolling back

RunPod console → endpoint → Builds → the previous build's `⋯` menu → **Rollback**; if old workers
keep serving, scale workers to 0 so they restart on the rolled-back build. On the Sermount side
the faster rollback is `CLIPS_ENABLED=false` on the Render web and worker services: no request
reaches this endpoint at all.

## Console test

`test_input.json` renders seconds 1 to 9 of the sample as a 9:16 captioned clip with one cut from `samples/sample-10s.mp4`
(a synthetic ffmpeg `testsrc2` pattern with a 440 Hz tone, no church content) as a cache hit.
Paste it into the endpoint's **Requests** tab after replacing `output.put_url` and
`output.token` with a signed upload URL from the Supabase `clips` bucket (`python -c` in the app
venv: `from app.storage import create_signed_upload_url; print(create_signed_upload_url('clips',
'test/console.mp4'))`). Expect `ok: true`, `duration: 7.0`, `cache_hit: true`, `encoder:
h264_nvenc`.

## Local checks (no GPU needed)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q tests            # shape tests; yt-dlp's spawn, the network and ffmpeg are faked
.venv/bin/python -m pytest -q tests -m real    # one real libx264 render of the sample (needs ffmpeg + ffprobe)
```

The real run writes `out/sample-9x16.mp4`: 8 s of source, 1 s removed, 7 s of 1080×1920 output
with captions, `libx264`.
