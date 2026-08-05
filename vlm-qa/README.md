# vlm-qa — ask questions about images and pre-recorded video on the IQ-9075

A small Python app that answers questions about an **image file** or a
**pre-recorded video file** using a vision-language model running on the
**Hexagon NPU** of a Qualcomm Dragonwing IQ-9075 (QCS9075).

Not for live streams — media is read from disk, in full, before inference.
Obtaining the media is out of scope; point the app at a file that already exists.

## The stack

| Layer | Choice | Why |
|---|---|---|
| Model | `Qwen3-VL-4B-Instruct`, W4A16 | Qualcomm AI Hub publishes this bundle **precompiled for `qualcomm-qcs9075`** — no export, no build host |
| Runtime | GenieX `qairt` backend | Qualcomm AI Engine Direct; pinned to the NPU |
| Server | `geniex serve` | OpenAI-compatible HTTP API on `127.0.0.1:18181` |
| Media | `ffmpeg` / `ffprobe` | Frame sampling and tiling with no ARM64 wheel problems |
| App | this repo | Media → frames → multimodal request → answer |
| Front end | WebSocket (`vlmqa serve`) | Upload media, ask, stream tokens back; `web/index.html` is a client |

Python dependencies are `requests` and — for the WebSocket server only —
`websockets`. Both are pure Python; nothing has to be built on the board.

## Install

```bash
ssh ubuntu@iq9 'bash -s' < setup-iq9.sh
```

Idempotent: installs the Qualcomm driver packages and QAIRT runtime, the GenieX
CLI, pulls the 4.1 GiB model bundle, creates the venv, and starts the server.

## Use

```bash
cd ~/vlm-qa

# An image
.venv/bin/python -m vlmqa ask -m photo.jpg -q "How many people are there?"

# A video
.venv/bin/python -m vlmqa ask -m clip.mp4 -q "What does the person do?"

# More frames, streamed token by token
.venv/bin/python -m vlmqa ask -m clip.mp4 -q "Describe the sequence." -n 10 --stream

# Machine-readable
.venv/bin/python -m vlmqa ask -m clip.mp4 -q "What moves?" --json

# Several questions about one file
.venv/bin/python -m vlmqa chat -m clip.mp4

# Serve it over a WebSocket instead
.venv/bin/python -m vlmqa serve

# Is the server up, is the model loaded?
.venv/bin/python -m vlmqa status
```

### Library

```python
from pathlib import Path
from vlmqa.qa import ask_about, Session

answer, prepared = ask_about(Path("clip.mp4"), "What happens?", frames=8)
print(answer.text, answer.latency_s)

with Session(Path("clip.mp4"), frames=6) as s:
    print(s.ask("What is the setting?").text)
    print(s.ask("What colour is the vehicle?").text)   # follow-up keeps context
```

## WebSocket server

```bash
.venv/bin/python -m vlmqa serve                  # ws://127.0.0.1:8765
.venv/bin/python -m vlmqa serve --host 0.0.0.0   # set VLMQA_WS_TOKEN first
```

Then open `web/index.html` from disk — no web server, no build step. Pick a
photo or a video, type a question, watch the answer arrive token by token.

One connection is one conversation about one piece of media: upload once, ask
as many questions as you like, and the follow-ups keep their context exactly as
`chat` does. Streaming is the point of using a socket here — on this board the
first token lands seconds ahead of the last.

### Protocol

Control messages are JSON text frames. Media bodies are **raw binary frames**,
so a video costs its own size on the wire rather than the +33% base64 adds.

| Client → server | |
|---|---|
| `{"type":"auth","token":"…"}` | Only if the server was started with a token |
| `{"type":"upload","name":"clip.mp4","size":N,"frames":8,"strategy":"frames"}` | Header; exactly `N` bytes of binary frames follow |
| `{"type":"upload","name":"cat.jpg","data":"<base64>"}` | Small files, in one message |
| `{"type":"ask","prompt":"What happens?","stream":true}` | `frames`/`strategy` may be repeated here, which re-samples the media |
| `{"type":"reset"}` `{"type":"status"}` `{"type":"ping"}` | |

| Server → client | |
|---|---|
| `{"type":"ready","protocol":1,"model":"…","defaults":{…}}` | Sent on connect; `defaults` carries `max_frames` and the upload cap |
| `{"type":"progress","received":N,"size":M}` | Throttled to ~5% steps |
| `{"type":"media","kind":"video","images":6,"frames":["00:00.83",…],"note":null}` | Media is sampled and ready |
| `{"type":"queued"}` | Another connection has the NPU |
| `{"type":"token","text":"…"}` | Streaming only |
| `{"type":"answer","text":"…","latency_s":4.8,"prompt_tokens":…}` | Always sent, streamed or not |
| `{"type":"error","code":"auth\|protocol\|media\|vlm\|internal","message":"…"}` | The connection stays open |

A malformed message costs you an `error`, not the socket and the video you
already uploaded.

```python
import asyncio, json, websockets
from pathlib import Path

async def ask(path: Path, prompt: str) -> str:
    async with websockets.connect("ws://127.0.0.1:8765", max_size=None) as ws:
        await ws.recv()                                    # ready
        body = path.read_bytes()
        await ws.send(json.dumps({"type": "upload", "name": path.name, "size": len(body)}))
        await ws.send(body)
        await ws.send(json.dumps({"type": "ask", "prompt": prompt}))
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "token":  print(msg["text"], end="", flush=True)
            if msg["type"] == "answer": return msg["text"]
            if msg["type"] == "error":  raise RuntimeError(msg["message"])

print(asyncio.run(ask(Path("clip.mp4"), "What happens?")))
```

### What it does with the board

- **One inference at a time.** The NPU is serial whatever the server does, so
  requests take a shared lock; a client that has to wait gets `queued` first and
  a truthful `latency_s` after.
- **Nothing blocks the event loop.** ffmpeg and the call to GenieX run in worker
  threads, so uploads and pings still get answered while the model is decoding.
- **Uploads stream to disk** in the connection's temp dir, which is deleted when
  the socket closes. A phone video never sits in RAM in one piece.
- **Loopback by default.** `--host 0.0.0.0` on a shared network wants
  `VLMQA_WS_TOKEN` set; the first message then has to be `auth`.

One piece of media per connection — re-uploading replaces it, and 4096 tokens of
context has no room for two videos anyway.

## How video is handled

A VLM only ever sees stills, so a video is sampled into frames taken from the
**midpoint** of N equal slices (avoids black lead-in frames and unreadable
trailing packets). Two ways to present them:

| Strategy | Cost | Behaviour |
|---|---|---|
| `frames` (default) | ~275 tokens/frame | Each frame sent full-size. Best at motion and fine detail. |
| `sheet` | ~4× cheaper | Frames tiled into one contact sheet. Cheap and keeps ordering, but each frame shrinks. |

The difference is real, measured on the same 10 s clip and question:

- `sheet`, 6 frames — *"no discernible movement… the bus remains in the same position"* — **wrong**
- `frames`, 6 frames — *"a white bus… moving from left to right across the frames"* — **correct**

So `frames` is the default. Use `sheet` when frame *coverage* matters more than
per-frame detail (a long video where you want 30 sample points).

### The context ceiling

The `qairt` bundle's context length is **baked in at compile time** — 4096
tokens, and `--nctx` cannot raise it. Measured prompt cost at `max_image_edge=896`:

| Frames | Prompt tokens |
|---|---|
| 6 | 1695 |
| 8 | 2231 |
| 12 | 3304 |

That works out to ~275 tokens/frame, giving a ceiling of **12 frames** with 512
tokens reserved for the answer. Ask for more and the app tiles them into a
contact sheet instead of overflowing, and says so in a `note`.

## Measured performance

Qwen3-VL-4B-Instruct W4A16 on the NPU, this board:

| Workload | Latency |
|---|---|
| Single image | 2.9 s |
| Video, 6 frames (`frames`) | 4.8 s |
| Video, 12 frames (`frames`) | 9.7 s |
| Follow-up in a `chat` session | 2.6–3.5 s |

Qualcomm's published figures for this bundle on the IQ-9075 EVK: 1397 tok/s
prefill, 18.4 tok/s decode, TTFT 92 ms–2.9 s.

## Configuration

Every setting is an environment variable; CLI flags win over them.

| Variable | Default |
|---|---|
| `VLMQA_BASE_URL` | `http://127.0.0.1:18181/v1` |
| `VLMQA_MODEL` | `qualcomm/Qwen3-VL-4B-Instruct:W4A16` |
| `VLMQA_MAX_TOKENS` | `512` |
| `VLMQA_TEMPERATURE` | `0.2` |
| `VLMQA_MAX_IMAGE_EDGE` | `896` |
| `VLMQA_DEFAULT_FRAMES` | `6` |
| `VLMQA_CONTEXT_TOKENS` | `4096` |
| `VLMQA_READ_TIMEOUT` | `600` |
| `VLMQA_WS_HOST` | `127.0.0.1` |
| `VLMQA_WS_PORT` | `8765` |
| `VLMQA_WS_TOKEN` | *(unset — no auth)* |
| `VLMQA_WS_MAX_UPLOAD_MB` | `512` |
| `VLMQA_WS_MAX_MESSAGE_MB` | `16` |

The app is a plain HTTP client, so it can also drive a GenieX server on another
box — set `VLMQA_BASE_URL` and run it from anywhere.

## Verifying it is really on the NPU

The `qairt` runtime is NPU-only by construction, but to confirm:

```bash
PID=$(pgrep -f 'geniex serve' | head -1)
sudo ls -l /proc/$PID/fd | grep cdsp          # /dev/fastrpc-cdsp -> Hexagon DSP
sudo grep -o 'libQnnHtp[^ ]*' /proc/$PID/maps | sort -u
```

On a healthy setup this shows open handles on `/dev/fastrpc-cdsp` and
`libQnnHtp.so` / `libQnnHtpV73Stub.so` mapped in.

## Notes and limitations

- **No image parts in history.** GenieX returns
  `SDKError(Multimodal generation failed)` if a request replays images inside
  conversation history. `Session` therefore re-sends the images on every turn
  and keeps prior Q&A as plain text. Follow-ups cost about as much as the first
  question; they are not free.
- **Frame sampling is uniform, not content-aware.** An event shorter than the
  gap between samples can be missed entirely. Raise `--frames` for short clips
  with fast action.
- **`ffmpeg` warns** `libOpenCL.so.1: no version information available` on this
  image. It is harmless — `qcom-adreno1` replaces the generic ICD loader.
- **Context is fixed at 4096.** For a longer window, a bundle compiled for a
  larger context would have to be pulled from AI Hub.
- The QUAD `run.ps1` convention does not apply here: this app's target is Linux
  ARM64 on the board, so `setup-iq9.sh` is the equivalent entry point.

## Why not the QUAD MCP bundle registry

`aihub_select` was the intended path, but on the current server it cannot
deliver this model:

- Catalog actions (`search` / `pick` / `info` / `get`) fail — `qai-hub-models`
  is not installed server-side.
- `pull` reports no published bundle; the registry holds only `cifar10`,
  `mobilenetish`, and `dl-test`.
- `ensure` profiles the **MCP server's own** hardware (an AMD EPYC 7B12) rather
  than the board, so its build advice does not apply.
- `request_build` accepts a job but returns empty `build_commands`.

The bundle already exists precompiled for `qualcomm-qcs9075`, and GenieX pulls
it directly — which is what `setup-iq9.sh` does.
