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

Only Python dependency is `requests`.

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
