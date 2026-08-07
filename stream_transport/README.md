# DragonAssist WebSocket transport

One laptop service accepts continuous VLM text from IQ9 and questions from the
phone. IQ9 text is appended verbatim to one durable, unbounded file per video.
The latest accepted question for each video is stored as plain UTF-8 text at
`received/<video_id>/query.txt`.

## The whole path

```
IQ-9075                     qcworkshop3                        phone
-------                     -----------                        -----
vlmqa publish  --ws-->  :8001/v1/iq9  --> received/<id>/context.txt
                                                  |
                                                  v
                        :8001/v1/phone <--ws--  AssistClient
                                |                    ^
                                v                    |
                    received/<id>/.phone_requests/*.json
                                |                    |
                                v                    |
                        laptop_worker -> ask.py -----+
```

The board captions video window by window and streams the text here
(`vlmqa publish`, see `vlm-qa/README.md`). The phone asks questions about a
`video_id` and gets answers from the accumulated context, routed by `ask.py`
to either the NPU reader or the cloud model.

## Run on `qcworkshop3`

**Two processes, two different virtualenvs.** Both must be running.

```powershell
.\stream_transport\run.ps1          # the server,  on stream_transport\.venv
.\stream_transport\run-worker.ps1   # the worker,  on npu_qa\.venv
```

The split is not cosmetic. The server needs only `websockets`. The worker loads
`ask.py`, which pulls in the query router and the NPU reader, and those live
only in `npu_qa\.venv` — on the server's venv it constructs `HybridQA` fine and
then dies at the first real question with `ModuleNotFoundError: numpy`.

Without the worker, queries are accepted, stored, acknowledged as `pending`,
and never answered.

Both endpoints use port 8001:

```text
ws://qcworkshop3:8001/v1/iq9
ws://qcworkshop3:8001/v1/phone
```

`/` remains an alias for the IQ9 endpoint. Video output is stored at:

```text
received/<video_id>/context.txt
```

Configuration:

| Variable | Default | Purpose |
|---|---|---|
| `DRAGONASSIST_STREAM_HOST` | `0.0.0.0` | Bind interface |
| `DRAGONASSIST_STREAM_PORT` | `8001` | Shared WebSocket port |
| `DRAGONASSIST_STREAM_ROOT` | `received` | Context and broker storage |
| `DRAGONASSIST_STREAM_MAX_MESSAGE_BYTES` | `1048576` | Per-frame limit |

There is no application-level authentication. The tailnet is the security
boundary: the Windows Firewall rule for TCP 8001 is scoped to `100.64.0.0/10`,
so only Tailscale peers can reach the port. Keep that rule in place — without
it, anyone who can route to 8001 can read and write video context.

## IQ9 sender

In production the sender is `vlmqa publish` on the board, which captions a
video window by window and streams each caption here as it is generated:

```bash
.venv/bin/python -m vlmqa publish --media clip.mp4 \
  --url ws://qcworkshop3:8001/v1/iq9 --video-id loading-dock
```

To exercise this end without the board, replay a text file instead:

```powershell
python -m stream_transport.mock_iq9 `
  --url ws://qcworkshop3:8001/v1/iq9 `
  --video-id demo-video `
  --input sample.txt `
  --chunk-chars 256
```

Both resume from `next_sequence` after a disconnect. Text is appended exactly
as supplied; the receiver adds no separators, so the sender owns its own line
endings.

## Phone query and laptop worker

The phone client is `AssistClient` in the Android app
(`mobile/.../assist/AssistClient.kt`); point it at this host with an
`assist.properties` pushed to the device. Prefer its `askDurable`, which
reconnects and collects a stored answer by `request_id` rather than re-asking.

To exercise the same conversation from the laptop:

```powershell
python -m stream_transport.mock_phone `
  --url ws://qcworkshop3:8001/v1/phone `
  --request-id request-1 `
  --video-id demo-video `
  --question "What happened?"
```

Exercise the worker handoff without loading a model:

```powershell
python -m stream_transport.laptop_worker `
  --storage-root received --once `
  --mock-answer "The person entered the room."
```

Without `--mock-answer`, the worker loads the repository's `ask.HybridQA` and
answers against the matching `context.txt`. Sriram can instead import
`PhoneQueryBroker` and use `claim_next()`, `complete()`, and `fail()` from his
own service.

## Protocol v1

An IQ9 socket carries one video. Messages are `start`, `text`, and optional
`end`; replies are `started`, `ack`, and `error`. Sequence acknowledgements are
sent only after the UTF-8 text and state are durable.

A phone starts with `phone_start`, submits `query`, receives `query_ack`, then
receives a pushed `query_result`. After reconnecting, `query_status` retrieves
a stored result. Stable request IDs make retries idempotent, and other laptop
components can read the question directly from the matching `query.txt` file.

The two endpoints are still not interchangeable: each rejects the other's
opening message by shape, so a client pointed at the wrong path fails loudly
rather than silently misbehaving.

## Tests

```powershell
python -m pip install -r stream_transport\requirements.txt
python -m unittest discover -s stream_transport\tests -v
```
