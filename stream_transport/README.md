# DragonAssist WebSocket transport

One laptop service accepts continuous VLM text from IQ9 and questions from the
phone. IQ9 text is appended verbatim to one durable, unbounded file per video.
The latest accepted question for each video is stored as plain UTF-8 text at
`received/<video_id>/query.txt`.

## Run on `qcworkshop3`

From the repository root in PowerShell:

```powershell
$env:DRAGONASSIST_STREAM_TOKEN = "replace-with-an-iq9-secret"
$env:DRAGONASSIST_PHONE_TOKEN = "replace-with-a-different-phone-secret"
.\stream_transport\run.ps1
```

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
| `DRAGONASSIST_STREAM_TOKEN` | required off loopback | IQ9 secret |
| `DRAGONASSIST_PHONE_TOKEN` | required off loopback | Phone secret |
| `DRAGONASSIST_STREAM_ROOT` | `received` | Context and query text storage |
| `DRAGONASSIST_STREAM_MAX_MESSAGE_BYTES` | `1048576` | Per-frame limit |

Restrict Windows Firewall TCP 8001 to Tailscale. Never commit tokens or put
them in a WebSocket URL.

## IQ9 sender

```powershell
python -m stream_transport.mock_iq9 `
  --url ws://qcworkshop3:8001/v1/iq9 `
  --video-id demo-video `
  --input sample.txt `
  --token "iq9-secret" `
  --chunk-chars 256
```

The sender resumes from `next_sequence` after a disconnect. Text is appended
exactly as supplied; the receiver adds no separators.

## Phone query and laptop worker

Submit a question and wait for the server to push its result:

```powershell
python -m stream_transport.mock_phone `
  --url ws://qcworkshop3:8001/v1/phone `
  --token "phone-secret" `
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
receives a pushed `query_result`. Stable request IDs make retries idempotent
while the server is running. Other laptop components can read the question
directly from the matching `query.txt` file.

## Tests

```powershell
python -m pip install -r stream_transport\requirements.txt
python -m unittest discover -s stream_transport\tests -v
```
