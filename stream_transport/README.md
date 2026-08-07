# IQ9-to-laptop text stream

This standalone transport receives plain VLM text over WebSocket and appends it
verbatim to one durable, unbounded file per video. It does not run inference,
parse the text, or depend on the laptop's RAG implementation.

## Laptop receiver (`qcworkshop3`)

From the repository root in PowerShell:

```powershell
$env:DRAGONASSIST_STREAM_TOKEN = "replace-with-a-shared-secret"
.\stream_transport\run.ps1
```

The default endpoint is `ws://qcworkshop3:8001`. Output is written to:

```text
received/<video_id>/context.txt
```

Configuration:

| Variable | Default | Purpose |
|---|---|---|
| `DRAGONASSIST_STREAM_HOST` | `0.0.0.0` | Bind interface |
| `DRAGONASSIST_STREAM_PORT` | `8001` | WebSocket port |
| `DRAGONASSIST_STREAM_TOKEN` | required off loopback | Shared secret |
| `DRAGONASSIST_STREAM_ROOT` | `received` | Output directory |
| `DRAGONASSIST_STREAM_MAX_MESSAGE_BYTES` | `1048576` | Per-frame limit |

Restrict Windows Firewall TCP 8001 to the Tailscale network. Never commit or
put the token in a WebSocket URL.

## Mock IQ9 sender

Do not run or deploy anything on IQ9 while device benchmarking is active. Test
from another machine with:

```powershell
python -m stream_transport.mock_iq9 `
  --url ws://qcworkshop3:8001 `
  --video-id demo-video `
  --input sample.txt `
  --token "replace-with-a-shared-secret" `
  --chunk-chars 256
```

The sender resumes from the receiver's `next_sequence` after a disconnect.
Text is appended exactly as supplied; the receiver adds no separators.

## Protocol v1

One socket carries one video. Client messages are `start`, `text`, and optional
`end`; server messages are `started`, `ack`, and `error`. Each text message has
a monotonically increasing sequence. An acknowledgement is sent only after the
UTF-8 bytes and sequence state are durable. Retries are idempotent.

Sriram's laptop code can read or tail `received/<video_id>/context.txt` while it
grows. The SQLite file beside the video directories is private transport state,
not an application integration API.

## Tests

```powershell
python -m pip install -r stream_transport\requirements.txt
python -m unittest discover -s stream_transport\tests -v
```
