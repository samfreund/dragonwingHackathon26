"""WebSocket front end for the VLM question-answering app.

One connection is one conversation about one piece of media: the client
uploads a photo or a video, then asks as many questions about it as it
likes. Answers stream back token by token, which is most of the reason to
use a socket rather than the CLI -- on this board the first token lands
seconds before the last one.

Wire protocol. Control messages are JSON text frames; media bodies are raw
binary frames, so a video costs its own size on the wire rather than the
+33% base64 would add.

  client -> server
    {"type":"auth",   "token":"..."}                       if the server has one
    {"type":"upload", "name":"clip.mp4", "size":1048576,   binary frames follow
                      "frames":8, "strategy":"frames",
                      "video_id":"loading-dock"}          optional; names the
                                                          transcript upstream
    {"type":"upload", "name":"cat.jpg", "data":"<base64>"} small files, one shot
    {"type":"ask",    "prompt":"What happens?", "stream":true}
    {"type":"reset"}                                       forget media + history
    {"type":"status"}
    {"type":"ping"}

  server -> client
    {"type":"ready",    "protocol":1, "model":"...", "defaults":{...}}
    {"type":"progress", "received":N, "size":M}
    {"type":"media",    "kind":"video", "images":6, "frames":[...], ...}
    {"type":"publishing",     "video_id":"..."}   if VLMQA_PUBLISH_URL is set
    {"type":"published",      "video_id":"...", "captions":3}
    {"type":"publish_failed", "video_id":"...", "message":"..."}
    {"type":"queued"}                                      NPU busy elsewhere
    {"type":"token",    "text":"..."}                      streaming only
    {"type":"answer",   "text":"...", "latency_s":4.8, ...}
    {"type":"auth",     "ok":true}                         failure is an error
    {"type":"error",    "code":"auth|protocol|media|vlm|internal", "message":"..."}
    {"type":"status",   "up":true, "models":[...]}
    {"type":"pong"}

The NPU serves one request at a time, so inference is serialised behind a
lock shared by every connection; a client that has to wait is told so with
a `queued` message. All blocking work -- ffmpeg frame extraction and the
HTTP call to GenieX -- runs in a worker thread, so the event loop stays
free to answer pings and accept uploads while the model is busy.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

try:  # websockets >= 13 ships the asyncio implementation used here
    from websockets.asyncio.server import serve as _ws_serve
except ImportError:  # pragma: no cover -- websockets 12 and older
    try:
        from websockets import serve as _ws_serve
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The WebSocket server needs the `websockets` package. Install it with:\n"
            "  pip install -r requirements.txt"
        ) from exc

from websockets.exceptions import ConnectionClosed

from .client import Answer, VLMClient, VLMError
from .config import Settings, settings as default_settings
from .media import MediaError
from .publisher import PublishError, publish
from .qa import Session

PROTOCOL_VERSION = 1

# Publish jobs outlive the connection that triggered them: the phone is free to
# hang up the moment it has its answer, and captioning the rest of the clip is
# no longer its business. Holding a strong reference keeps the loop from
# garbage-collecting a task nobody is awaiting.
_PUBLISH_JOBS: set[asyncio.Task] = set()

# The client names the file it is sending; only its extension is kept, and
# only if it is boring. `classify()` falls back to ffprobe when the suffix
# means nothing to it, so an unusable extension costs nothing.
_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,8}$")

STRATEGIES = ("frames", "sheet")


class ProtocolError(ValueError):
    """The client sent something that doesn't make sense at this point."""


def _suffix_of(name: str) -> str:
    suffix = Path(str(name)).suffix.lower()
    return suffix if _SAFE_SUFFIX.match(suffix) else ".bin"


def _safe_video_id(value) -> str:
    """Coerce a client-supplied id into what the receiver will accept.

    The id names a directory on the laptop, so the receiver enforces
    `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` and closes the socket on anything else.
    Rewriting here means a phone sending "My Clip (2).mp4" gets a usable id
    instead of a disconnect it cannot diagnose. Empty means "pick one for me".
    """
    if value is None:
        return ""
    cleaned = "".join(
        c if (c.isalnum() and c.isascii()) or c in "._-" else "-" for c in str(value)
    ).lstrip("._-")[:128]
    return cleaned


def _int_or_none(value, field: str, *, minimum: int = 1) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"`{field}` must be an integer, got {value!r}") from exc
    if number < minimum:
        raise ProtocolError(f"`{field}` must be >= {minimum}, got {number}")
    return number


def _strategy_or_none(value, field: str = "strategy") -> str | None:
    if value is None:
        return None
    if value not in STRATEGIES:
        raise ProtocolError(f"`{field}` must be one of {STRATEGIES}, got {value!r}")
    return value


class _Upload:
    """A media file arriving over one or more binary frames.

    Chunks are written straight through to disk. A phone-shot video is
    hundreds of megabytes and this board does not have RAM to spare for a
    copy of it that ffmpeg is only going to read off disk anyway.
    """

    def __init__(
        self,
        path: Path,
        size: int,
        frames: int | None,
        strategy: str | None,
        video_id=None,
    ) -> None:
        self.path = path
        self.size = size
        self.frames = frames
        self.strategy = strategy
        self.video_id = video_id
        self.received = 0
        self._reported = 0
        self._fh = path.open("wb")

    def write(self, chunk: bytes) -> None:
        self._fh.write(chunk)
        self.received += len(chunk)

    @property
    def complete(self) -> bool:
        return self.received >= self.size

    def should_report(self) -> bool:
        """Throttle progress to ~5% steps; a chunked upload can be thousands
        of frames and the client only needs a progress bar out of it."""
        if self.complete or self.received - self._reported >= self.size // 20:
            self._reported = self.received
            return True
        return False

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class _Connection:
    """Per-socket state: the uploaded file, the Session over it, its history."""

    _counter = 0

    def __init__(self, ws, settings: Settings, gate: asyncio.Lock) -> None:
        _Connection._counter += 1
        self.id = _Connection._counter
        self.ws = ws
        self.settings = settings
        self.gate = gate
        self._scratch = tempfile.TemporaryDirectory(
            prefix="vlmqa-ws-", ignore_cleanup_errors=True
        )
        self.dir = Path(self._scratch.name)
        self.client = VLMClient(settings)
        self.session: Session | None = None
        self.media: Path | None = None
        self.upload: _Upload | None = None
        self.authed = not settings.ws_token

    # ------------------------------------------------------------ plumbing

    def log(self, message: str) -> None:
        print(f"[ws {self.id}] {message}", file=sys.stderr, flush=True)

    async def _send(self, payload: dict) -> None:
        await self.ws.send(json.dumps(payload))

    async def _error(self, code: str, message: str) -> None:
        self.log(f"error ({code}): {message}")
        await self._send({"type": "error", "code": code, "message": message})

    # --------------------------------------------------------- message loop

    async def run(self) -> None:
        await self._send({
            "type": "ready",
            "protocol": PROTOCOL_VERSION,
            "model": self.settings.model,
            "auth_required": not self.authed,
            "defaults": {
                "frames": self.settings.default_frames,
                "strategy": "frames",
                "max_frames": self.settings.max_frames(),
                "max_upload_bytes": self.settings.ws_max_upload_bytes,
                "max_message_bytes": self.settings.ws_max_message_bytes,
            },
        })

        async for message in self.ws:
            # One bad message should cost the client an `error`, not the
            # connection and the frames it has already uploaded.
            try:
                if isinstance(message, (bytes, bytearray)):
                    await self._on_binary(bytes(message))
                else:
                    await self._on_text(message)
            except ProtocolError as exc:
                await self._error("protocol", str(exc))
            except MediaError as exc:
                await self._error("media", str(exc))
            except VLMError as exc:
                await self._error("vlm", str(exc))
            except ConnectionClosed:
                raise
            except Exception as exc:  # noqa: BLE001 - never drop the socket
                await self._error("internal", f"{type(exc).__name__}: {exc}")

    async def _on_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except ValueError as exc:
            raise ProtocolError(f"Not valid JSON: {exc}") from exc
        if not isinstance(msg, dict):
            raise ProtocolError("Expected a JSON object")

        kind = msg.get("type")
        if kind == "auth":
            return await self._on_auth(msg)
        self._require_auth()

        handlers = {
            "upload": self._on_upload,
            "ask": self._on_ask,
            "reset": self._on_reset,
            "status": self._on_status,
            "ping": self._on_ping,
        }
        handler = handlers.get(kind)
        if handler is None:
            raise ProtocolError(
                f"Unknown message type {kind!r}; expected one of "
                f"{', '.join(sorted(handlers))}"
            )
        await handler(msg)

    def _require_auth(self) -> None:
        if not self.authed:
            raise ProtocolError('Send {"type":"auth","token":"..."} first')

    # ------------------------------------------------------------ handlers

    async def _on_auth(self, msg: dict) -> None:
        # One message per outcome: `auth` on success, `error` on failure. A
        # client that renders errors and nothing else still sees the refusal.
        if not self.authed and str(msg.get("token") or "") != self.settings.ws_token:
            return await self._error("auth", "Bad token")
        self.authed = True
        await self._send({"type": "auth", "ok": True})

    async def _on_ping(self, msg: dict) -> None:
        await self._send({"type": "pong"})

    async def _on_upload(self, msg: dict) -> None:
        # Validate everything before touching state: a rejected upload must
        # leave the media already loaded on this connection alone.
        frames = _int_or_none(msg.get("frames"), "frames")
        strategy = _strategy_or_none(msg.get("strategy"))
        name = str(msg.get("name") or "media")
        inline = msg.get("data")
        blob = self._decode_inline(inline) if inline is not None else None
        size = len(blob) if blob is not None else _int_or_none(msg.get("size"), "size")
        if size is None:
            raise ProtocolError(
                "`upload` needs either `data` (base64) or a positive `size`, "
                "followed by that many bytes of binary frames"
            )
        self._check_size(size)

        # This upload supersedes whatever was here: a half-finished body and
        # the previous file, which would otherwise sit in the temp dir until
        # the socket closed.
        self._abort_upload()
        self._close_session()
        self._discard_media()
        path = self.dir / f"media{_suffix_of(name)}"

        video_id = msg.get("video_id")

        if blob is not None:
            path.write_bytes(blob)
            self.log(f"upload {name} ({size} bytes, inline)")
            await self._prepare(path, frames, strategy)
            await self._maybe_publish(video_id)
            return

        self.upload = _Upload(path, size, frames, strategy, video_id)
        self.log(f"upload {name} ({size} bytes, streaming)")
        await self._send({"type": "progress", "received": 0, "size": size})

    async def _on_binary(self, chunk: bytes) -> None:
        self._require_auth()
        upload = self.upload
        if upload is None:
            raise ProtocolError(
                'Unexpected binary frame; send an {"type":"upload"} header first'
            )
        if upload.received + len(chunk) > upload.size:
            self._abort_upload()
            raise ProtocolError(
                f"Upload body is longer than the declared {upload.size} bytes"
            )

        upload.write(chunk)
        if upload.should_report():
            await self._send({
                "type": "progress",
                "received": upload.received,
                "size": upload.size,
            })
        if upload.complete:
            upload.close()
            self.upload = None
            await self._prepare(upload.path, upload.frames, upload.strategy)
            await self._maybe_publish(upload.video_id)

    async def _on_ask(self, msg: dict) -> None:
        prompt = str(msg.get("prompt") or msg.get("question") or "").strip()
        if not prompt:
            raise ProtocolError("`ask` needs a non-empty `prompt`")
        if self.upload is not None:
            raise ProtocolError(
                f"Upload still in progress ({self.upload.received} of "
                f"{self.upload.size} bytes); send the rest first"
            )
        if self.session is None or self.media is None:
            raise ProtocolError(
                'No media loaded; send an {"type":"upload"} message first'
            )

        # Sampling can also be re-specified here. It is a property of the
        # media rather than the question, so changing it re-runs ffmpeg and
        # starts a fresh conversation.
        frames = _int_or_none(msg.get("frames"), "frames") or self.session.n_frames
        strategy = _strategy_or_none(msg.get("strategy")) or self.session.strategy
        if frames != self.session.n_frames or strategy != self.session.strategy:
            await self._prepare(
                self.media, frames, strategy,
                note="Re-sampled the media; earlier questions were dropped.",
            )

        stream = bool(msg.get("stream", True))
        self.log(f"ask ({'streaming' if stream else 'blocking'}): {prompt[:60]!r}")

        # The board runs one inference at a time whatever we do here; taking
        # the lock explicitly keeps latencies honest and lets us say so.
        if self.gate.locked():
            await self._send({"type": "queued"})
        async with self.gate:
            answer = await self._infer(prompt, stream)

        await self._send({
            "type": "answer",
            "text": answer.text,
            "streamed": stream,
            "latency_s": round(answer.latency_s, 2),
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
        })
        self.log(f"answered in {answer.latency_s:.1f}s")

    async def _on_reset(self, msg: dict) -> None:
        self._abort_upload()
        self._close_session()
        self._discard_media()
        await self._send({"type": "reset", "ok": True})

    async def _on_status(self, msg: dict) -> None:
        up = await asyncio.to_thread(self.client.is_up)
        models = await asyncio.to_thread(self.client.loaded_models) if up else []
        await self._send({
            "type": "status",
            "server": self.settings.base_url,
            "up": up,
            "models": models,
            "model": self.settings.model,
            "media": self.session.prepared.description if self.session else None,
        })

    # ------------------------------------------------------------ internals

    def _decode_inline(self, data) -> bytes:
        if not isinstance(data, str):
            raise ProtocolError("`data` must be a base64 string")
        # Tolerate a browser's `data:image/jpeg;base64,...` prefix.
        _, _, payload = data.rpartition(",") if data.startswith("data:") else ("", "", data)
        try:
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProtocolError(f"`data` is not valid base64: {exc}") from exc

    def _check_size(self, size: int) -> None:
        cap = self.settings.ws_max_upload_bytes
        if size > cap:
            raise ProtocolError(
                f"Upload is {size} bytes; the limit is {cap} "
                f"({self.settings.ws_max_upload_mb} MB). Raise VLMQA_WS_MAX_UPLOAD_MB "
                "or send a shorter clip."
            )

    async def _prepare(
        self,
        path: Path,
        frames: int | None,
        strategy: str | None,
        *,
        note: str = "",
    ) -> None:
        """Extract frames and open a Session over them, off the event loop."""
        self._close_session()
        session = await asyncio.to_thread(
            Session,
            path,
            frames=frames,
            strategy=strategy or "frames",
            settings=self.settings,
        )
        self.session = session
        self.media = path

        prepared = session.prepared
        self.log(f"prepared {prepared.description}")
        await self._send({
            "type": "media",
            "kind": prepared.kind,
            "strategy": prepared.strategy,
            "images": prepared.n_images,
            "frames": [f.label for f in prepared.frames],
            "description": prepared.description,
            "note": "\n".join(filter(None, (note, prepared.note))) or None,
        })

    async def _maybe_publish(self, requested_id) -> None:
        """Stream captions of the just-uploaded video to the configured receiver.

        Only text leaves this board. The video is captioned here and the
        captions are sent as JSON text frames; the file itself stays in the
        scratch directory and dies with the connection.

        The job deliberately outlives this connection. Captioning a clip takes
        far longer than answering one question, and a phone that uploads, asks,
        and disconnects should still leave a complete transcript behind.
        """
        cfg = self.settings
        if not cfg.publish_url or self.media is None or self.session is None:
            return
        if self.session.prepared.kind != "video":
            return  # a photo has no windows to walk

        video_id = _safe_video_id(requested_id) or f"upload-{int(time.time())}"

        # Take our own reference to the file before handing off. close() wipes
        # the connection's scratch directory, and a hard link costs nothing on
        # the same filesystem, so the job cannot lose the media underneath it.
        job_dir = Path(tempfile.mkdtemp(prefix="vlmqa-pub-"))
        target = job_dir / self.media.name
        try:
            os.link(self.media, target)
        except OSError:
            await asyncio.to_thread(shutil.copy2, self.media, target)

        task = asyncio.create_task(
            self._publish_job(target, job_dir, video_id),
            name=f"publish-{video_id}",
        )
        _PUBLISH_JOBS.add(task)
        task.add_done_callback(_PUBLISH_JOBS.discard)

        self.log(f"publishing as {video_id} -> {cfg.publish_url}")
        await self._send_quietly({"type": "publishing", "video_id": video_id})

    async def _publish_job(self, media: Path, job_dir: Path, video_id: str) -> None:
        cfg = self.settings
        try:
            sent = await publish(
                media,
                cfg.publish_url,
                video_id,
                window_s=cfg.publish_window_s,
                settings=cfg,
                gate=self.gate,
            )
            self.log(f"published {sent} caption(s) as {video_id}")
            await self._send_quietly({
                "type": "published", "video_id": video_id, "captions": sent,
            })
        except (PublishError, MediaError, VLMError, OSError) as exc:
            # A failed publish must not take the QA session with it; the phone
            # can still ask this board questions about the media it holds.
            self.log(f"publish failed for {video_id}: {type(exc).__name__}: {exc}")
            await self._send_quietly({
                "type": "publish_failed", "video_id": video_id, "message": str(exc),
            })
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    async def _send_quietly(self, payload: dict) -> None:
        """Send if the socket is still there; a hung-up phone is not an error."""
        try:
            await self._send(payload)
        except (ConnectionClosed, RuntimeError):
            pass

    async def _infer(self, prompt: str, stream: bool) -> Answer:
        session = self.session
        if not stream:
            return await asyncio.to_thread(session.ask, prompt)

        # `on_token` is called from the worker thread, so tokens are handed
        # to the loop through a queue and sent by a task that owns the socket.
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue = asyncio.Queue()

        def on_token(piece: str) -> None:
            loop.call_soon_threadsafe(outbox.put_nowait, piece)

        async def pump() -> None:
            while True:
                piece = await outbox.get()
                if piece is None:
                    return
                await self._send({"type": "token", "text": piece})

        pump_task = asyncio.create_task(pump())
        try:
            return await asyncio.to_thread(
                session.ask, prompt, stream=True, on_token=on_token
            )
        finally:
            # Callbacks queued by the worker are already ahead of this one in
            # the loop's ready queue, so the sentinel cannot overtake a token.
            outbox.put_nowait(None)
            await pump_task

    def _abort_upload(self) -> None:
        if self.upload is not None:
            self.upload.close()
            self.upload = None

    def _close_session(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def _discard_media(self) -> None:
        if self.media is not None:
            self.media.unlink(missing_ok=True)
            self.media = None

    def close(self) -> None:
        self._abort_upload()
        self._close_session()
        self._scratch.cleanup()


async def serve(
    settings: Settings | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Run the WebSocket server until cancelled."""
    cfg = settings or default_settings
    host = host or cfg.ws_host
    port = port if port is not None else cfg.ws_port
    gate = asyncio.Lock()

    async def handler(ws) -> None:
        conn = _Connection(ws, cfg, gate)
        conn.log("connected")
        try:
            await conn.run()
        except ConnectionClosed:
            pass
        finally:
            conn.close()
            conn.log("disconnected")

    async with _ws_serve(
        handler,
        host,
        port,
        max_size=cfg.ws_max_message_bytes,
        ping_interval=20,
        ping_timeout=60,
    ):
        print(
            f"vlmqa websocket on ws://{host}:{port}  "
            f"(model {cfg.model}, upstream {cfg.base_url})"
            + ("  [token required]" if cfg.ws_token else ""),
            file=sys.stderr,
            flush=True,
        )
        await asyncio.get_running_loop().create_future()  # run forever
