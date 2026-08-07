from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from websockets.asyncio.server import serve as websocket_serve
except ImportError:  # pragma: no cover - compatibility with websockets 12
    from websockets import serve as websocket_serve
from websockets.exceptions import ConnectionClosed

from .protocol import ProtocolError, parse_message, parse_start, sequence, video_id
from .storage import TextStore


@dataclass(frozen=True)
class ReceiverConfig:
    host: str = os.getenv("DRAGONASSIST_STREAM_HOST", "0.0.0.0")
    port: int = int(os.getenv("DRAGONASSIST_STREAM_PORT", "8001"))
    storage_root: Path = Path(os.getenv("DRAGONASSIST_STREAM_ROOT", "received"))
    max_message_bytes: int = int(os.getenv("DRAGONASSIST_STREAM_MAX_MESSAGE_BYTES", str(1024 * 1024)))


class StreamReceiver:
    def __init__(self, store: TextStore) -> None:
        self.store = store
        self._active: set[str] = set()
        self._active_lock = asyncio.Lock()

    async def handler(self, websocket) -> None:
        bound_video: str | None = None
        try:
            try:
                first = await asyncio.wait_for(websocket.recv(), timeout=10)
                start = parse_start(parse_message(first))
                async with self._active_lock:
                    if start.video_id in self._active:
                        raise ProtocolError("video_busy", "Another connection is streaming this video")
                    self._active.add(start.video_id)
                    bound_video = start.video_id
                state = await asyncio.to_thread(self.store.state, bound_video)
                await self._send(websocket, {
                    "type": "started",
                    "video_id": bound_video,
                    "next_sequence": state.last_sequence + 1,
                    "bytes_written": state.bytes_written,
                    "complete": state.complete,
                })
            except ProtocolError as exc:
                await self._send(websocket, exc.payload())
                await websocket.close(code=1008, reason=exc.code)
                return

            async for raw in websocket:
                try:
                    message = parse_message(raw)
                    kind = message.get("type")
                    if kind == "ping":
                        await self._send(websocket, {"type": "pong"})
                    elif kind == "text":
                        await self._on_text(websocket, bound_video, message)
                    elif kind == "end":
                        await self._on_end(websocket, bound_video, message)
                    else:
                        raise ProtocolError("unknown_type", f"Unknown message type {kind!r}")
                except ProtocolError as exc:
                    await self._send(websocket, exc.payload())
        except (ConnectionClosed, asyncio.TimeoutError):
            pass
        finally:
            if bound_video is not None:
                async with self._active_lock:
                    self._active.discard(bound_video)

    async def _on_text(self, websocket, bound_video: str, message: dict) -> None:
        self._require_video(bound_video, message)
        seq = sequence(message.get("sequence"))
        text = message.get("text")
        if not isinstance(text, str):
            raise ProtocolError("invalid_text", "text must be a string")
        result = await asyncio.to_thread(self.store.append, bound_video, seq, text)
        await self._send(websocket, {
            "type": "ack",
            "video_id": bound_video,
            "sequence": result.sequence,
            "bytes_written": result.bytes_written,
            "duplicate": result.duplicate,
        })

    async def _on_end(self, websocket, bound_video: str, message: dict) -> None:
        self._require_video(bound_video, message)
        seq = sequence(message.get("sequence"), allow_zero=True)
        state = await asyncio.to_thread(self.store.finish, bound_video, seq)
        await self._send(websocket, {
            "type": "ack",
            "video_id": bound_video,
            "sequence": state.last_sequence,
            "bytes_written": state.bytes_written,
            "complete": True,
        })

    @staticmethod
    def _require_video(bound_video: str, message: dict) -> None:
        received = video_id(message.get("video_id"))
        if received != bound_video:
            raise ProtocolError(
                "video_mismatch",
                f"Socket is bound to {bound_video!r}, received {received!r}",
            )

    @staticmethod
    async def _send(websocket, payload: dict) -> None:
        await websocket.send(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


async def run(config: ReceiverConfig) -> None:
    receiver = StreamReceiver(TextStore(config.storage_root))
    async with websocket_serve(
        receiver.handler,
        config.host,
        config.port,
        max_size=config.max_message_bytes,
        ping_interval=20,
        ping_timeout=60,
    ):
        print(
            f"DragonAssist stream receiver on ws://{config.host}:{config.port} "
            f"-> {config.storage_root.resolve()}",
            flush=True,
        )
        await asyncio.Future()


def main() -> None:
    defaults = ReceiverConfig()
    parser = argparse.ArgumentParser(description="Receive IQ9 text and append it per video")
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--storage-root", type=Path, default=defaults.storage_root)
    args = parser.parse_args()
    try:
        asyncio.run(run(ReceiverConfig(
            host=args.host,
            port=args.port,
            storage_root=args.storage_root,
            max_message_bytes=defaults.max_message_bytes,
        )))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
