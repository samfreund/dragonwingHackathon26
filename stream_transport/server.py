from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from websockets.asyncio.server import serve as websocket_serve
except ImportError:  # pragma: no cover
    from websockets import serve as websocket_serve

from .phone_broker import PhoneQueryBroker
from .phone_receiver import PhoneReceiver
from .receiver import StreamReceiver
from .storage import TextStore


@dataclass(frozen=True)
class ServerConfig:
    host: str = os.getenv("DRAGONASSIST_STREAM_HOST", "0.0.0.0")
    port: int = int(os.getenv("DRAGONASSIST_STREAM_PORT", "8001"))
    storage_root: Path = Path(os.getenv("DRAGONASSIST_STREAM_ROOT", "received"))
    max_message_bytes: int = int(os.getenv("DRAGONASSIST_STREAM_MAX_MESSAGE_BYTES", str(1024 * 1024)))


class CombinedServer:
    def __init__(self, iq9: StreamReceiver, phone: PhoneReceiver) -> None:
        self.iq9 = iq9
        self.phone = phone

    async def handler(self, websocket) -> None:
        request = getattr(websocket, "request", None)
        path = (getattr(request, "path", "/") or "/").split("?", 1)[0]
        if path in {"/", "/v1/iq9"}:
            await self.iq9.handler(websocket)
            return
        if path == "/v1/phone":
            await self.phone.handler(websocket)
            return
        await websocket.send(json.dumps({
            "type": "error", "code": "unknown_path", "message": f"Unknown path {path}",
        }))
        await websocket.close(code=1008, reason="unknown_path")


def build(config: ServerConfig) -> tuple[CombinedServer, PhoneQueryBroker]:
    store = TextStore(config.storage_root)
    broker = PhoneQueryBroker(config.storage_root / "phone_queries.sqlite3")
    return CombinedServer(StreamReceiver(store), PhoneReceiver(broker)), broker


async def run(config: ServerConfig) -> None:
    combined, _ = build(config)
    async with websocket_serve(
        combined.handler,
        config.host,
        config.port,
        max_size=config.max_message_bytes,
        ping_interval=20,
        ping_timeout=60,
    ):
        print(
            f"DragonAssist WebSockets on ws://{config.host}:{config.port} "
            "(/v1/iq9, /v1/phone)",
            flush=True,
        )
        await asyncio.Future()


def main() -> None:
    defaults = ServerConfig()
    parser = argparse.ArgumentParser(description="DragonAssist IQ9 and phone WebSocket server")
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--storage-root", type=Path, default=defaults.storage_root)
    args = parser.parse_args()
    config = ServerConfig(
        host=args.host,
        port=args.port,
        storage_root=args.storage_root,
        max_message_bytes=defaults.max_message_bytes,
    )
    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
