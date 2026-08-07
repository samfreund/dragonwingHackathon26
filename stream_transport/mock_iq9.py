from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from . import PROTOCOL_VERSION


async def stream_file(
    url: str,
    video_id: str,
    source: Path,
    *,
    chunk_chars: int = 1024,
    delay: float = 0,
    retries: int = 5,
) -> None:
    attempts = 0
    while True:
        try:
            async with connect(url, max_size=1024 * 1024) as websocket:
                await websocket.send(json.dumps({
                    "type": "start",
                    "protocol": PROTOCOL_VERSION,
                    "video_id": video_id,
                }))
                started = json.loads(await websocket.recv())
                if started.get("type") == "error":
                    raise RuntimeError(f"Receiver refused stream: {started}")
                next_sequence = int(started["next_sequence"])
                last_sequence = 0
                with source.open("r", encoding="utf-8", newline="") as handle:
                    sequence = 0
                    while chunk := handle.read(chunk_chars):
                        sequence += 1
                        last_sequence = sequence
                        if sequence < next_sequence:
                            continue
                        await websocket.send(json.dumps({
                            "type": "text",
                            "video_id": video_id,
                            "sequence": sequence,
                            "text": chunk,
                        }, ensure_ascii=False))
                        reply = json.loads(await websocket.recv())
                        if reply.get("type") != "ack" or reply.get("sequence") != sequence:
                            raise RuntimeError(f"Unexpected receiver response: {reply}")
                        if delay:
                            await asyncio.sleep(delay)
                await websocket.send(json.dumps({
                    "type": "end",
                    "video_id": video_id,
                    "sequence": last_sequence,
                }))
                reply = json.loads(await websocket.recv())
                if reply.get("type") != "ack" or not reply.get("complete"):
                    raise RuntimeError(f"Receiver did not complete stream: {reply}")
                print(f"streamed {source} as {video_id} ({last_sequence} chunks)")
                return
        except ConnectionClosed:
            attempts += 1
            if attempts > retries:
                raise
            await asyncio.sleep(min(2 ** (attempts - 1), 10))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock IQ9 text-stream publisher")
    parser.add_argument("--url", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--chunk-chars", type=int, default=1024)
    parser.add_argument("--delay", type=float, default=0)
    args = parser.parse_args()
    if args.chunk_chars < 1:
        parser.error("--chunk-chars must be positive")
    asyncio.run(stream_file(
        args.url,
        args.video_id,
        args.input,
        chunk_chars=args.chunk_chars,
        delay=args.delay,
    ))


if __name__ == "__main__":
    main()
