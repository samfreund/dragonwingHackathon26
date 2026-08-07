from __future__ import annotations

import argparse
import asyncio
import json

from websockets.asyncio.client import connect

from . import PROTOCOL_VERSION


async def ask(
    url: str,
    token: str,
    device_id: str,
    request_id: str,
    video_id: str,
    question: str,
    *,
    reconnect_after_ack: bool = False,
) -> dict:
    async def start(websocket) -> None:
        await websocket.send(json.dumps({
            "type": "phone_start",
            "protocol": PROTOCOL_VERSION,
            "token": token,
            "device_id": device_id,
        }))
        ready = json.loads(await websocket.recv())
        if ready.get("type") != "phone_ready":
            raise RuntimeError(f"Phone connection refused: {ready}")

    async with connect(url) as websocket:
        await start(websocket)
        await websocket.send(json.dumps({
            "type": "query",
            "request_id": request_id,
            "video_id": video_id,
            "question": question,
        }))
        acknowledgement = json.loads(await websocket.recv())
        if acknowledgement.get("type") != "query_ack":
            raise RuntimeError(f"Query was not acknowledged: {acknowledgement}")
        print(json.dumps(acknowledgement))
        if not reconnect_after_ack:
            while True:
                result = json.loads(await websocket.recv())
                if result.get("type") == "query_result":
                    return result

    async with connect(url) as websocket:
        await start(websocket)
        await websocket.send(json.dumps({
            "type": "query_status", "request_id": request_id,
        }))
        while True:
            result = json.loads(await websocket.recv())
            if result.get("type") == "query_result":
                return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock DragonAssist phone")
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--device-id", default="mock-phone")
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--reconnect-after-ack", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(ask(
        args.url,
        args.token,
        args.device_id,
        args.request_id,
        args.video_id,
        args.question,
        reconnect_after_ack=args.reconnect_after_ack,
    ))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
