from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from stream_transport.server import ServerConfig, build


class CombinedServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        config = ServerConfig(
            host="127.0.0.1",
            port=0,
            stream_token="iq9-secret",
            phone_token="phone-secret",
            storage_root=Path(self.temp.name),
        )
        self.combined, self.broker = build(config)
        self.server = await serve(self.combined.handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        self.temp.cleanup()

    def url(self, path: str) -> str:
        return f"ws://127.0.0.1:{self.port}{path}"

    async def phone_start(self, websocket, token="phone-secret", device="phone-1"):
        await websocket.send(json.dumps({
            "type": "phone_start",
            "protocol": 1,
            "token": token,
            "device_id": device,
        }))
        return json.loads(await websocket.recv())

    async def submit(self, websocket, request_id="request-1"):
        await websocket.send(json.dumps({
            "type": "query",
            "request_id": request_id,
            "video_id": "video-1",
            "question": "What happened?",
        }))
        return json.loads(await websocket.recv())

    async def test_paths_and_separate_tokens(self):
        async with connect(self.url("/v1/phone")) as phone:
            self.assertEqual("auth", (await self.phone_start(phone, token="iq9-secret"))["code"])

        async with connect(self.url("/v1/iq9")) as iq9:
            await iq9.send(json.dumps({
                "type": "start", "protocol": 1,
                "token": "phone-secret", "video_id": "video-1",
            }))
            self.assertEqual("auth", json.loads(await iq9.recv())["code"])

        async with connect(self.url("/not-real")) as unknown:
            self.assertEqual("unknown_path", json.loads(await unknown.recv())["code"])

    async def test_query_ack_then_pushed_result(self):
        async with connect(self.url("/v1/phone")) as phone:
            self.assertEqual("phone_ready", (await self.phone_start(phone))["type"])
            acknowledgement = await self.submit(phone)
            self.assertEqual("pending", acknowledgement["status"])

            claimed = await asyncio.to_thread(self.broker.claim_next)
            await asyncio.to_thread(
                self.broker.complete,
                claimed.request_id,
                "A person entered.",
                route="local",
                sources=[{"start_s": 0, "end_s": 5}],
            )
            result = json.loads(await asyncio.wait_for(phone.recv(), timeout=2))
            self.assertEqual("query_result", result["type"])
            self.assertEqual("A person entered.", result["answer"])

    async def test_disconnect_then_status_returns_stored_result(self):
        async with connect(self.url("/v1/phone")) as phone:
            await self.phone_start(phone)
            await self.submit(phone, "request-reconnect")

        claimed = await asyncio.to_thread(self.broker.claim_next)
        await asyncio.to_thread(
            self.broker.complete, claimed.request_id, "Stored answer", route="cloud"
        )

        async with connect(self.url("/v1/phone")) as phone:
            await self.phone_start(phone)
            await phone.send(json.dumps({
                "type": "query_status", "request_id": "request-reconnect",
            }))
            result = json.loads(await phone.recv())
            self.assertEqual("complete", result["status"])
            self.assertEqual("Stored answer", result["answer"])

    async def test_existing_iq9_protocol_remains_available_at_root(self):
        async with connect(self.url("/")) as iq9:
            await iq9.send(json.dumps({
                "type": "start", "protocol": 1,
                "token": "iq9-secret", "video_id": "video-1",
            }))
            self.assertEqual("started", json.loads(await iq9.recv())["type"])
            await iq9.send(json.dumps({
                "type": "text", "video_id": "video-1",
                "sequence": 1, "text": "context",
            }))
            self.assertEqual("ack", json.loads(await iq9.recv())["type"])


if __name__ == "__main__":
    unittest.main()
