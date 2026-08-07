from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from stream_transport.mock_iq9 import stream_file
from stream_transport.receiver import StreamReceiver
from stream_transport.storage import TextStore


class ReceiverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TextStore(Path(self.temp.name))
        self.receiver = StreamReceiver(self.store)
        self.server = await serve(self.receiver.handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        self.temp.cleanup()

    async def start(self, websocket, video="video-1"):
        await websocket.send(json.dumps({
            "type": "start", "protocol": 1, "video_id": video,
        }))
        return json.loads(await websocket.recv())

    async def test_stream_reconnect_and_duplicate(self):
        async with connect(self.url) as websocket:
            started = await self.start(websocket)
            self.assertEqual(1, started["next_sequence"])
            await websocket.send(json.dumps({
                "type": "text", "video_id": "video-1", "sequence": 1, "text": "one ",
            }))
            self.assertEqual("ack", json.loads(await websocket.recv())["type"])

        async with connect(self.url) as websocket:
            started = await self.start(websocket)
            self.assertEqual(2, started["next_sequence"])
            await websocket.send(json.dumps({
                "type": "text", "video_id": "video-1", "sequence": 1, "text": "duplicate",
            }))
            duplicate = json.loads(await websocket.recv())
            self.assertTrue(duplicate["duplicate"])
            await websocket.send(json.dumps({
                "type": "text", "video_id": "video-1", "sequence": 2, "text": "two",
            }))
            await websocket.recv()

        self.assertEqual("one two", self.store.context_path("video-1").read_text())

    async def test_gap_mismatch_and_path_validation(self):
        async with connect(self.url) as websocket:
            error = await self.start(websocket, video="../escape")
            self.assertEqual("invalid_video_id", error["code"])

        async with connect(self.url) as websocket:
            await self.start(websocket)
            await websocket.send(json.dumps({
                "type": "text", "video_id": "other", "sequence": 1, "text": "x",
            }))
            self.assertEqual("video_mismatch", json.loads(await websocket.recv())["code"])
            await websocket.send(json.dumps({
                "type": "text", "video_id": "video-1", "sequence": 2, "text": "x",
            }))
            self.assertEqual("sequence_gap", json.loads(await websocket.recv())["code"])

    async def test_same_video_busy_but_different_video_concurrent(self):
        async with connect(self.url) as first, connect(self.url) as second, connect(self.url) as third:
            self.assertEqual("started", (await self.start(first))["type"])
            self.assertEqual("video_busy", (await self.start(second))["code"])
            self.assertEqual("started", (await self.start(third, video="video-2"))["type"])

    async def test_end_is_optional_and_completion_is_persistent(self):
        async with connect(self.url) as websocket:
            await self.start(websocket)
            await websocket.send(json.dumps({
                "type": "text", "video_id": "video-1", "sequence": 1, "text": "done",
            }))
            await websocket.recv()
            await websocket.send(json.dumps({
                "type": "end", "video_id": "video-1", "sequence": 1,
            }))
            reply = json.loads(await websocket.recv())
            self.assertTrue(reply["complete"])

        self.assertTrue(self.store.state("video-1").complete)

    async def test_mock_sender_is_byte_for_byte_verbatim(self):
        source = Path(self.temp.name) / "source.txt"
        expected = "first line\r\nUnicode: 世界\nno final newline"
        source.write_text(expected, encoding="utf-8", newline="")

        await stream_file(
            self.url,
            "mock-video",
            source,
            chunk_chars=3,
        )

        with self.store.context_path("mock-video").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            received = handle.read()
        self.assertEqual(expected, received)


if __name__ == "__main__":
    unittest.main()
