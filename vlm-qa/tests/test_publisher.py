"""Publisher tests that do not need the board.

Captioning is stubbed out throughout: what is under test here is the window
arithmetic and the conversation with `stream_transport`, not the VLM. The
receiver is the real one, so a change to either side's protocol breaks these.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "vlm-qa"))

from websockets.asyncio.client import connect  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402

from stream_transport.receiver import StreamReceiver  # noqa: E402
from stream_transport.storage import TextStore  # noqa: E402
from vlmqa.publisher import PublishError, Window, plan_windows, publish  # noqa: E402


class PlanWindowsTests(unittest.TestCase):
    def test_exact_division(self):
        windows = plan_windows(30.0, 10.0)
        self.assertEqual(3, len(windows))
        self.assertEqual([(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)],
                         [(w.start_s, w.end_s) for w in windows])
        self.assertEqual([1, 2, 3], [w.sequence for w in windows])

    def test_short_tail_is_folded_into_the_previous_window(self):
        # 21s at 10s windows would leave a 1s sliver with nothing to caption.
        windows = plan_windows(21.0, 10.0)
        self.assertEqual(2, len(windows))
        self.assertEqual(21.0, windows[-1].end_s)

    def test_tail_worth_keeping_stays_its_own_window(self):
        windows = plan_windows(26.0, 10.0)
        self.assertEqual(3, len(windows))
        self.assertEqual((20.0, 26.0), (windows[-1].start_s, windows[-1].end_s))

    def test_video_shorter_than_one_window(self):
        windows = plan_windows(4.0, 10.0)
        self.assertEqual(1, len(windows))
        self.assertEqual((0.0, 4.0), (windows[0].start_s, windows[0].end_s))

    def test_labels_are_absolute_timestamps(self):
        self.assertEqual("01:05-01:15", Window(0, 65.0, 75.0).label)

    def test_rejects_unusable_input(self):
        for duration, window in ((0.0, 10.0), (-1.0, 10.0), (10.0, 0.0)):
            with self.assertRaises(PublishError):
                plan_windows(duration, window)


class PublishAgainstRealReceiverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TextStore(Path(self.temp.name))
        self.receiver = StreamReceiver(self.store)
        self.server = await serve(self.receiver.handler, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        self.media = Path(self.temp.name) / "clip.mp4"
        self.media.write_bytes(b"not really a video; probe_video is patched")

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        self.temp.cleanup()

    def _patches(self, duration_s: float = 30.0):
        """Stand in for ffprobe and the NPU."""
        return (
            mock.patch("vlmqa.publisher.classify", return_value="video"),
            mock.patch("vlmqa.publisher.probe_video",
                       return_value=mock.Mock(duration_s=duration_s)),
            mock.patch("vlmqa.publisher.caption",
                       side_effect=lambda media, window, **kw: f"[{window.label}] caption\n"),
        )

    async def test_captions_arrive_in_order_and_land_verbatim(self):
        with self._patches()[0], self._patches()[1], self._patches()[2]:
            sent = await publish(self.media, self.url, "clip-1", window_s=10.0)

        self.assertEqual(3, sent)
        self.assertEqual(
            "[00:00-00:10] caption\n[00:10-00:20] caption\n[00:20-00:30] caption\n",
            self.store.context_path("clip-1").read_text(encoding="utf-8"),
        )

    async def test_resume_skips_windows_the_receiver_already_has(self):
        classify_p, probe_p, caption_p = self._patches()
        with classify_p, probe_p, caption_p:
            await publish(self.media, self.url, "clip-2", window_s=10.0)

        # A second run over the same video_id must caption nothing: every
        # sequence is already durable, so the NPU is never woken.
        captioned: list[str] = []

        def track(media, window, **kw):
            captioned.append(window.label)
            return f"[{window.label}] caption\n"

        with self._patches()[0], self._patches()[1], \
                mock.patch("vlmqa.publisher.caption", side_effect=track):
            sent = await publish(self.media, self.url, "clip-2", window_s=10.0)

        self.assertEqual(0, sent)
        self.assertEqual([], captioned)

    async def test_partial_stream_resumes_at_the_missing_window(self):
        # Land window 1 by hand, then let the publisher fill in 2 and 3.
        async with connect(self.url) as socket:
            import json
            await socket.send(json.dumps({
                "type": "start", "protocol": 1, "video_id": "clip-3",
            }))
            await socket.recv()
            await socket.send(json.dumps({
                "type": "text", "video_id": "clip-3",
                "sequence": 1, "text": "[00:00-00:10] partial\n",
            }))
            await socket.recv()

        captioned: list[str] = []

        def track(media, window, **kw):
            captioned.append(window.label)
            return f"[{window.label}] caption\n"

        with self._patches()[0], self._patches()[1], \
                mock.patch("vlmqa.publisher.caption", side_effect=track):
            sent = await publish(self.media, self.url, "clip-3", window_s=10.0)

        self.assertEqual(2, sent)
        self.assertEqual(["00:10-00:20", "00:20-00:30"], captioned)
        self.assertEqual(
            "[00:00-00:10] partial\n[00:10-00:20] caption\n[00:20-00:30] caption\n",
            self.store.context_path("clip-3").read_text(encoding="utf-8"),
        )

    async def test_an_image_is_refused(self):
        with mock.patch("vlmqa.publisher.classify", return_value="image"):
            with self.assertRaises(PublishError):
                await publish(self.media, self.url, "clip-4")


if __name__ == "__main__":
    unittest.main()
