from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stream_transport.protocol import ProtocolError
from stream_transport.storage import TextStore


class SimulatedCrash(BaseException):
    pass


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_verbatim_unicode_duplicate_and_completion(self):
        store = TextStore(self.root)
        first = store.append("video-1", 1, "hello\n")
        second = store.append("video-1", 2, "世界")
        duplicate = store.append("video-1", 2, "ignored retry body")
        state = store.finish("video-1", 2)

        self.assertEqual(
            "hello\n世界",
            store.context_path("video-1").read_text(encoding="utf-8"),
        )
        self.assertFalse(first.duplicate)
        self.assertFalse(second.duplicate)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(second.bytes_written, duplicate.bytes_written)
        self.assertTrue(state.complete)

    def test_gap_rejected_without_write(self):
        store = TextStore(self.root)
        with self.assertRaises(ProtocolError) as raised:
            store.append("video-1", 2, "lost sequence one")
        self.assertEqual("sequence_gap", raised.exception.code)
        self.assertEqual(0, store.state("video-1").bytes_written)

    def test_recover_crash_before_file_append(self):
        def crash():
            raise SimulatedCrash()

        store = TextStore(self.root, before_file_append=crash)
        with self.assertRaises(SimulatedCrash):
            store.append("video-1", 1, "recovered")

        recovered = TextStore(self.root)
        self.assertEqual("recovered", recovered.context_path("video-1").read_text())
        self.assertEqual(1, recovered.state("video-1").last_sequence)

    def test_recover_crash_after_file_sync(self):
        def crash():
            raise SimulatedCrash()

        store = TextStore(self.root, after_file_sync=crash)
        with self.assertRaises(SimulatedCrash):
            store.append("video-1", 1, "only once")

        recovered = TextStore(self.root)
        self.assertEqual("only once", recovered.context_path("video-1").read_text())
        self.assertEqual(1, recovered.state("video-1").last_sequence)

    def test_invalid_video_id(self):
        store = TextStore(self.root)
        for value in ("../escape", "/absolute", "spaces are bad", ""):
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                store.state(value)


if __name__ == "__main__":
    unittest.main()
