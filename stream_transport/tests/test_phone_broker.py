from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stream_transport.phone_broker import PhoneQueryBroker
from stream_transport.protocol import ProtocolError


class PhoneBrokerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.broker = PhoneQueryBroker(Path(self.temp.name) / "queries.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_submit_is_idempotent_but_conflicts_are_rejected(self):
        first = self.broker.submit("request-1", "phone-1", "video-1", "What happened?")
        duplicate = self.broker.submit("request-1", "phone-1", "video-1", "What happened?")
        self.assertEqual(first, duplicate)
        with self.assertRaises(ProtocolError) as raised:
            self.broker.submit("request-1", "phone-1", "video-1", "Different question")
        self.assertEqual("request_conflict", raised.exception.code)

    def test_claim_once_and_complete(self):
        self.broker.submit("request-1", "phone-1", "video-1", "What happened?")
        claimed = self.broker.claim_next()
        self.assertEqual("processing", claimed.status)
        self.assertIsNone(self.broker.claim_next())
        completed = self.broker.complete(
            "request-1",
            "A person entered.",
            route="local",
            sources=[{"start_s": 0, "end_s": 5}],
        )
        self.assertTrue(completed.terminal)
        self.assertEqual("A person entered.", completed.answer)
        self.assertEqual("local", completed.result_payload()["route"])

    def test_failure_is_terminal(self):
        self.broker.submit("request-1", "phone-1", "video-1", "What happened?")
        failed = self.broker.fail("request-1", "No context")
        self.assertEqual("failed", failed.status)
        self.assertEqual("No context", failed.result_payload()["error"])


if __name__ == "__main__":
    unittest.main()
