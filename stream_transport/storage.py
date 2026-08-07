from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from .protocol import ProtocolError, video_id as validate_video_id


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;

CREATE TABLE IF NOT EXISTS streams (
    video_id TEXT PRIMARY KEY,
    last_sequence INTEGER NOT NULL DEFAULT 0,
    bytes_written INTEGER NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 0,
    recovery_error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pending_appends (
    video_id TEXT PRIMARY KEY,
    sequence INTEGER NOT NULL,
    byte_offset INTEGER NOT NULL,
    payload BLOB NOT NULL,
    sha256 TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class StreamState:
    video_id: str
    last_sequence: int
    bytes_written: int
    complete: bool


@dataclass(frozen=True)
class AppendResult:
    sequence: int
    bytes_written: int
    duplicate: bool


class TextStore:
    """Crash-recoverable, append-only text files with sequence deduplication."""

    def __init__(
        self,
        root: Path,
        *,
        before_file_append: Callable[[], None] | None = None,
        after_file_sync: Callable[[], None] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "receiver.sqlite3"
        self.before_file_append = before_file_append
        self.after_file_sync = after_file_sync
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}
        with self._connect() as connection:
            connection.executescript(SCHEMA)
        self.recover_all()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _lock(self, video: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(video, threading.RLock())

    def context_path(self, video: str) -> Path:
        video = validate_video_id(video)
        directory = self.root / video
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "context.txt"

    def state(self, video: str) -> StreamState:
        video = validate_video_id(video)
        with self._lock(video):
            self._recover(video)
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO streams(video_id) VALUES (?) ON CONFLICT DO NOTHING",
                    (video,),
                )
                row = connection.execute(
                    "SELECT * FROM streams WHERE video_id=?", (video,)
                ).fetchone()
            if row["recovery_error"]:
                raise ProtocolError("recovery_error", row["recovery_error"])
            return StreamState(
                video, row["last_sequence"], row["bytes_written"], bool(row["complete"])
            )

    def append(self, video: str, seq: int, text: str) -> AppendResult:
        video = validate_video_id(video)
        if not isinstance(text, str):
            raise ProtocolError("invalid_text", "text must be a string")
        payload = text.encode("utf-8")
        with self._lock(video):
            state = self.state(video)
            if seq <= state.last_sequence:
                return AppendResult(seq, state.bytes_written, True)
            if state.complete:
                raise ProtocolError("stream_complete", "This video stream is already complete")
            expected = state.last_sequence + 1
            if seq != expected:
                raise ProtocolError(
                    "sequence_gap",
                    f"Expected sequence {expected}, received {seq}",
                    expected=expected,
                    received=seq,
                )

            path = self.context_path(video)
            actual_size = path.stat().st_size if path.exists() else 0
            if actual_size != state.bytes_written:
                self._fail(video, f"context.txt is {actual_size} bytes; expected {state.bytes_written}")
                raise ProtocolError("recovery_error", "Context file size does not match durable state")

            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO pending_appends(video_id, sequence, byte_offset, payload, sha256)
                       VALUES (?, ?, ?, ?, ?)""",
                    (video, seq, state.bytes_written, payload, hashlib.sha256(payload).hexdigest()),
                )

            if self.before_file_append:
                self.before_file_append()
            self._write_and_sync(path, state.bytes_written, payload)
            if self.after_file_sync:
                self.after_file_sync()
            total = state.bytes_written + len(payload)
            self._commit_pending(video, seq, total)
            return AppendResult(seq, total, False)

    def finish(self, video: str, final_sequence: int) -> StreamState:
        video = validate_video_id(video)
        with self._lock(video):
            state = self.state(video)
            if final_sequence != state.last_sequence:
                raise ProtocolError(
                    "sequence_gap",
                    f"Cannot end at sequence {final_sequence}; last committed is {state.last_sequence}",
                    expected=state.last_sequence,
                    received=final_sequence,
                )
            with self._connect() as connection:
                connection.execute(
                    "UPDATE streams SET complete=1, updated_at=CURRENT_TIMESTAMP WHERE video_id=?",
                    (video,),
                )
            return StreamState(video, state.last_sequence, state.bytes_written, True)

    def recover_all(self) -> None:
        with self._connect() as connection:
            videos = [row[0] for row in connection.execute("SELECT video_id FROM pending_appends")]
        for video in videos:
            with self._lock(video):
                self._recover(video)

    def _recover(self, video: str) -> None:
        with self._connect() as connection:
            pending = connection.execute(
                "SELECT * FROM pending_appends WHERE video_id=?", (video,)
            ).fetchone()
            stream = connection.execute(
                "SELECT * FROM streams WHERE video_id=?", (video,)
            ).fetchone()
        if pending is None:
            return
        if stream is None:
            self._fail(video, "Pending append has no stream state")
            return

        seq = pending["sequence"]
        offset = pending["byte_offset"]
        payload = bytes(pending["payload"])
        if seq != stream["last_sequence"] + 1 or offset != stream["bytes_written"]:
            self._fail(video, "Pending append does not follow committed stream state")
            return
        if hashlib.sha256(payload).hexdigest() != pending["sha256"]:
            self._fail(video, "Pending append payload hash is invalid")
            return

        path = self.context_path(video)
        size = path.stat().st_size if path.exists() else 0
        if size == offset:
            self._write_and_sync(path, offset, payload)
        elif size == offset + len(payload):
            with path.open("rb") as source:
                source.seek(offset)
                existing = source.read(len(payload))
            if existing != payload:
                self._fail(video, "File bytes conflict with pending append")
                return
        else:
            self._fail(video, f"Cannot recover pending append from file size {size}")
            return
        self._commit_pending(video, seq, offset + len(payload))

    @staticmethod
    def _write_and_sync(path: Path, offset: int, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+b" if path.exists() else "w+b"
        with path.open(mode) as destination:
            destination.seek(0, os.SEEK_END)
            if destination.tell() != offset:
                raise ProtocolError("recovery_error", "Context file changed during append")
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())

    def _commit_pending(self, video: str, seq: int, total: int) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE streams SET last_sequence=?, bytes_written=?,
                   updated_at=CURRENT_TIMESTAMP WHERE video_id=?""",
                (seq, total, video),
            )
            connection.execute("DELETE FROM pending_appends WHERE video_id=?", (video,))

    def _fail(self, video: str, message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE streams SET recovery_error=?, updated_at=CURRENT_TIMESTAMP WHERE video_id=?",
                (message, video),
            )
