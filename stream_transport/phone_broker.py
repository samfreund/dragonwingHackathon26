from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .protocol import ProtocolError, safe_id, video_id


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;

CREATE TABLE IF NOT EXISTS phone_queries (
    request_id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    answer TEXT,
    route TEXT,
    sources_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass(frozen=True)
class PhoneQuery:
    request_id: str
    device_id: str
    video_id: str
    question: str
    status: str
    answer: str | None = None
    route: str | None = None
    sources: tuple[dict, ...] = ()
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in {"complete", "failed"}

    def result_payload(self) -> dict:
        payload = {
            "type": "query_result",
            "request_id": self.request_id,
            "video_id": self.video_id,
            "status": self.status,
        }
        if self.status == "complete":
            payload.update(answer=self.answer or "", route=self.route, sources=list(self.sources))
        else:
            payload["error"] = self.error or "Query failed"
        return payload


class PhoneQueryBroker:
    """Durable boundary between phone WebSockets and the laptop QA worker."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def submit(
        self,
        request_id: str,
        device_id: str,
        video: str,
        question: str,
    ) -> PhoneQuery:
        request_id = safe_id(request_id, "request_id")
        device_id = safe_id(device_id, "device_id")
        video = video_id(video)
        if not isinstance(question, str) or not question.strip():
            raise ProtocolError("invalid_question", "question must be a non-empty string")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM phone_queries WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["device_id"] != device_id
                    or existing["video_id"] != video
                    or existing["question"] != question
                ):
                    raise ProtocolError(
                        "request_conflict",
                        "request_id already exists with different query data",
                    )
                return self._from_row(existing)
            connection.execute(
                """INSERT INTO phone_queries(request_id, device_id, video_id, question)
                   VALUES (?, ?, ?, ?)""",
                (request_id, device_id, video, question),
            )
            row = connection.execute(
                "SELECT * FROM phone_queries WHERE request_id=?", (request_id,)
            ).fetchone()
        return self._from_row(row)

    def get(self, request_id: str) -> PhoneQuery | None:
        request_id = safe_id(request_id, "request_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM phone_queries WHERE request_id=?", (request_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def claim_next(self) -> PhoneQuery | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM phone_queries WHERE status='pending'
                   ORDER BY created_at, rowid LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE phone_queries SET status='processing', updated_at=CURRENT_TIMESTAMP
                   WHERE request_id=? AND status='pending'""",
                (row["request_id"],),
            )
            claimed = connection.execute(
                "SELECT * FROM phone_queries WHERE request_id=?", (row["request_id"],)
            ).fetchone()
        return self._from_row(claimed)

    def complete(
        self,
        request_id: str,
        answer: str,
        *,
        route: str | None = None,
        sources: list[dict] | None = None,
    ) -> PhoneQuery:
        return self._finish(
            request_id,
            status="complete",
            answer=str(answer),
            route=route,
            sources_json=json.dumps(sources or [], ensure_ascii=False),
            error=None,
        )

    def fail(self, request_id: str, error: str) -> PhoneQuery:
        return self._finish(
            request_id,
            status="failed",
            answer=None,
            route=None,
            sources_json="[]",
            error=str(error),
        )

    def _finish(self, request_id: str, **values) -> PhoneQuery:
        request_id = safe_id(request_id, "request_id")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM phone_queries WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown request_id {request_id}")
            if row["status"] in {"complete", "failed"}:
                return self._from_row(row)
            connection.execute(
                """UPDATE phone_queries SET status=?, answer=?, route=?, sources_json=?, error=?,
                   updated_at=CURRENT_TIMESTAMP WHERE request_id=?""",
                (
                    values["status"], values["answer"], values["route"],
                    values["sources_json"], values["error"], request_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM phone_queries WHERE request_id=?", (request_id,)
            ).fetchone()
        return self._from_row(updated)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> PhoneQuery:
        return PhoneQuery(
            request_id=row["request_id"],
            device_id=row["device_id"],
            video_id=row["video_id"],
            question=row["question"],
            status=row["status"],
            answer=row["answer"],
            route=row["route"],
            sources=tuple(json.loads(row["sources_json"])),
            error=row["error"],
        )
