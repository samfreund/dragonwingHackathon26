from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .protocol import ProtocolError, safe_id, video_id


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
    """File-based boundary between phone WebSockets and the laptop worker.

    The integration contract is `received/<video_id>/query.txt`. Small JSON
    state files under `.phone_requests` replace SQLite so the separate server
    and worker processes can still exchange status and answers.
    """

    def __init__(self, path: Path) -> None:
        # Keep the historical constructor shape; the old database filename is
        # ignored and only its parent storage directory is used.
        self.root = path.resolve().parent
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_root = self.root / ".phone_requests"
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

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

        with self._lock:
            existing = self._read(request_id)
            if existing is not None:
                if (
                    existing.device_id != device_id
                    or existing.video_id != video
                    or existing.question != question
                ):
                    raise ProtocolError(
                        "request_conflict",
                        "request_id already exists with different query data",
                    )
                return existing

            query = PhoneQuery(request_id, device_id, video, question, "pending")
            self._write_query_text(video, question)
            self._write(query)
            return query

    def _write_query_text(self, video: str, question: str) -> None:
        directory = self.root / video
        directory.mkdir(parents=True, exist_ok=True)
        self._write_bytes(directory / "query.txt", (question + "\n").encode("utf-8"))

    def _state_path(self, request_id: str) -> Path:
        return self.state_root / f"{request_id}.json"

    def _read(self, request_id: str) -> PhoneQuery | None:
        path = self._state_path(request_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["sources"] = tuple(payload.get("sources") or ())
        return PhoneQuery(**payload)

    def _write(self, query: PhoneQuery) -> None:
        payload = json.dumps(asdict(query), ensure_ascii=False, separators=(",", ":"))
        self._write_bytes(self._state_path(query.request_id), payload.encode("utf-8"))

    @staticmethod
    def _write_bytes(path: Path, payload: bytes) -> None:
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        descriptor = os.open(temporary, os.O_TRUNC | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)

    def get(self, request_id: str) -> PhoneQuery | None:
        request_id = safe_id(request_id, "request_id")
        with self._lock:
            return self._read(request_id)

    def claim_next(self) -> PhoneQuery | None:
        with self._lock:
            for path in sorted(self.state_root.glob("*.json"), key=lambda item: item.stat().st_mtime_ns):
                query = self._read(path.stem)
                if query is not None and query.status == "pending":
                    claimed = replace(query, status="processing")
                    self._write(claimed)
                    return claimed
            return None

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
            sources=tuple(sources or []),
            error=None,
        )

    def fail(self, request_id: str, error: str) -> PhoneQuery:
        return self._finish(
            request_id,
            status="failed",
            answer=None,
            route=None,
            sources=(),
            error=str(error),
        )

    def _finish(self, request_id: str, **values) -> PhoneQuery:
        request_id = safe_id(request_id, "request_id")
        with self._lock:
            query = self._read(request_id)
            if query is None:
                raise KeyError(f"Unknown request_id {request_id}")
            if query.terminal:
                return query
            updated = replace(query, **values)
            self._write(updated)
            return updated
