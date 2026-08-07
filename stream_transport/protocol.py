from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from . import PROTOCOL_VERSION


VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def payload(self) -> dict[str, Any]:
        return {"type": "error", "code": self.code, "message": self.message, **self.details}


def parse_message(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raise ProtocolError("binary_not_supported", "Expected a JSON text frame")
    try:
        message = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("invalid_json", f"Invalid JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("invalid_message", "Expected a JSON object")
    return message


def video_id(value: Any) -> str:
    return safe_id(value, "video_id")


def safe_id(value: Any, field: str) -> str:
    value = str(value or "")
    if not VIDEO_ID_RE.fullmatch(value):
        raise ProtocolError(
            f"invalid_{field}",
            f"{field} must be 1-128 filename-safe characters",
        )
    return value


def sequence(value: Any, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ProtocolError("invalid_sequence", "sequence must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("invalid_sequence", "sequence must be an integer") from exc
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise ProtocolError("invalid_sequence", f"sequence must be >= {minimum}")
    return result


@dataclass(frozen=True)
class Start:
    video_id: str
    token: str


def parse_start(message: dict[str, Any]) -> Start:
    if message.get("type") != "start":
        raise ProtocolError("start_required", "First message must have type 'start'")
    if message.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(
            "unsupported_protocol",
            f"Supported protocol is {PROTOCOL_VERSION}",
            supported=PROTOCOL_VERSION,
        )
    return Start(video_id(message.get("video_id")), str(message.get("token") or ""))
