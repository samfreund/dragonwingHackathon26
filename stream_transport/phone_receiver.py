from __future__ import annotations

import asyncio
import json

from websockets.exceptions import ConnectionClosed

from . import PROTOCOL_VERSION
from .phone_broker import PhoneQuery, PhoneQueryBroker
from .protocol import ProtocolError, parse_message, safe_id, video_id


class PhoneReceiver:
    def __init__(
        self,
        broker: PhoneQueryBroker,
        *,
        max_question_chars: int = 16_000,
        result_poll_seconds: float = 0.2,
    ) -> None:
        self.broker = broker
        self.max_question_chars = max_question_chars
        self.result_poll_seconds = result_poll_seconds

    async def handler(self, websocket) -> None:
        device_id: str | None = None
        subscribed: set[str] = set()
        delivered: set[str] = set()
        try:
            try:
                first = parse_message(await asyncio.wait_for(websocket.recv(), timeout=10))
                if first.get("type") != "phone_start":
                    raise ProtocolError("phone_start_required", "First message must be phone_start")
                if first.get("protocol") != PROTOCOL_VERSION:
                    raise ProtocolError(
                        "unsupported_protocol",
                        f"Supported protocol is {PROTOCOL_VERSION}",
                        supported=PROTOCOL_VERSION,
                    )
                device_id = safe_id(first.get("device_id"), "device_id")
                await self._send(websocket, {
                    "type": "phone_ready",
                    "protocol": PROTOCOL_VERSION,
                    "device_id": device_id,
                })
            except ProtocolError as exc:
                await self._send(websocket, exc.payload())
                await websocket.close(code=1008, reason=exc.code)
                return

            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.recv(), timeout=self.result_poll_seconds
                    )
                except asyncio.TimeoutError:
                    raw = None
                except ConnectionClosed:
                    return

                if raw is not None:
                    try:
                        request = await self._handle(websocket, device_id, parse_message(raw))
                        if request is not None:
                            subscribed.add(request.request_id)
                    except ProtocolError as exc:
                        await self._send(websocket, exc.payload())

                for request_id in tuple(subscribed - delivered):
                    query = await asyncio.to_thread(self.broker.get, request_id)
                    if query and query.terminal:
                        await self._send(websocket, query.result_payload())
                        delivered.add(request_id)
        except (ConnectionClosed, asyncio.TimeoutError):
            return

    async def _handle(
        self,
        websocket,
        device_id: str,
        message: dict,
    ) -> PhoneQuery | None:
        kind = message.get("type")
        if kind == "ping":
            await self._send(websocket, {"type": "pong"})
            return None
        if kind == "query":
            question = message.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ProtocolError("invalid_question", "question must be a non-empty string")
            if len(question) > self.max_question_chars:
                raise ProtocolError(
                    "question_too_large",
                    f"question exceeds {self.max_question_chars} characters",
                )
            query = await asyncio.to_thread(
                self.broker.submit,
                safe_id(message.get("request_id"), "request_id"),
                device_id,
                video_id(message.get("video_id")),
                question,
            )
            await self._send(websocket, {
                "type": "query_ack",
                "request_id": query.request_id,
                "status": query.status,
            })
            return query
        if kind == "query_status":
            request_id = safe_id(message.get("request_id"), "request_id")
            query = await asyncio.to_thread(self.broker.get, request_id)
            if query is None:
                raise ProtocolError("unknown_request", f"Unknown request_id {request_id}")
            if query.device_id != device_id:
                raise ProtocolError("forbidden", "request_id belongs to another device")
            if query.terminal:
                await self._send(websocket, query.result_payload())
                return None
            await self._send(websocket, {
                "type": "query_ack",
                "request_id": request_id,
                "status": query.status,
            })
            return query
        raise ProtocolError("unknown_type", f"Unknown message type {kind!r}")

    @staticmethod
    async def _send(websocket, payload: dict) -> None:
        await websocket.send(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
