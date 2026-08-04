"""Thin client for GenieX's OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests

from .config import Settings, settings as default_settings
from .media import to_data_url


class VLMError(RuntimeError):
    """Raised when the VLM server rejects a request or can't be reached."""


@dataclass
class Answer:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict, repr=False)


def image_part(path: Path) -> dict:
    """An OpenAI-style image content part, inlined as a base64 data URL."""
    return {"type": "image_url", "image_url": {"url": to_data_url(path)}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


class VLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
        })

    # ---------------------------------------------------------------- health

    def is_up(self) -> bool:
        try:
            resp = self._session.get(
                self.settings.models_url, timeout=self.settings.connect_timeout
            )
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def loaded_models(self) -> list[str]:
        try:
            resp = self._session.get(
                self.settings.models_url, timeout=self.settings.connect_timeout
            )
            resp.raise_for_status()
            return [m.get("id", "") for m in resp.json().get("data", [])]
        except (requests.RequestException, ValueError):
            return []

    # ------------------------------------------------------------- inference

    def ask(
        self,
        content: list[dict],
        *,
        system: str | None = None,
        history: list[dict] | None = None,
        stream: bool = False,
        on_token=None,
    ) -> Answer:
        """Send one user turn (text + any images) and return the reply.

        `history` is a list of prior {role, content} messages, used for
        follow-up questions. Keep it text-only: GenieX fails with
        `SDKError(Multimodal generation failed)` if image parts appear
        anywhere but the current turn.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": content})

        payload = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "stream": stream,
        }

        started = time.monotonic()
        try:
            resp = self._session.post(
                self.settings.chat_url,
                data=json.dumps(payload),
                timeout=(self.settings.connect_timeout, self.settings.read_timeout),
                stream=stream,
            )
        except requests.ConnectionError as exc:
            raise VLMError(
                f"Cannot reach the VLM server at {self.settings.base_url}. "
                "Start it with:  geniex serve"
            ) from exc
        except requests.Timeout as exc:
            raise VLMError(
                f"Request timed out after {self.settings.read_timeout}s. "
                "Try fewer frames, or raise VLMQA_READ_TIMEOUT."
            ) from exc

        if resp.status_code >= 400:
            raise VLMError(f"Server returned {resp.status_code}: {_error_detail(resp)}")

        if stream:
            text = "".join(self._stream_tokens(resp, on_token))
            return Answer(text=text.strip(), latency_s=time.monotonic() - started)

        try:
            body = resp.json()
        except ValueError as exc:
            raise VLMError(f"Malformed JSON from server: {resp.text[:200]}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise VLMError(f"Server returned no choices: {json.dumps(body)[:300]}")

        usage = body.get("usage") or {}
        return Answer(
            text=(choices[0].get("message", {}).get("content") or "").strip(),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_s=time.monotonic() - started,
            raw=body,
        )

    @staticmethod
    def _stream_tokens(resp: requests.Response, on_token) -> Iterator[str]:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if on_token:
                        on_token(piece)
                    yield piece


def _error_detail(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:300]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return err.get("message") or json.dumps(err)[:300]
        if err:
            return str(err)[:300]
    return json.dumps(body)[:300]
