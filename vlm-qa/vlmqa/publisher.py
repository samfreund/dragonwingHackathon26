"""Publish a running commentary of a video to the laptop's context store.

The rest of vlm-qa answers questions about a file that already exists. This
module turns the same machinery into a *producer*: a video is walked in fixed
time windows, each window is captioned by the VLM on the NPU, and each caption
is appended to one durable file on the laptop.

The far end is `stream_transport.server` at ws://<laptop>:8001/v1/iq9, which
stores text verbatim and acknowledges a sequence number only once the bytes
are on disk. That makes a dropped socket cheap: on reconnect the server
reports the `next_sequence` it wants, and captioning resumes from that window
rather than re-running the ones the laptop already has. Captions are the
expensive half -- each is a full multimodal generation on the board, seconds
rather than milliseconds -- so windows are captioned one at a time and sent as
they finish, never batched to the end where a disconnect would lose all of them.

Window N carries sequence N+1; the mapping is fixed so that resuming is a
matter of arithmetic rather than bookkeeping on the wire.

    python -m vlmqa publish --media clip.mp4 \
      --url ws://qcworkshop3:8001/v1/iq9 --video-id loading-dock
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from .config import Settings, settings as default_settings
from .media import MediaError, classify, probe_video
from .qa import ask_about

PROTOCOL_VERSION = 1

# Deliberately not a question. The laptop's ask.py answers questions later,
# against whatever this leaves behind, so the useful thing to store is a dense
# factual record rather than an interpretation that has already thrown detail away.
DEFAULT_PROMPT = (
    "Describe what happens in these frames in two or three plain sentences. "
    "Report only what is visible: people, objects, actions, and any text or "
    "signage you can read. Do not speculate about intent or what happens next."
)


class PublishError(RuntimeError):
    """The receiver refused the stream, or the video cannot be walked."""


@dataclass(frozen=True)
class Window:
    """One slice of the video, and the sequence number its caption carries."""

    index: int
    start_s: float
    end_s: float

    @property
    def sequence(self) -> int:
        return self.index + 1

    @property
    def label(self) -> str:
        return f"{_stamp(self.start_s)}-{_stamp(self.end_s)}"


def _stamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def plan_windows(duration_s: float, window_s: float) -> list[Window]:
    """Split a duration into consecutive windows, the last one short if needed."""
    if duration_s <= 0:
        raise PublishError("Video has no measurable duration")
    if window_s <= 0:
        raise PublishError("Window length must be positive")

    windows: list[Window] = []
    start = 0.0
    while start < duration_s:
        end = min(start + window_s, duration_s)
        # A sliver left by an inexact division has too few distinct frames to
        # caption usefully; fold it into the window before it instead.
        if duration_s - end < window_s * 0.25 and windows:
            end = duration_s
        windows.append(Window(len(windows), start, end))
        start = end
    return windows


def caption(
    media: Path,
    window: Window,
    *,
    prompt: str = DEFAULT_PROMPT,
    frames: int | None = None,
    strategy: str = "frames",
    settings: Settings | None = None,
) -> str:
    """Caption one window. Blocking: this is where the NPU time goes."""
    answer, _ = ask_about(
        media,
        prompt,
        frames=frames,
        strategy=strategy,
        settings=settings or default_settings,
        window=(window.start_s, window.end_s),
    )
    text = " ".join(answer.text.split())
    return f"[{window.label}] {text}\n"


async def publish(
    media: Path,
    url: str,
    video_id: str,
    *,
    window_s: float = 10.0,
    prompt: str = DEFAULT_PROMPT,
    frames: int | None = None,
    strategy: str = "frames",
    settings: Settings | None = None,
    retries: int = 5,
    on_caption=None,
    gate: asyncio.Lock | None = None,
) -> int:
    """Caption `media` window by window and stream the text to the laptop.

    Returns the number of windows this call actually captioned, which is fewer
    than the total when the receiver already held part of the stream.

    `gate` is the NPU lock when this runs inside `vlmqa serve`. It is taken and
    released per window rather than held for the whole video, so a phone asking
    a question waits for one caption to finish rather than for the entire clip.
    """
    if classify(media) != "video":
        raise PublishError(f"{media.name} is not a video; publish needs one to walk")

    cfg = settings or default_settings
    info = probe_video(media)
    windows = plan_windows(info.duration_s, window_s)
    published = 0
    attempts = 0

    while True:
        try:
            async with connect(url, max_size=1024 * 1024) as socket:
                await socket.send(json.dumps({
                    "type": "start",
                    "protocol": PROTOCOL_VERSION,
                    "video_id": video_id,
                }))
                started = json.loads(await socket.recv())
                if started.get("type") != "started":
                    raise PublishError(f"Receiver refused the stream: {started}")
                next_sequence = int(started["next_sequence"])

                last_sequence = 0
                for window in windows:
                    last_sequence = window.sequence
                    if window.sequence < next_sequence:
                        continue  # the laptop already has this one

                    # to_thread keeps the socket's pings answered while the NPU
                    # works; a caption can outlast the server's ping timeout.
                    async with (gate or contextlib.nullcontext()):
                        text = await asyncio.to_thread(
                            caption,
                            media,
                            window,
                            prompt=prompt,
                            frames=frames,
                            strategy=strategy,
                            settings=cfg,
                        )
                    await socket.send(json.dumps({
                        "type": "text",
                        "video_id": video_id,
                        "sequence": window.sequence,
                        "text": text,
                    }, ensure_ascii=False))
                    reply = json.loads(await socket.recv())
                    if reply.get("type") != "ack" or reply.get("sequence") != window.sequence:
                        raise PublishError(f"Unexpected receiver response: {reply}")

                    published += 1
                    if on_caption is not None:
                        on_caption(window, text)

                await socket.send(json.dumps({
                    "type": "end",
                    "video_id": video_id,
                    "sequence": last_sequence,
                }))
                reply = json.loads(await socket.recv())
                if reply.get("type") != "ack" or not reply.get("complete"):
                    raise PublishError(f"Receiver did not complete the stream: {reply}")
                return published

        except ConnectionClosed:
            # Captions already acknowledged are durable, so reconnecting costs
            # at most the window that was in flight.
            attempts += 1
            if attempts > retries:
                raise
            await asyncio.sleep(min(2 ** (attempts - 1), 10))
