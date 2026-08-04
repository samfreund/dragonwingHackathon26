"""Ask questions about an image or a pre-recorded video.

A VLM only ever sees still images, so a video is reduced to a small set of
sampled frames before it reaches the model. Two strategies:

  frames  (default) -- send each frame as its own image, at full resolution.
                       Measurably better at motion: on a 10s clip this
                       correctly reported a bus "moving from left to right"
                       where the contact sheet called the scene static.
                       Costs ~275 tokens per frame.
  sheet             -- tile the frames into one contact-sheet image. Roughly
                       4x cheaper in tokens and the model still perceives
                       ordering, but each frame shrinks, so fine motion and
                       small details get lost. Use it for long videos where
                       many frames matter more than per-frame detail.

The qairt bundle's context is fixed at compile time, so `frames` has a hard
ceiling (~13 at the default settings). Past that we fall back to `sheet`
rather than overflowing the context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .client import Answer, VLMClient, image_part, text_part
from .config import (
    TOKEN_OVERHEAD,
    TOKENS_PER_IMAGE,
    Settings,
    settings as default_settings,
)
from .media import (
    Frame,
    MediaError,
    classify,
    make_contact_sheet,
    prepare_image,
    probe_video,
    sample_frames,
    workspace,
)

SYSTEM_PROMPT = (
    "You are a careful visual analyst. Answer only from what is visible in the "
    "images provided. If the answer is not visible, say so plainly rather than "
    "guessing. Be specific and concise."
)


@dataclass
class Prepared:
    """Media reduced to the images that will be sent to the model."""

    kind: str                      # "image" | "video"
    images: list[Path]
    frames: list[Frame] = field(default_factory=list)
    strategy: str = "single"
    description: str = ""
    note: str = ""                 # set when the request was adjusted

    @property
    def n_images(self) -> int:
        return len(self.images)


def prepare(
    media: Path,
    scratch: Path,
    *,
    frames: int,
    strategy: str,
    max_edge: int,
    max_frames: int | None = None,
) -> Prepared:
    """Turn a media file into the list of images to send."""
    kind = classify(media)
    note = ""

    # Sending each frame separately is the accurate path, but it is bounded by
    # the bundle's fixed context. Tiling degrades gracefully; overflowing does
    # not, so prefer the sheet over a truncated prompt.
    if kind == "video" and strategy == "frames" and max_frames and frames > max_frames:
        note = (
            f"{frames} separate frames exceed what fits in the model's fixed "
            f"context (limit {max_frames}); tiled them into a contact sheet "
            f"instead. Pass --frames {max_frames} or fewer to keep full "
            "per-frame detail."
        )
        strategy = "sheet"

    if kind == "image":
        return Prepared(
            kind="image",
            images=[prepare_image(media, max_edge, scratch)],
            description=f"image {media.name}",
        )

    info = probe_video(media)
    sampled = sample_frames(media, frames, max_edge, scratch)
    stamps = ", ".join(f.label for f in sampled)

    if strategy == "frames":
        return Prepared(
            kind="video",
            images=[f.path for f in sampled],
            frames=sampled,
            strategy="frames",
            description=f"video {media.name} ({info}) -- {len(sampled)} frames at {stamps}",
            note=note,
        )

    sheet = make_contact_sheet(sampled, max_edge, scratch)
    return Prepared(
        kind="video",
        images=[sheet],
        frames=sampled,
        note=note,
        strategy="sheet",
        description=f"video {media.name} ({info}) -- {len(sampled)} frames at {stamps}",
    )


def build_content(prepared: Prepared, question: str) -> list[dict]:
    """Assemble the multimodal user turn.

    Images come first: Qwen3-VL attends better when the question follows the
    visual context, and it keeps the instruction closest to the generation.
    """
    parts: list[dict] = []

    if prepared.kind == "video":
        stamps = [f.label for f in prepared.frames]
        if prepared.strategy == "sheet":
            preamble = (
                f"The image below is a contact sheet of {len(stamps)} frames sampled "
                f"evenly from a video, in chronological order (left to right, top to "
                f"bottom). The frames are at timestamps: {', '.join(stamps)}. "
                "Treat them as a time sequence, not as unrelated pictures."
            )
        else:
            preamble = (
                f"The {len(stamps)} images below are frames sampled evenly from a "
                f"video, in chronological order at timestamps: {', '.join(stamps)}. "
                "Treat them as a time sequence, not as unrelated pictures."
            )
        parts.append(text_part(preamble))

    parts.extend(image_part(p) for p in prepared.images)
    parts.append(text_part(question))
    return parts


def ask_about(
    media: Path,
    question: str,
    *,
    frames: int | None = None,
    strategy: str = "frames",
    settings: Settings | None = None,
    client: VLMClient | None = None,
    stream: bool = False,
    on_token=None,
) -> tuple[Answer, Prepared]:
    """One-shot: prepare the media, ask the question, return the answer."""
    cfg = settings or default_settings
    vlm = client or VLMClient(cfg)
    n_frames = frames or cfg.default_frames

    if strategy not in ("sheet", "frames"):
        raise MediaError(f"Unknown strategy {strategy!r}; expected 'sheet' or 'frames'")

    with workspace() as tmp:
        prepared = prepare(
            media,
            Path(tmp),
            frames=n_frames,
            strategy=strategy,
            max_edge=cfg.max_image_edge,
            max_frames=cfg.max_frames(),
        )
        content = build_content(prepared, question)
        answer = vlm.ask(
            content, system=SYSTEM_PROMPT, stream=stream, on_token=on_token
        )
    return answer, prepared


class Session:
    """Ask several questions about one piece of media.

    Frames are extracted once and cached on disk for the life of the session,
    so follow-ups skip the ffmpeg work. The requests themselves are stateless:
    GenieX will not accept image parts inside conversation history, so every
    turn re-sends the images and carries the prior Q&A as plain text.
    """

    def __init__(
        self,
        media: Path,
        *,
        frames: int | None = None,
        strategy: str = "frames",
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.client = VLMClient(self.settings)
        self.media = media
        self.strategy = strategy
        self.n_frames = frames or self.settings.default_frames
        self._scratch = workspace()
        self._history: list[dict] = []

        self.prepared = prepare(
            media,
            Path(self._scratch.name),
            frames=self.n_frames,
            strategy=strategy,
            max_edge=self.settings.max_image_edge,
            max_frames=self.settings.max_frames(),
        )

    def ask(self, question: str, *, stream: bool = False, on_token=None) -> Answer:
        # The images ride on the *current* turn, every turn. GenieX rejects a
        # request whose history contains image parts -- replaying them fails
        # with `SDKError(Multimodal generation failed)` -- so prior turns are
        # kept as plain text and the visual context is re-supplied each time.
        content = build_content(self.prepared, question)

        answer = self.client.ask(
            content,
            system=SYSTEM_PROMPT,
            history=self._history,
            stream=stream,
            on_token=on_token,
        )

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer.text})
        self._trim_history()
        return answer

    def _trim_history(self) -> None:
        """Drop the oldest turns so images + history stay inside the context.

        The images are re-sent every turn and dominate the prompt, so only a
        modest text budget is left. Turns are dropped in pairs to keep the
        user/assistant alternation intact.
        """
        image_cost = self.prepared.n_images * TOKENS_PER_IMAGE
        budget = (
            self.settings.context_tokens
            - self.settings.max_tokens
            - image_cost
            - TOKEN_OVERHEAD
        )
        # ~4 characters per token is a good enough estimate for plain text.
        while self._history and sum(len(m["content"]) for m in self._history) // 4 > max(0, budget):
            del self._history[:2]

    def close(self) -> None:
        self._scratch.cleanup()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
