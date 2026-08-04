"""Configuration for the VLM question-answering app.

Every value can be overridden by an environment variable so the same code
runs unchanged on the IQ-9075 and on a developer box pointing at a remote
GenieX server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# GenieX's local server default (see `geniex serve -h`).
DEFAULT_BASE_URL = "http://127.0.0.1:18181/v1"

# Qwen3-VL-4B-Instruct w4a16 is the AI Hub bundle precompiled for
# qualcomm-qcs9075, so it runs pinned to the Hexagon NPU. This is the id
# GenieX reports in `geniex list` once cached, which is what the server
# expects in the "model" field.
# The server reports (and matches on) the id *with* its precision suffix.
DEFAULT_MODEL = "qualcomm/Qwen3-VL-4B-Instruct:W4A16"

# The qairt bundle bakes its context length in at compile time -- it cannot be
# raised at runtime (--nctx has no effect). Every image costs tokens, so the
# frame budget has a hard ceiling.
DEFAULT_CONTEXT_TOKENS = 4096

# Measured on Qwen3-VL-4B w4a16 at max_image_edge=896: prompt tokens grew
# 1695 -> 2231 -> 3304 for 6 -> 8 -> 12 frames, i.e. ~265 tokens per image
# plus a small fixed overhead. Rounded up to stay on the safe side.
TOKENS_PER_IMAGE = 275
TOKEN_OVERHEAD = 200


@dataclass(frozen=True)
class Settings:
    base_url: str = os.getenv("VLMQA_BASE_URL", DEFAULT_BASE_URL)
    model: str = os.getenv("VLMQA_MODEL", DEFAULT_MODEL)
    api_key: str = os.getenv("VLMQA_API_KEY", "not-needed")

    # Generation
    max_tokens: int = int(os.getenv("VLMQA_MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("VLMQA_TEMPERATURE", "0.2"))

    # Media handling
    max_image_edge: int = int(os.getenv("VLMQA_MAX_IMAGE_EDGE", "896"))
    default_frames: int = int(os.getenv("VLMQA_DEFAULT_FRAMES", "6"))
    context_tokens: int = int(os.getenv("VLMQA_CONTEXT_TOKENS", str(DEFAULT_CONTEXT_TOKENS)))

    def max_frames(self) -> int:
        """How many separate images fit in context, leaving room to answer."""
        budget = self.context_tokens - self.max_tokens - TOKEN_OVERHEAD
        return max(1, budget // TOKENS_PER_IMAGE)

    # Networking. First token on a cold model can take a while: the bundle is
    # memory-mapped and the vision encoder runs before any text is emitted.
    connect_timeout: float = float(os.getenv("VLMQA_CONNECT_TIMEOUT", "10"))
    read_timeout: float = float(os.getenv("VLMQA_READ_TIMEOUT", "600"))

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"


settings = Settings()
