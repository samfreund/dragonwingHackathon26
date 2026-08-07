"""Turn a file on disk into something a VLM can look at.

Images go through a downscale pass; videos are sampled into a handful of
still frames. Everything is done with ffmpeg/ffprobe so the only Python
dependency is `requests` -- no OpenCV or PyAV wheels to fight with on ARM64.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}


class MediaError(RuntimeError):
    """Raised when media can't be read or decoded."""


def _require_ffmpeg() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise MediaError(
            f"{' and '.join(missing)} not found on PATH. Install with: sudo apt-get install -y ffmpeg"
        )


# The Qualcomm OpenCL ICD (qcom-adreno1) has no version symbols, so every
# ffmpeg/ffprobe invocation on this board emits a few loader warnings on
# stderr. They are harmless and would otherwise bury the real error.
_NOISE = ("no version information available",)


def _clean_stderr(text: str) -> list[str]:
    return [
        line for line in text.strip().splitlines()
        if line.strip() and not any(n in line for n in _NOISE)
    ]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        lines = _clean_stderr(proc.stderr or "") or _clean_stderr(proc.stdout or "")
        detail = " | ".join(lines[-3:]) if lines else f"exit code {proc.returncode}"
        raise MediaError(f"{cmd[0]} failed: {detail}")
    return proc


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    width: int
    height: int
    fps: float

    def __str__(self) -> str:
        return f"{self.width}x{self.height}, {self.duration_s:.1f}s @ {self.fps:.2f}fps"


@dataclass(frozen=True)
class Frame:
    """One sampled still, with the timestamp it was taken from."""

    path: Path
    timestamp_s: float

    @property
    def label(self) -> str:
        m, s = divmod(self.timestamp_s, 60)
        return f"{int(m):02d}:{s:05.2f}"


def classify(path: Path) -> str:
    """Return 'image' or 'video' for a media path."""
    if not path.exists():
        raise MediaError(f"No such file: {path}")
    if not path.is_file():
        raise MediaError(f"Not a file: {path}")

    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"

    # Unknown extension -- ask ffprobe what streams are actually in there.
    # A single-frame video stream with no duration is a still image.
    try:
        info = probe_video(path)
    except MediaError as exc:
        raise MediaError(f"Unrecognised media type for {path.name} ({exc})") from exc
    return "image" if info.duration_s <= 0 else "video"


def probe_video(path: Path) -> VideoInfo:
    """Read duration/dimensions/fps from the first video stream."""
    _require_ffmpeg()
    proc = _run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ])
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise MediaError(f"No video stream found in {path.name}")
    stream = streams[0]

    # avg_frame_rate arrives as a rational string like "30000/1001".
    fps = 0.0
    raw_fps = stream.get("avg_frame_rate") or "0/0"
    try:
        num, _, den = raw_fps.partition("/")
        fps = float(num) / float(den) if float(den or 0) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    try:
        duration = float(data.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    return VideoInfo(
        duration_s=duration,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=fps,
    )


def _scale_filter(max_edge: int) -> str:
    """Downscale so the longest edge is `max_edge`, never upscaling.

    Dimensions are forced even; some encoders reject odd sizes.
    """
    return (
        f"scale=w='if(gt(iw,ih),min({max_edge},iw),-2)'"
        f":h='if(gt(iw,ih),-2,min({max_edge},ih))'"
    )


def prepare_image(path: Path, max_edge: int, out_dir: Path) -> Path:
    """Downscale a still image and normalise it to JPEG."""
    _require_ffmpeg()
    out = out_dir / "image.jpg"
    _run([
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-i", str(path),
        "-vf", _scale_filter(max_edge),
        "-frames:v", "1", "-q:v", "3",
        str(out),
    ])
    if not out.exists():
        raise MediaError(f"Could not decode image: {path.name}")
    return out


def sample_frames(
    path: Path,
    count: int,
    max_edge: int,
    out_dir: Path,
    *,
    window: tuple[float, float] | None = None,
) -> list[Frame]:
    """Extract `count` stills spread evenly across the video.

    Each frame is taken from the midpoint of its slice rather than the
    boundary, so the first frame isn't a black lead-in and the last isn't
    past the final decodable packet.

    `window` restricts sampling to a `(start_s, end_s)` span, which is how the
    publisher captions a long video piece by piece. Timestamps on the returned
    frames stay absolute, so a caption can be labelled with its real position
    in the video rather than an offset into the window.
    """
    if count < 1:
        raise MediaError("Frame count must be at least 1")

    info = probe_video(path)
    if info.duration_s <= 0:
        raise MediaError(f"Could not determine duration of {path.name}")

    start_s, end_s = window or (0.0, info.duration_s)
    start_s = max(0.0, start_s)
    end_s = min(info.duration_s, end_s)
    if end_s <= start_s:
        raise MediaError(
            f"Empty sampling window [{start_s:.3f}, {end_s:.3f}) in a "
            f"{info.duration_s:.1f}s video"
        )
    span = end_s - start_s

    frames: list[Frame] = []
    for i in range(count):
        ts = start_s + span * (i + 0.5) / count
        out = out_dir / f"frame_{i:03d}.jpg"
        # -ss before -i seeks by keyframe (fast); accurate enough for sampling.
        _run([
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-ss", f"{ts:.3f}", "-i", str(path),
            "-vf", _scale_filter(max_edge),
            "-frames:v", "1", "-q:v", "3",
            str(out),
        ])
        if out.exists() and out.stat().st_size > 0:
            frames.append(Frame(path=out, timestamp_s=ts))

    if not frames:
        raise MediaError(f"Could not extract any frames from {path.name}")
    return frames


def make_contact_sheet(frames: list[Frame], max_edge: int, out_dir: Path) -> Path:
    """Tile frames into a single grid image, read left-to-right, top-to-bottom.

    One image costs far fewer tokens than N images, which matters a lot
    against a 4096-token context.
    """
    _require_ffmpeg()
    if not frames:
        raise MediaError("No frames to tile")
    if len(frames) == 1:
        return frames[0].path

    cols = math_ceil_sqrt(len(frames))
    rows = -(-len(frames) // cols)  # ceil division

    # Match the cell shape to the source aspect ratio. Square cells would
    # letterbox 16:9 frames into ~40% black, wasting both resolution and the
    # tokens the encoder spends on that padding.
    src_w, src_h = image_size(frames[0].path)
    aspect = (src_w / src_h) if src_h else 16 / 9

    cell_w = max(160, max_edge // cols)
    cell_h = max(90, _even(cell_w / aspect))
    cell_w = _even(cell_w)

    out = out_dir / "contact_sheet.jpg"

    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    for frame in frames:
        cmd += ["-i", str(frame.path)]

    # Normalise every cell to identical dimensions -- xstack requires it.
    # Frames that don't match the first one's aspect get letterboxed here.
    filters = "".join(
        f"[{i}:v]scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease,"
        f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:color=black[c{i}];"
        for i in range(len(frames))
    )
    filters += "".join(f"[c{i}]" for i in range(len(frames)))
    if len(frames) == 2 and rows == 1:
        filters += "hstack=inputs=2"
    else:
        filters += (
            f"xstack=inputs={len(frames)}"
            f":layout={_xstack_layout(len(frames), cols, cell_w, cell_h)}"
            ":fill=black"
        )

    cmd += ["-filter_complex", filters, "-frames:v", "1", "-q:v", "3", str(out)]
    _run(cmd)
    if not out.exists():
        raise MediaError("Failed to build contact sheet")
    return out


def _even(value: float) -> int:
    """Nearest even integer -- odd dimensions upset some encoders."""
    n = int(round(value))
    return n if n % 2 == 0 else n + 1


def image_size(path: Path) -> tuple[int, int]:
    """Pixel dimensions of a still image."""
    proc = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path),
    ])
    try:
        w, _, h = proc.stdout.strip().partition("x")
        return int(w), int(h)
    except ValueError as exc:
        raise MediaError(f"Could not read dimensions of {path.name}") from exc


def _xstack_layout(n: int, cols: int, cell_w: int, cell_h: int) -> str:
    """Build an xstack layout string placing n cells in a `cols`-wide grid."""
    parts = []
    for i in range(n):
        col, row = i % cols, i // cols
        parts.append(f"{col * cell_w}_{row * cell_h}")
    return "|".join(parts)


def math_ceil_sqrt(n: int) -> int:
    """Smallest c with c*c >= n -- gives a near-square grid."""
    c = 1
    while c * c < n:
        c += 1
    return c


def to_data_url(path: Path) -> str:
    """Base64 data URL. Avoids any ambiguity about which filesystem the
    server resolves a path against (matters under Docker)."""
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def workspace() -> tempfile.TemporaryDirectory:
    """Scratch dir for derived frames; cleaned up by the caller's context."""
    return tempfile.TemporaryDirectory(prefix="vlmqa-")
