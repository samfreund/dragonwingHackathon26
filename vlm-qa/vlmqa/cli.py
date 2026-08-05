"""Command-line entry point.

    vlmqa ask   --media cat.jpg   --question "What breed is this?"
    vlmqa ask   --media clip.mp4  --question "What does the person do?" --frames 8
    vlmqa chat  --media clip.mp4
    vlmqa status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import VLMClient, VLMError
from .config import settings
from .media import MediaError
from .qa import Session, ask_about


def _add_media_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--media", "-m", required=True, type=Path,
                   help="Path to an image or a pre-recorded video file.")
    p.add_argument("--frames", "-n", type=int, default=None,
                   help=f"Frames to sample from a video (default {settings.default_frames}). "
                        "Ignored for images.")
    p.add_argument("--strategy", "-s", choices=("frames", "sheet"), default="frames",
                   help="How video frames reach the model: 'frames' sends each separately "
                        "(default; best at motion and detail); 'sheet' tiles them into one "
                        "image (~4x cheaper in tokens, but fine detail is lost). "
                        f"'frames' falls back to 'sheet' above {settings.max_frames()} frames.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlmqa",
        description="Ask a vision-language model questions about an image or a pre-recorded video, "
                    "running on the Hexagon NPU via GenieX.",
    )
    parser.add_argument("--model", default=None, help=f"Override model id (default {settings.model}).")
    parser.add_argument("--base-url", default=None, help=f"GenieX server (default {settings.base_url}).")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask one question and print the answer.")
    _add_media_args(ask)
    ask.add_argument("--question", "-q", required=True, help="The question to ask.")
    ask.add_argument("--stream", action="store_true", help="Stream tokens as they are generated.")
    ask.add_argument("--json", action="store_true", help="Emit a JSON object instead of prose.")

    chat = sub.add_parser("chat", help="Interactive follow-up questions about one file.")
    _add_media_args(chat)

    serve = sub.add_parser("serve", help="Run the WebSocket server (upload media, ask, stream back).")
    serve.add_argument("--host", default=None,
                       help=f"Interface to bind (default {settings.ws_host}). "
                            "Use 0.0.0.0 to accept connections from the network -- "
                            "set VLMQA_WS_TOKEN if you do.")
    serve.add_argument("--port", "-p", type=int, default=None,
                       help=f"Port to listen on (default {settings.ws_port}).")

    sub.add_parser("status", help="Check the GenieX server and loaded models.")
    return parser


def _apply_overrides(args) -> None:
    """argparse overrides beat environment defaults."""
    import dataclasses

    from . import config

    changes = {}
    if args.model:
        changes["model"] = args.model
    if args.base_url:
        changes["base_url"] = args.base_url
    if changes:
        config.settings = dataclasses.replace(config.settings, **changes)


def cmd_status() -> int:
    from .config import settings as cfg

    client = VLMClient(cfg)
    up = client.is_up()
    print(f"server   : {cfg.base_url}  [{'up' if up else 'DOWN'}]")
    if not up:
        print("\nThe GenieX server is not responding. Start it with:\n  geniex serve")
        return 1

    models = client.loaded_models()
    print(f"models   : {', '.join(models) if models else '(none reported)'}")
    print(f"configured: {cfg.model}")
    if models and cfg.model not in models:
        print(f"\nWarning: {cfg.model} is not in the server's model list. Pull it with:\n"
              f"  geniex pull {cfg.model}:w4a16 --model-type vlm")
    return 0


def cmd_ask(args) -> int:
    from .config import settings as cfg

    on_token = None
    if args.stream and not args.json:
        def on_token(piece: str) -> None:
            sys.stdout.write(piece)
            sys.stdout.flush()

    answer, prepared = ask_about(
        args.media,
        args.question,
        frames=args.frames,
        strategy=args.strategy,
        settings=cfg,
        stream=args.stream and not args.json,
        on_token=on_token,
    )

    if args.json:
        print(json.dumps({
            "media": str(args.media),
            "kind": prepared.kind,
            "strategy": prepared.strategy,
            "images_sent": prepared.n_images,
            "frame_timestamps": [f.label for f in prepared.frames],
            "question": args.question,
            "answer": answer.text,
            "latency_s": round(answer.latency_s, 2),
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "note": prepared.note or None,
        }, indent=2))
        return 0

    if args.stream:
        print()
    else:
        print(answer.text)

    if prepared.note:
        print(f"\nnote: {prepared.note}", file=sys.stderr)
    print(f"\n--- {prepared.description}", file=sys.stderr)
    print(f"--- {prepared.n_images} image(s) sent, {answer.latency_s:.1f}s", file=sys.stderr)
    return 0


def cmd_chat(args) -> int:
    from .config import settings as cfg

    print(f"Loading {args.media.name} ...", file=sys.stderr)
    with Session(args.media, frames=args.frames, strategy=args.strategy, settings=cfg) as session:
        print(f"Ready: {session.prepared.description}", file=sys.stderr)
        if session.prepared.note:
            print(f"note: {session.prepared.note}", file=sys.stderr)
        print("Ask a question, or Ctrl-D / 'exit' to quit.\n", file=sys.stderr)
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                return 0
            try:
                answer = session.ask(
                    question,
                    stream=True,
                    on_token=lambda p: (sys.stdout.write(p), sys.stdout.flush()),
                )
                print(f"\n  [{answer.latency_s:.1f}s]\n")
            except VLMError as exc:
                print(f"\nerror: {exc}\n", file=sys.stderr)


def cmd_serve(args) -> int:
    import asyncio

    from .config import settings as cfg

    try:
        from .ws_server import serve
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        asyncio.run(serve(cfg, host=args.host, port=args.port))
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        print(f"error: cannot listen on {args.host or cfg.ws_host}:"
              f"{args.port if args.port is not None else cfg.ws_port} -- {exc}",
              file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_overrides(args)

    try:
        if args.command == "status":
            return cmd_status()
        if args.command == "ask":
            return cmd_ask(args)
        if args.command == "chat":
            return cmd_chat(args)
        if args.command == "serve":
            return cmd_serve(args)
    except (MediaError, VLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
