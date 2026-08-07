from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

from .phone_broker import PhoneQueryBroker
from .storage import TextStore


def _load_hybrid_qa(repo_root: Path):
    source = repo_root / "ask.py"
    spec = importlib.util.spec_from_file_location("dragonassist_ask", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.HybridQA()


def run_worker(
    storage_root: Path,
    *,
    once: bool = False,
    mock_answer: str | None = None,
    poll_seconds: float = 0.25,
) -> None:
    broker = PhoneQueryBroker(storage_root / "phone_queries.sqlite3")
    store = TextStore(storage_root)
    engine = None
    repo_root = Path(__file__).resolve().parent.parent

    while True:
        query = broker.claim_next()
        if query is None:
            if once:
                return
            time.sleep(poll_seconds)
            continue
        try:
            context_path = store.context_path(query.video_id)
            query_path = storage_root / query.video_id / "query.txt"
            if not query_path.exists():
                raise FileNotFoundError(f"Phone question not found: {query_path}")
            if not context_path.exists():
                raise FileNotFoundError(f"IQ9 context not found: {context_path}")
            question = query_path.read_text(encoding="utf-8").strip()
            context = context_path.read_text(encoding="utf-8").strip()
            if not question:
                raise ValueError(f"Phone question is empty: {query_path}")
            if not context:
                raise ValueError(f"IQ9 context is empty: {context_path}")
            if mock_answer is not None:
                answer = mock_answer
                route = "mock"
            else:
                if engine is None:
                    engine = _load_hybrid_qa(repo_root)
                reply = engine.answer(
                    question,
                    context,
                    context_source=f"{query_path} + {context_path}",
                )
                answer = reply.answer
                route = reply.backend
            broker.complete(query.request_id, answer, route=route)
        except Exception as exc:
            broker.fail(query.request_id, f"{type(exc).__name__}: {exc}")
        if once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Process durable phone queries")
    parser.add_argument("--storage-root", type=Path, default=Path("received"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--mock-answer", default=None)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()
    run_worker(
        args.storage_root,
        once=args.once,
        mock_answer=args.mock_answer,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
