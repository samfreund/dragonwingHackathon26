"""Answer a question about a file, on the smallest backend that can answer it.

This is the glue between three pieces that already exist in this repo, each
with its own CLI and README:

  query_router/router.py   decides *on-device* whether the question can be
      answered by copying a span out of the file, or needs a real generative
      model. One ONNX prefill pass of Qwen2.5-0.5B on the CPU, ~0.5 s.
  npu_qa/npu_qa.py         extractive QA with distilbert-base-cased-distilled-
      squad on the Hexagon NPU. Milliseconds, offline, nothing billed -- but it
      can only return a phrase copied verbatim out of the file.
  crewai_imagine_test.py   Llama-3.1-8B on Qualcomm's Imagine cloud (AI 100).
      Reasons, summarises, computes, rewrites, and knows things the file does
      not say. Costs a network round trip and tokens.

      question + context file
                |
                v
        +-----------------+  p(extractive) >= 0.60   +------------------------+
        |  Qwen2.5-0.5B   | -----------------------> | DistilBERT on Hexagon  |
        |  router (CPU)   |                          +------------------------+
        +-----------------+  otherwise               +------------------------+
                +------------------------------->    | Llama-3.1-8B on AI 100 |
                                                     +------------------------+

Nothing here re-implements the components: router.py and npu_qa.py are loaded
as modules from their own directories and used through their public functions,
so their standalone CLIs keep working unchanged and any tuning done there (the
routing prompt, the 0.60 threshold, the QNN device selection) applies here too.

What this file adds on top of the three
---------------------------------------
* One entry point that takes the two things a user actually has -- a question
  and a file -- and returns an answer plus the record of how it was reached.
* Lazy construction. The DistilBERT session (~2 s on the NPU) is never built on
  a run that goes to the cloud, and the cloud client is never created on a run
  that stays local. Only the router is always paid for.
* Two safety nets the components cannot provide alone, because each only knows
  about itself:
    - if the Hexagon NPU is unavailable, the reader falls back to the CPU
      rather than aborting a run the router already committed to (--strict-npu
      to abort instead);
    - if the reader returns an empty span -- its way of saying "the answer is
      not in here" -- the question is escalated to the cloud (--no-escalate to
      keep the empty answer).
  There is deliberately no fallback in the other direction: answering a
  synthesis question with a copied span produces something confidently wrong,
  which is the failure mode the router exists to prevent. If the cloud is
  unreachable, this exits with an error instead.

Setup
-----
One interpreter has to satisfy all three components: onnxruntime(-qnn),
transformers and numpy for the two local models, plus the Imagine SDK for the
cloud. npu_qa/.venv is the one built from a native arm64 Python (it has the
NPU wheel), so add the two missing pieces to it:

    npu_qa\\.venv\\Scripts\\python.exe -m pip install jinja2
    npu_qa\\.venv\\Scripts\\python.exe -m pip install <path>\\imagine_sdk-0.4.2-py3-none-any.whl

imagine-sdk is not on PyPI -- it ships as a wheel from the Imagine console.
Missing pieces degrade rather than break: without onnxruntime-qnn the reader
runs on the CPU, without the Imagine SDK everything except the cloud path
still works.

Usage
-----
    python ask.py "How many Nobel Prizes did Marie Curie win?" notes.txt
    python ask.py "Summarise the incident" incident.log --json
    python ask.py "What port does it use?" -            # context from stdin
    python ask.py "Who signed it?" deal.txt --force cloud
    python ask.py "What is the model number?" spec.txt --quiet   # answer only

Credentials come from IMAGINE_API_KEY / IMAGINE_API_ENDPOINT when set, and
otherwise from the hackathon defaults already committed in
crewai_imagine_test.py.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTER_PATH = HERE / "query_router" / "router.py"
READER_PATH = HERE / "npu_qa" / "npu_qa.py"

# Cloud backend. Defaults match crewai_imagine_test.py (the Indonesia endpoint
# it ends up using); the environment wins when set.
DEFAULT_ENDPOINT = "https://aisuite-indonesia.cirrascale.com/apis/v2"
DEFAULT_API_KEY = "745f6cf2-f53b-4ea8-ac62-1e26d7a1646b"
DEFAULT_CLOUD_MODEL = "Llama-3.1-8B"
# DEFAULT_CLOUD_MODEL = "Llama-3.3-70B"

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2  # question answering, not creative writing

# The cloud model has room for far more than this, but a runaway file should
# not turn one question into a six-figure token bill. Head+tail, so the shape
# of a long document survives the cut.
DEFAULT_CLOUD_CONTEXT_CHARS = 48_000

# The cloud gets the questions the reader cannot do: synthesis, arithmetic,
# rewriting, and questions whose answer simply is not in the file. So the
# prompt cannot say "answer only from the context" -- it has to allow going
# beyond it while making clear when that is happening.
CLOUD_SYSTEM_PROMPT = (
    "You answer questions about a reference text supplied by the user. "
    "Ground your answer in that text wherever it covers the question, and quote or cite "
    "the relevant part when it helps. If the text does not contain the answer, say so "
    "plainly and then answer from general knowledge, marking clearly which part is which. "
    "Be direct and concise; do not pad the answer with restatements of the question."
)


# -- loading the components --------------------------------------------------


def _load_module(name: str, path: Path):
    """Import router.py / npu_qa.py from its own directory, by file path.

    They are standalone scripts in sibling folders with no package __init__ and
    no shared parent module, so there is nothing to `import` conventionally.
    Loading by location keeps this file the only new thing on disk, and keeps
    each component's own CLI working exactly as it did before.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"component not found: {path}\n"
            f"ask.py expects to sit next to query_router/ and npu_qa/."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _quiet_stdout():
    """Divert everything written to stdout -- Python's and native code's -- to stderr.

    QNN's graph compiler prints its stage timings and a progress bar directly
    to file descriptor 1 from native code while a session is being built, where
    contextlib.redirect_stdout cannot see it. That output would corrupt --json,
    so the descriptor itself is pointed at stderr for the duration of the build.
    """
    try:
        stderr_fd = sys.stderr.fileno()
        saved = os.dup(1)
    except (AttributeError, OSError, ValueError):
        # stdout/stderr are not real file descriptors (embedded, captured in a
        # test harness); the Python-level redirect is the best available.
        with contextlib.redirect_stdout(sys.stderr):
            yield
        return

    sys.stdout.flush()
    try:
        os.dup2(stderr_fd, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _head_tail(text: str, limit: int) -> tuple[str, bool]:
    """Trim to `limit` chars keeping both ends. Returns (text, was_trimmed)."""
    if limit <= 0 or len(text) <= limit:
        return text, False
    head = int(limit * 0.75)
    tail = limit - head
    return f"{text[:head]}\n\n[... {len(text) - limit} characters omitted ...]\n\n{text[-tail:]}", True


# -- result ------------------------------------------------------------------


@dataclass
class Reply:
    """An answer plus the full record of how it was produced."""

    question: str
    answer: str
    backend: str  # "npu" | "cpu" | "cloud"
    backend_detail: str  # model + execution provider / endpoint
    routed_to: str  # what the router picked, before any escalation
    routing: dict  # RoutingDecision as a dict, verbatim from router.py
    context_source: str = ""
    context_chars: int = 0
    context_words: int = 0
    escalated: bool = False
    notes: list[str] = field(default_factory=list)
    span: dict | None = None  # extractive path: which tokens were copied
    usage: dict | None = None  # cloud path: token accounting
    setup_ms: float = 0.0  # one-time model loading, charged to whoever pays it first
    route_ms: float = 0.0
    answer_ms: float = 0.0
    total_ms: float = 0.0

    def render(self) -> str:
        r = self.routing
        pe = r.get("p_extractive")
        detail = f"{r.get('decided_by')}"
        if pe is not None:
            detail += f", p_extractive={pe:.3f}"
        detail += f", {self.route_ms:.0f} ms"

        lines = [
            f"question   {self.question}",
            f"context    {self.context_source} -- {self.context_words} words, "
            f"{self.context_chars} chars",
            "",
            f"route      {self.routed_to}  ({detail})",
            f"reason     {r.get('reason', '')}",
            f"backend    {self.backend_detail}",
        ]
        for note in self.notes:
            lines.append(f"note       {note}")
        body = textwrap.indent(self.answer.strip() or "(empty)", "    ")
        setup = f"setup {self.setup_ms:.0f} ms + " if self.setup_ms >= 1 else ""
        lines += [
            "",
            "answer",
            body,
            "",
            f"timing     {setup}route {self.route_ms:.0f} ms + answer {self.answer_ms:.0f} ms "
            f"= {self.total_ms:.0f} ms",
        ]
        if self.usage:
            used = ", ".join(f"{k}={v}" for k, v in sorted(self.usage.items()) if v is not None)
            lines.append(f"tokens     {used}")
        return "\n".join(lines)


# -- the pipeline ------------------------------------------------------------


class HybridQA:
    """Router plus both backends, with every expensive part built on first use.

    Constructing this is free. The router session, the DistilBERT session and
    the cloud client are each created the first time a question actually needs
    them, so one process can serve many questions and pay for a backend only
    once and only if it is used. That also makes this class the natural thing
    to hold open in a server (see vlm-qa/) rather than shelling out per query.
    """

    def __init__(
        self,
        *,
        threshold: float | None = None,
        router_npu: bool = False,
        reader_npu: bool = True,
        strict_npu: bool = False,
        allow_long_context: bool = False,
        cloud_model: str = DEFAULT_CLOUD_MODEL,
        api_key: str | None = None,
        endpoint: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        cloud_context_chars: int = DEFAULT_CLOUD_CONTEXT_CHARS,
        escalate_empty: bool = True,
        verbose: bool = True,
    ) -> None:
        self.threshold = threshold
        self.router_npu = router_npu
        self.reader_npu = reader_npu
        self.strict_npu = strict_npu
        self.allow_long_context = allow_long_context
        self.cloud_model = cloud_model
        self.api_key = api_key or os.environ.get("IMAGINE_API_KEY") or DEFAULT_API_KEY
        self.endpoint = endpoint or os.environ.get("IMAGINE_API_ENDPOINT") or DEFAULT_ENDPOINT
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cloud_context_chars = cloud_context_chars
        self.escalate_empty = escalate_empty
        self.verbose = verbose

        self._router_mod = None
        self._router = None
        self._reader_mod = None
        self._reader = None  # (session, tokenizer)
        self._reader_note: str | None = None
        self._cloud = None
        self._setup_ms = 0.0  # model loading done during the current answer()

    # -- progress, on stderr so --json and --quiet stay pipeable --

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[ask] {message}", file=sys.stderr, flush=True)

    # -- lazily built components --

    def router_module(self):
        """router.py itself -- cheap, and holds the backend name constants."""
        if self._router_mod is None:
            self._router_mod = _load_module("query_router_router", ROUTER_PATH)
        return self._router_mod

    def router(self):
        if self._router is None:
            mod = self.router_module()
            self._log("loading the router (Qwen2.5-0.5B, CPU) ...")
            started = time.perf_counter()
            kwargs = {"use_npu": self.router_npu, "allow_long_context": self.allow_long_context}
            if self.threshold is not None:
                kwargs["threshold"] = self.threshold
            with _quiet_stdout():
                self._router = mod.QueryRouter(**kwargs)
            elapsed = (time.perf_counter() - started) * 1000
            self._setup_ms += elapsed
            self._log(f"router ready on {self._router.provider} ({elapsed:.0f} ms)")
        return self._router

    def reader(self, notes: list[str]):
        """(session, tokenizer) for DistilBERT, on the NPU where possible."""
        if self._reader is None:
            mod = self._reader_module()
            from transformers import AutoTokenizer

            where = "Hexagon NPU" if self.reader_npu else "CPU"
            self._log(f"loading the extractive reader (DistilBERT, {where}) ...")
            started = time.perf_counter()
            with _quiet_stdout():  # ensure_model() and the QNN compiler both print
                model_path = mod.ensure_model(Path(mod.MODEL_PATH))
                tokenizer = AutoTokenizer.from_pretrained(mod.TOKENIZER_REPO)
                try:
                    session = mod.build_session(model_path, use_npu=self.reader_npu)
                except Exception as exc:
                    if not self.reader_npu or self.strict_npu:
                        raise
                    # The router already decided this question is extractive.
                    # Answering it on the CPU is worth far more than aborting.
                    self._reader_note = (
                        f"Hexagon NPU unavailable ({type(exc).__name__}: {exc}); "
                        f"the extractive reader ran on the CPU instead"
                    )
                    session = mod.build_session(model_path, use_npu=False)
            self._reader = (session, tokenizer)
            elapsed = (time.perf_counter() - started) * 1000
            self._setup_ms += elapsed
            self._log(f"reader ready on {session.get_providers()[0]} ({elapsed:.0f} ms)")
        if self._reader_note:
            notes.append(self._reader_note)
        return self._reader

    def _reader_module(self):
        if self._reader_mod is None:
            self._reader_mod = _load_module("npu_qa_npu_qa", READER_PATH)
        return self._reader_mod

    def cloud(self):
        if self._cloud is None:
            try:
                from imagine import ImagineClient
            except ImportError as exc:  # the SDK is not on PyPI -- be specific
                raise RuntimeError(
                    "the Imagine SDK is not installed in this interpreter, so the cloud "
                    "backend is unavailable. Install the wheel from the Imagine console:\n"
                    "    python -m pip install <path>\\imagine_sdk-0.4.2-py3-none-any.whl\n"
                    "or re-run with --force local to stay on the extractive reader."
                ) from exc
            self._log(f"connecting to {self.endpoint} ...")
            started = time.perf_counter()
            self._cloud = ImagineClient(api_key=self.api_key, endpoint=self.endpoint)
            self._setup_ms += (time.perf_counter() - started) * 1000
        return self._cloud

    # -- the two backends --

    def _answer_local(self, question: str, context: str, notes: list[str]) -> tuple[str, dict, str]:
        mod = self._reader_module()
        session, tokenizer = self.reader(notes)
        answer, span = mod.answer_question(session, tokenizer, context, question)
        return answer, span, session.get_providers()[0]

    def _answer_cloud(self, question: str, context: str, notes: list[str]) -> tuple[str, dict | None]:
        from imagine import ChatMessage

        client = self.cloud()
        context = context.strip()
        if context:
            shown, trimmed = _head_tail(context, self.cloud_context_chars)
            if trimmed:
                notes.append(
                    f"context trimmed to {self.cloud_context_chars} characters "
                    f"(head+tail) before being sent to {self.cloud_model}"
                )
            user = f'Reference text:\n"""\n{shown}\n"""\n\nQuestion: {question}'
        else:
            notes.append("no reference text supplied -- answered from the model's own knowledge")
            user = question

        response = client.chat(
            messages=[
                ChatMessage(role="system", content=CLOUD_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user),
            ],
            model=self.cloud_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        usage = None
        if getattr(response, "usage", None) is not None:
            with contextlib.suppress(Exception):
                usage = dict(response.usage)

        # A truncated answer has to be flagged, or it reads as a complete one.
        # finish_reason alone is not enough: this endpoint returns "stop" even
        # when it stopped only because max_tokens ran out, so the token count
        # is the signal that actually fires.
        choice = response.choices[0] if response.choices else None
        cut_by_reason = choice is not None and choice.finish_reason == "length"
        cut_by_count = bool(usage) and (usage.get("completion_tokens") or 0) >= self.max_tokens
        if cut_by_reason or cut_by_count:
            notes.append(
                f"the answer hit the {self.max_tokens}-token limit and is cut off "
                f"(raise it with --max-tokens)"
            )
        return (response.first_content or "").strip(), usage

    # -- route, then answer --

    def answer(
        self,
        question: str,
        context: str,
        *,
        force: str = "auto",
        context_source: str = "(inline)",
    ) -> Reply:
        started = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("the question is empty")

        mod = self.router_module()
        notes: list[str] = []
        self._setup_ms = 0.0  # anything loaded from here on is this call's setup

        if force == "auto":
            decision = self.router().route(question, context)
        else:
            target = mod.DISTILBERT if force == "local" else mod.LLAMA
            decision = mod.RoutingDecision(
                model=target,
                confidence=1.0,
                reason=f"backend forced with --force {force}; the router was not consulted",
                decided_by="forced",
            )
        routing = asdict(decision)
        self._log(f"routed to {decision.model} ({decision.decided_by})")

        setup_before_answer = self._setup_ms
        answer_started = time.perf_counter()
        escalated = False
        span = usage = None

        if decision.model == mod.DISTILBERT:
            answer, span, provider = self._answer_local(question, context, notes)
            backend = "npu" if provider == "QNNExecutionProvider" else "cpu"
            detail = f"distilbert-base-cased-distilled-squad on {provider}"

            # An empty span is how the reader says "the answer is not in here":
            # the best-scoring span landed on [CLS]/[SEP]. That is not an answer,
            # so hand the question to the model that can actually produce one.
            if not answer.strip() and self.escalate_empty and force == "auto":
                notes.append(
                    "the extractive reader found no span to copy -- escalated to "
                    f"{self.cloud_model}"
                )
                self._log(f"empty span; escalating to {self.cloud_model} ...")
                answer, usage = self._answer_cloud(question, context, notes)
                escalated = True
                backend, span = "cloud", None
                detail = f"{self.cloud_model} on {self.endpoint}"
        else:
            answer, usage = self._answer_cloud(question, context, notes)
            backend = "cloud"
            detail = f"{self.cloud_model} on {self.endpoint}"

        # Model loading is reported on its own so the backend numbers stay
        # comparable across runs -- a warm process pays it once, not per query.
        now = time.perf_counter()
        answer_ms = (now - answer_started) * 1000 - (self._setup_ms - setup_before_answer)
        return Reply(
            question=question,
            answer=answer,
            backend=backend,
            backend_detail=detail,
            routed_to=decision.model,
            routing=routing,
            context_source=context_source,
            context_chars=len(context),
            context_words=len(context.split()),
            escalated=escalated,
            notes=notes,
            span=span,
            usage=usage,
            setup_ms=self._setup_ms,
            route_ms=decision.latency_ms,
            answer_ms=max(answer_ms, 0.0),
            total_ms=(now - started) * 1000,
        )


# -- CLI ---------------------------------------------------------------------


def read_context(path: str) -> tuple[str, str]:
    """Return (text, human-readable source) for a path, or '-' for stdin."""
    if path == "-":
        return sys.stdin.read(), "(stdin)"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"context file not found: {p}")
    try:
        return p.read_text(encoding="utf-8"), p.name
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="replace"), f"{p.name} (undecodable bytes replaced)"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("question", nargs="?", help="The question to answer")
    ap.add_argument("context_file", nargs="?", help="File holding the reference text ('-' for stdin)")
    ap.add_argument("-q", "--question", dest="question_opt", help="Alternative to the positional question")
    ap.add_argument("-f", "--context-file", dest="context_file_opt", help="Alternative to the positional file")
    ap.add_argument("--context", help="Inline reference text instead of a file (pass '' for none)")

    ap.add_argument(
        "--force",
        choices=("auto", "local", "cloud"),
        default="auto",
        help="Skip the router and pin the backend (default: %(default)s)",
    )
    ap.add_argument("--threshold", type=float, help="Override the router's p(extractive) threshold")
    ap.add_argument("--router-npu", action="store_true", help="Run the router on the NPU (slower; see router.py)")
    ap.add_argument("--reader-cpu", action="store_true", help="Run the extractive reader on the CPU")
    ap.add_argument("--strict-npu", action="store_true", help="Fail instead of falling back to the CPU")
    ap.add_argument(
        "--allow-long-context",
        action="store_true",
        help="Do not auto-escalate files longer than the reader's 384-token window",
    )
    ap.add_argument("--no-escalate", action="store_true", help="Keep an empty extractive answer instead of escalating")

    ap.add_argument("--cloud-model", default=DEFAULT_CLOUD_MODEL, help="Imagine model (default: %(default)s)")
    ap.add_argument("--endpoint", help="Imagine endpoint (default: $IMAGINE_API_ENDPOINT or the committed one)")
    ap.add_argument("--api-key", help="Imagine API key (default: $IMAGINE_API_KEY or the committed one)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Cloud answer length cap (default: %(default)s)")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Cloud sampling temperature (default: %(default)s)")
    ap.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_CLOUD_CONTEXT_CHARS,
        help="Trim the file to this many characters before sending it to the cloud (default: %(default)s)",
    )

    ap.add_argument("--json", action="store_true", help="Emit the whole record as JSON")
    ap.add_argument("--quiet", action="store_true", help="Print only the answer")
    args = ap.parse_args()

    question = args.question_opt or args.question
    if not question:
        ap.error("give a question, e.g.  ask.py \"Who signed it?\" deal.txt")

    context_file = args.context_file_opt or args.context_file
    if args.context is not None:
        context, source = args.context, "(inline)"
    elif context_file:
        try:
            context, source = read_context(context_file)
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        ap.error(
            "give a file holding the context, e.g.  ask.py \"Who signed it?\" deal.txt\n"
            "       (use --context TEXT for inline text, or --context '' for none)"
        )

    engine = HybridQA(
        threshold=args.threshold,
        router_npu=args.router_npu,
        reader_npu=not args.reader_cpu,
        strict_npu=args.strict_npu,
        allow_long_context=args.allow_long_context,
        cloud_model=args.cloud_model,
        api_key=args.api_key,
        endpoint=args.endpoint,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        cloud_context_chars=args.max_context_chars,
        escalate_empty=not args.no_escalate,
        verbose=not (args.quiet or args.json),
    )

    try:
        reply = engine.answer(question, context, force=args.force, context_source=source)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(reply), indent=2, ensure_ascii=False))
    elif args.quiet:
        print(reply.answer)
    else:
        print(reply.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
