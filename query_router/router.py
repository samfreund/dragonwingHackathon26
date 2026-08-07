"""Route a (question, context) pair to the right QA backend, decided on-device.

Two backends with very different costs:

  distilbert-base-cased-distilled-squad -- 66M params, *extractive*, runs on
     the Hexagon NPU (see ../npu_qa). It can only return a short verbatim span
     copied out of the context. Milliseconds, no cloud, no tokens billed.
  Llama-3.1-8B -- ~120x the parameters. Needed when the answer has to be
     reasoned about, summarised, computed or rewritten, or simply is not
     written down in the context.

How the decision is made
------------------------
The router is Qwen2.5-0.5B-Instruct (Apache-2.0) run as a single ONNX Runtime
*prefill* pass -- it never generates text. The prompt ends exactly where the
model must emit one word, and we read the logits for the "Yes" and "No" tokens
straight off the last position. One forward pass, no KV cache, no sampling
loop: deterministic, and it yields a calibrated confidence rather than a string
to regex.

The model is not asked to pick a backend. It is asked a concrete semantic
question it is far better at:

    "Is the answer written in the passage as a short phrase that could be
     copied word-for-word?"

That reframing matters more than it looks. Asking a 0.5B model to choose
between labelled options ("answer A or B") produced a severe label prior: on
content-free input it answered A with p=0.75-0.91, and it routed 12 of 12
generative questions to the small model -- 45% accuracy, useless. The same
rubric asked as Yes/No has a near-zero content-free prior (-0.13 logits) and
scores 95%. See `--calibrate` below for the fallback when a prior does appear.

Two cheap deterministic rules run *before* the LLM, because they encode hard
capability limits the model shouldn't get a vote on:

  * no context at all             -> Llama (nothing to extract from)
  * context longer than the       -> Llama (npu_qa truncates at 384 tokens, so
    reader's input window            the answer may be cut away entirely)

Measured on the 22-case development set in `--self-test`:

    threshold   accuracy   extractive recall   generative recall
      0.50       19/22          10/10               9/12
      0.60       21/22          10/10              11/12     <- default
      0.70       20/22           9/10              11/12

The 0.60 default sits on a plateau rather than a spike, and the threshold is
deliberately asymmetric: routing an extractive question to Llama costs latency,
routing a synthesis question to DistilBERT produces a confidently wrong answer.
Known weak spot: questions whose answer needs arithmetic over numbers that *are*
present in the passage ("what is the average of the three figures") still read
as copyable. Those are the ones to watch if you re-tune.

Where this runs
---------------
The router defaults to the **CPU**, deliberately, and the measurements back it
up. `--npu` does work -- QNN EP accepts this graph and becomes the active
provider -- but on an X-Elite it produced identical routing (21/22) at a median
532 ms per decision versus 490 ms on the CPU, plus a much longer session build
while the graph is compiled. No speedup, because a decoder graph with dynamic
sequence length and KV cache mostly falls back node-by-node.

So the CPU is the right home for the router twice over: it is faster here, and
it leaves the Hexagon free for the DistilBERT session this router is feeding
rather than making the two contend for one accelerator. The supported NPU path
for real LLMs on this box is onnxruntime-genai with a Genie bundle, not an
ONNX decoder graph driven by hand.

Setup
-----
    .\\run.ps1
    .venv\\Scripts\\python.exe router.py --question "..." --context "..."

First run downloads the router model (~790 MB) into the HF cache.

Usage
-----
    # single query
    python router.py --question "Who wrote Hamlet?" \\
        --context "William Shakespeare wrote Hamlet around 1600."

    # context from a file, machine-readable output
    python router.py --question "..." --context-file notes.txt --json

    # score a whole file of {"question": ..., "context": ...} JSON lines
    python router.py --batch queries.jsonl

    # labelled development-set check of the routing prompt
    python router.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROUTER_REPO = "onnx-community/Qwen2.5-0.5B-Instruct"
ROUTER_ONNX = "onnx/model_q4.onnx"

# Capability limits of the small backend -- must match ../npu_qa/npu_qa.py.
QA_TOKENIZER_REPO = "distilbert-base-cased-distilled-squad"
QA_SEQ_LEN = 384
QA_SPECIAL_TOKENS = 3  # [CLS] question [SEP] context [SEP]

DISTILBERT = "distilbert-base-cased-distilled-squad"
LLAMA = "Llama-3.1-8B"

# Route to the small model only when the router is at least this confident.
# Asymmetric on purpose -- see the module docstring.
DEFAULT_THRESHOLD = 0.60

# Long contexts are truncated head+tail before being shown to the router. It
# only has to recognise the *shape* of the task, not read the whole document,
# and prefill cost is linear in prompt length.
DEFAULT_CONTEXT_HEAD = 900
DEFAULT_CONTEXT_TAIL = 300

# Content-free inputs used to measure the label prior when --calibrate is on.
CALIBRATION_FILLERS = [("N/A", "N/A"), ("", ""), ("[none]", "[none]")]

_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
}


@dataclass
class RoutingDecision:
    """Which backend should answer, and why."""

    model: str
    confidence: float
    reason: str
    decided_by: str  # "rule" | "llm"
    p_extractive: float | None = None
    latency_ms: float = 0.0

    def render(self) -> str:
        pe = "n/a" if self.p_extractive is None else f"{self.p_extractive:.3f}"
        return (
            f"route     -> {self.model}\n"
            f"confidence   {self.confidence:.3f}  (p_extractive={pe})\n"
            f"decided by   {self.decided_by}\n"
            f"reason       {self.reason}\n"
            f"latency      {self.latency_ms:.1f} ms"
        )


# -- routing prompt ----------------------------------------------------------

SYSTEM_PROMPT = """You judge whether a question about a passage can be answered by copying words directly out of the passage.

Answer Yes if the answer is written in the passage as a short phrase that could be highlighted word-for-word: a name, date, number, place, title, or term.

Answer No if answering needs anything more than copying: summarising, explaining, comparing, reasoning across sentences, arithmetic, translating, rewriting, writing something new, giving an opinion or judgement, a long or multi-part answer, or if the passage does not contain the answer.

Answer with exactly one word: Yes or No."""

# Few-shot turns. A 0.5B model follows a rubric far more reliably with worked
# examples than with instructions alone; these are kept tiny so they cost ~200
# tokens of prefill. Three Yes and three No, alternating, to avoid teaching a
# position or frequency bias.
#
# Tempting additions are not always improvements: a seventh example covering
# comparison arithmetic ("how much more expensive is X") *lowered* development
# accuracy from 95% to 82%. Re-measure before extending this list.
FEW_SHOT: list[tuple[str, str]] = [
    ("Passage: The Eiffel Tower was completed in 1889 for the World's Fair.\nQuestion: When was the Eiffel Tower completed?", "Yes"),
    ("Passage: Q1 revenue was 4.2M, Q2 was 5.1M, Q3 was 3.8M.\nQuestion: What was the total revenue over the three quarters?", "No"),
    ("Passage: The device ships with 16 GB of RAM and a 1 TB SSD.\nQuestion: How much RAM does the device have?", "Yes"),
    ("Passage: The build failed after the cache was invalidated and the runner ran out of disk.\nQuestion: Explain why the build failed and how to prevent it.", "No"),
    ("Passage: Dr. Amara Osei has led the lab since 2018.\nQuestion: Who leads the lab?", "Yes"),
    ("Passage: The policy covers water damage but excludes flooding from natural sources.\nQuestion: Would a burst river bank be covered?", "No"),
]


def _truncate(text: str, head: int, tail: int) -> str:
    text = text.strip()
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n[...]\n{text[-tail:]}"


def build_messages(question: str, context: str, head: int, tail: int) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, answer in FEW_SHOT:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": answer})
    messages.append(
        {
            "role": "user",
            "content": f"Passage: {_truncate(context, head, tail)}\nQuestion: {question.strip()}",
        }
    )
    return messages


# -- ONNX Runtime session ----------------------------------------------------


def build_session(model_path: Path, use_npu: bool = False):
    """Create an InferenceSession for the router LLM.

    CPU by default -- see the module docstring for why the NPU is intentionally
    left to the DistilBERT workload. The QNN path mirrors ../npu_qa: the plugin
    EP must be registered and the device selected explicitly, because
    onnxruntime-qnn 2.x never appears in get_available_providers() and the
    legacy providers=[...] form silently falls back to CPU.
    """
    import onnxruntime as ort

    if not use_npu:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    import onnxruntime_qnn as qnn

    os.add_dll_directory(os.path.dirname(qnn.__file__))
    ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())

    npu_devices = [
        d
        for d in ort.get_ep_devices()
        if d.ep_name == "QNNExecutionProvider" and str(d.device.type).endswith("NPU")
    ]
    if not npu_devices:
        raise RuntimeError(
            "No Hexagon NPU device found via QNNExecutionProvider. Drop --npu, or "
            "confirm this is a Snapdragon host running a native arm64 Python."
        )

    opts = ort.SessionOptions()
    opts.add_provider_for_devices(npu_devices, {})
    return ort.InferenceSession(str(model_path), sess_options=opts)


def ensure_model(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"[setup] resolving {repo}/{filename} ...", file=sys.stderr)
    return Path(hf_hub_download(repo, filename))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


# -- the router --------------------------------------------------------------


class QueryRouter:
    """Decides between the extractive reader and the 8B generative model.

    Holds the ONNX session and both tokenizers, so batch use pays session
    construction once.
    """

    def __init__(
        self,
        repo: str = ROUTER_REPO,
        onnx_file: str = ROUTER_ONNX,
        use_npu: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
        allow_long_context: bool = False,
        calibrate: bool = False,
        context_head: int = DEFAULT_CONTEXT_HEAD,
        context_tail: int = DEFAULT_CONTEXT_TAIL,
    ) -> None:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self.threshold = threshold
        self.allow_long_context = allow_long_context
        self.context_head = context_head
        self.context_tail = context_tail

        model_path = ensure_model(repo, onnx_file)
        self.config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.session = build_session(model_path, use_npu=use_npu)
        self.provider = self.session.get_providers()[0]

        # Each answer must be a single token, or comparing first-token logits
        # would be comparing the wrong thing.
        self.token_yes = self._single_token("Yes")
        self.token_no = self._single_token("No")

        self._qa_tokenizer = None  # lazy: only needed for the length rule
        self.bias = self._content_free_bias() if calibrate else 0.0

    def _single_token(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Router label {text!r} is not a single token for this tokenizer "
                f"(got {ids}); pick different labels."
            )
        return ids[0]

    # -- calibration --

    def _content_free_bias(self) -> float:
        """Measure the model's label prior on content-free input.

        Contextual calibration (Zhao et al., 2021): whatever the model answers
        when there is nothing to judge is prior, not evidence, so subtract it.
        Off by default because the Yes/No framing already measures near zero on
        Qwen2.5-0.5B and subtracting noise costs accuracy -- but if you swap in
        another router model with `--model-repo`, turn this on and re-run
        `--self-test`. On the A/B-labelled framing this router used to use, it
        moved development accuracy from 45% to 86%.
        """
        margins = [self._margin(q, c) for q, c in CALIBRATION_FILLERS]
        return float(np.mean(margins))

    # -- deterministic pre-checks --

    def _qa_context_budget(self, question: str, context: str) -> tuple[int, int]:
        """(context_tokens, budget) measured with DistilBERT's own tokenizer."""
        if self._qa_tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._qa_tokenizer = AutoTokenizer.from_pretrained(QA_TOKENIZER_REPO)
            except Exception:  # offline / no cache -- fall back to an estimate
                self._qa_tokenizer = False

        if self._qa_tokenizer is False:
            n_ctx = len(context) // 4 + 1
            n_q = len(question) // 4 + 1
        else:
            n_ctx = len(self._qa_tokenizer.encode(context, add_special_tokens=False))
            n_q = len(self._qa_tokenizer.encode(question, add_special_tokens=False))
        return n_ctx, QA_SEQ_LEN - n_q - QA_SPECIAL_TOKENS

    def _rule_check(self, question: str, context: str) -> RoutingDecision | None:
        if not context or not context.strip():
            return RoutingDecision(
                model=LLAMA,
                confidence=1.0,
                reason="no context supplied -- there is no passage to extract a span from",
                decided_by="rule",
            )

        n_ctx, budget = self._qa_context_budget(question, context)
        if n_ctx > budget and not self.allow_long_context:
            return RoutingDecision(
                model=LLAMA,
                confidence=1.0,
                reason=(
                    f"context is {n_ctx} tokens but the extractive reader sees only "
                    f"{budget} (384-token window); the answer may be truncated away"
                ),
                decided_by="rule",
            )
        return None

    # -- the LLM forward pass --

    def _prefill_logits(self, input_ids: np.ndarray) -> np.ndarray:
        """Run one prefill pass and return the logits at the final position."""
        seq_len = input_ids.shape[1]
        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        }

        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        n_kv = self.config.get("num_key_value_heads", self.config["num_attention_heads"])

        for inp in self.session.get_inputs():
            if inp.name in feeds:
                continue
            if inp.name == "position_ids":
                feeds[inp.name] = np.arange(seq_len, dtype=np.int64)[None, :]
            elif inp.name.startswith("past_key_values"):
                # Empty cache: this is a single pass, there is no history.
                dtype = _ORT_TO_NP.get(inp.type, np.float32)
                feeds[inp.name] = np.zeros((1, n_kv, 0, head_dim), dtype=dtype)

        # Ask for logits only -- the present.* KV outputs are dead weight here.
        (logits,) = self.session.run(["logits"], feeds)
        return logits[0, -1].astype(np.float64)

    def _margin(self, question: str, context: str) -> float:
        """logit("Yes") - logit("No") at the decision position."""
        messages = build_messages(question, context, self.context_head, self.context_tail)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = np.asarray(
            [self.tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64
        )
        logits = self._prefill_logits(input_ids)
        return float(logits[self.token_yes] - logits[self.token_no])

    def route(self, question: str, context: str) -> RoutingDecision:
        started = time.perf_counter()

        ruled = self._rule_check(question, context)
        if ruled is not None:
            ruled.latency_ms = (time.perf_counter() - started) * 1000
            return ruled

        # sigmoid of the Yes-vs-No logit gap == softmax restricted to those two
        # candidates. Mass on any other token is irrelevant here.
        p_extractive = _sigmoid(self._margin(question, context) - self.bias)

        if p_extractive >= self.threshold:
            decision = RoutingDecision(
                model=DISTILBERT,
                confidence=p_extractive,
                reason="answer looks like a short verbatim span present in the context",
                decided_by="llm",
                p_extractive=p_extractive,
            )
        else:
            leaned_extractive = p_extractive >= 0.5
            decision = RoutingDecision(
                model=LLAMA,
                confidence=1.0 - p_extractive,
                reason=(
                    f"router leaned extractive but below the {self.threshold:.2f} "
                    "threshold -- escalating rather than risk a confidently wrong span"
                    if leaned_extractive
                    else "answering needs synthesis, reasoning or knowledge beyond a copied span"
                ),
                decided_by="llm",
                p_extractive=p_extractive,
            )

        decision.latency_ms = (time.perf_counter() - started) * 1000
        return decision


# -- self-test ---------------------------------------------------------------

# Development set. The prompt and threshold above were tuned against it, so
# treat it as a smoke test and a regression guard, not an unbiased benchmark.
# None of these share wording with the few-shot block.
SELF_TEST: list[tuple[str, str, str]] = [
    ("Who is the chief executive?", "Rivera Foods named Priya Anand chief executive in 2021.", DISTILBERT),
    ("What port does the service listen on?", "The daemon binds to port 8443 and logs to /var/log/svc.", DISTILBERT),
    ("Which city hosted the 1992 summer games?", "Barcelona hosted the 1992 summer games, beating six other bids.", DISTILBERT),
    ("What is the warranty period?", "All units carry a 36-month warranty from the date of purchase.", DISTILBERT),
    ("What material is the frame made of?", "The frame is milled from 6061 aluminium and then anodised.", DISTILBERT),
    ("When does the trial end?", "The trial began on 1 June and ends on 30 September.", DISTILBERT),
    ("What is the model number?", "Service applies to model XR-4400 and later revisions.", DISTILBERT),
    ("Who signed the agreement?", "The agreement was signed by Helena Vasquez on behalf of the board.", DISTILBERT),
    ("What speed does the uplink run at?", "The uplink runs at 25 Gbps over single-mode fibre.", DISTILBERT),
    ("Which database does the service use?", "Data is persisted in PostgreSQL with a Redis cache in front.", DISTILBERT),
    ("Summarise the incident in one paragraph.", "At 02:14 the primary replica lost quorum. Failover stalled for eleven minutes while the coordinator retried. Traffic was drained manually and service returned at 02:41.", LLAMA),
    ("What is the average of the three latency figures?", "p50 was 12 ms, p90 was 34 ms and p99 was 120 ms.", LLAMA),
    ("Should we adopt this policy? Justify your answer.", "The policy requires all contractors to complete security training within 30 days of onboarding.", LLAMA),
    ("Rewrite this notice so an ordinary customer can understand it.", "Pursuant to clause 7(b), remittance obligations accrue upon invoice issuance notwithstanding delivery status.", LLAMA),
    ("What does this trend imply about next quarter?", "Signups grew 4% in July, 3% in August and 1% in September.", LLAMA),
    ("Who won the 2018 football world cup?", "Soil pH in the northern plots ranged from 5.8 to 6.4 across the season.", LLAMA),
    ("List the steps needed to reproduce this, in order.", "User clicked export. Log shows auth refresh, then a 502 from the render service, then a retry that also failed.", LLAMA),
    ("How much more does the premium plan cost than the basic plan?", "Basic is 12 per month. Premium is 29 per month.", LLAMA),
    ("Explain the trade-off between the two storage tiers.", "Hot tier keeps data on NVMe with instant reads. Cold tier is object storage, cheaper but minutes to restore.", LLAMA),
    ("Is this configuration safe for production?", "max_connections=10000, fsync=off, shared_buffers=128MB", LLAMA),
    ("Translate the warranty clause into French.", "The warranty does not cover damage caused by misuse.", LLAMA),
    ("Draft a one-line commit message for this change.", "Removed the deprecated retry helper and moved callers onto the shared backoff utility.", LLAMA),
]

# The development set is not fully separable by this router (see the docstring's
# note on arithmetic questions); fail the run only on a real regression.
SELF_TEST_MIN_ACCURACY = 0.90


def run_self_test(router: QueryRouter) -> int:
    correct, latencies = 0, []
    print(f"{'expected':>10}  {'routed':>10}  {'p_ext':>6}  {'by':>4}  question")
    print("-" * 92)
    for question, context, expected in SELF_TEST:
        d = router.route(question, context)
        latencies.append(d.latency_ms)
        got = "distilbert" if d.model == DISTILBERT else "llama"
        exp = "distilbert" if expected == DISTILBERT else "llama"
        ok = got == exp
        correct += ok
        pe = "  n/a " if d.p_extractive is None else f"{d.p_extractive:6.3f}"
        print(f"{exp:>10}  {got:>10}  {pe}  {d.decided_by:>4}  {' ' if ok else 'X'} {question}")

    total = len(SELF_TEST)
    accuracy = correct / total
    print("-" * 92)
    print(f"{correct}/{total} routed as expected ({accuracy:.0%})")
    print(f"median decision latency: {np.median(latencies):.0f} ms on {router.provider}")
    if accuracy < SELF_TEST_MIN_ACCURACY:
        print(f"FAIL: below the {SELF_TEST_MIN_ACCURACY:.0%} regression floor")
        return 1
    return 0


# -- CLI ---------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--question", default=None, help="The user query to route")
    ap.add_argument("--context", default=None, help="Reference text (overrides --context-file)")
    ap.add_argument("--context-file", default=None, help="Path to a file holding the reference text")
    ap.add_argument("--batch", default=None, help='JSONL file of {"question":..., "context":...}')
    ap.add_argument("--self-test", action="store_true", help="Run the labelled development check")
    ap.add_argument("--json", action="store_true", help="Emit the decision as JSON")
    ap.add_argument("--npu", action="store_true", help="Try the Hexagon NPU (see module docstring)")
    ap.add_argument("--model-repo", default=ROUTER_REPO, help="HF repo for the router LLM")
    ap.add_argument("--onnx-file", default=ROUTER_ONNX, help="ONNX variant within that repo")
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum p(extractive) required to use the small model (default %(default)s)",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Subtract the model's content-free label prior (worth it if you swap --model-repo)",
    )
    ap.add_argument(
        "--allow-long-context",
        action="store_true",
        help="Do not auto-escalate contexts longer than the reader's 384-token window",
    )
    args = ap.parse_args()

    if not (args.self_test or args.batch or args.question):
        ap.error("give --question, --batch or --self-test")

    router = QueryRouter(
        repo=args.model_repo,
        onnx_file=args.onnx_file,
        use_npu=args.npu,
        threshold=args.threshold,
        allow_long_context=args.allow_long_context,
        calibrate=args.calibrate,
    )
    bias = f", label prior {router.bias:+.2f} removed" if args.calibrate else ""
    print(f"[router] {args.model_repo} on {router.provider}{bias}", file=sys.stderr)

    if args.self_test:
        return run_self_test(router)

    if args.batch:
        for line in Path(args.batch).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            d = router.route(row["question"], row.get("context", ""))
            print(json.dumps({**row, "routing": asdict(d)}, ensure_ascii=False))
        return 0

    if args.context:
        context = args.context
    elif args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")
    else:
        context = ""

    decision = router.route(args.question, context)
    print(json.dumps(asdict(decision), indent=2) if args.json else decision.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
