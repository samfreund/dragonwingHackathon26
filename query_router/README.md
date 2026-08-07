# Query router — pick the QA backend with a small on-device LLM

Given a question and its context text, decide which engine should answer it:

| | Backend | Cost | Can answer |
|---|---|---|---|
| **small** | `distilbert-base-cased-distilled-squad` (66M) | milliseconds, on the Hexagon NPU, nothing billed | only what is written in the context as a short verbatim span |
| **large** | `Llama-3.1-8B` (~120× the parameters) | seconds and/or cloud tokens | anything — synthesis, reasoning, arithmetic, rewriting, world knowledge |

The extractive reader is nearly free but silently wrong when asked for anything
it cannot copy out of the passage. The router's job is to catch exactly that
case before it happens.

The small model half of this pair is the sibling sample [`../npu_qa`](../npu_qa),
which runs DistilBERT on the Hexagon NPU.

## How the decision is made

Qwen2.5-0.5B-Instruct (Apache-2.0), run as a **single ONNX Runtime prefill
pass** — it never generates text. The prompt stops exactly where the model must
emit one word, and we read the logits for the `Yes` and `No` tokens off the last
position. One forward pass, no KV cache, no sampling loop: deterministic, and it
gives a real confidence instead of a string to regex.

The model is never asked to pick a backend. It is asked a concrete semantic
question it is much better at:

> *Is the answer written in the passage as a short phrase that could be copied
> word-for-word?*

That reframing did most of the work. Asking the same 0.5B model to choose
between labelled options (*"answer A or B"*) produced a severe label prior — on
content-free input it answered `A` with p=0.75–0.91, and it sent **12 of 12**
generative questions to the small model. 45% accuracy, i.e. worse than useless,
since every error is the dangerous direction. The identical rubric asked as
Yes/No has a near-zero content-free prior (−0.13 logits) and scores 95%.

Two deterministic rules run *before* the LLM, encoding hard capability limits
the model shouldn't get a vote on:

- **no context at all** → Llama (nothing to extract from)
- **context longer than the reader's 384-token window** → Llama, because
  `npu_qa` truncates and the answer may be cut away entirely. Override with
  `--allow-long-context`.

## Measured results

22-case development set, `--self-test`:

| threshold | accuracy | extractive recall | generative recall |
|---|---|---|---|
| 0.50 | 19/22 | 10/10 | 9/12 |
| **0.60** | **21/22 (95%)** | **10/10** | **11/12** |
| 0.70 | 20/22 | 9/10 | 11/12 |

0.60 sits on a plateau rather than a spike. The threshold is asymmetric on
purpose: routing an extractive question to Llama costs latency, routing a
synthesis question to DistilBERT produces a confidently wrong answer.

**Latency** — median per decision on a Snapdragon X-Elite:

| provider | median | accuracy |
|---|---|---|
| CPU (default) | **490 ms** | 21/22 |
| QNN / Hexagon NPU (`--npu`) | 532 ms | 21/22 |

`--npu` works — QNN EP accepts the graph and becomes the active provider — but
it is not faster, because a decoder graph with dynamic sequence length and KV
cache mostly falls back node by node, and the session build costs a long graph
compile. So the CPU is the right home twice over: quicker here, and it leaves
the Hexagon free for the DistilBERT session the router is feeding. The supported
NPU path for real LLMs on this box is onnxruntime-genai with a Genie bundle.

**Known weak spot:** questions needing arithmetic over numbers that *are* present
in the passage ("what is the average of the three figures") still read as
copyable. That is the one remaining miss in the development set.

> The prompt and threshold were tuned against this 22-case set, so treat it as a
> smoke test and regression guard, not an unbiased benchmark.

## Setup

```powershell
.\run.ps1
```

Builds a venv from a native arm64 Python (required for the optional `--npu`
extras) and installs dependencies. First run of the router downloads ~790 MB of
model into the HF cache.

## Usage

```powershell
# route one query
.venv\Scripts\python.exe router.py --question "Who wrote Hamlet?" `
    --context "William Shakespeare wrote Hamlet around 1600."

# machine-readable, context from a file
.venv\Scripts\python.exe router.py --question "..." --context-file notes.txt --json

# a whole file of {"question": ..., "context": ...} JSON lines
.venv\Scripts\python.exe router.py --batch queries.jsonl

# the labelled development check
.venv\Scripts\python.exe router.py --self-test
```

Useful flags: `--threshold` (trade extractive coverage against safety),
`--model-repo` / `--onnx-file` (swap the router model), `--calibrate` (subtract
the model's content-free label prior — off by default, but worth turning on if
you swap `--model-repo`, since it took the A/B framing from 45% to 86%).

## Using it as a library

```python
from router import QueryRouter, DISTILBERT

router = QueryRouter()          # loads once; reuse across queries
decision = router.route(question, context)

if decision.model == DISTILBERT:
    answer = run_npu_qa(question, context)      # ../npu_qa
else:
    answer = call_llama(question, context)      # your 8B endpoint

print(decision.reason, decision.confidence)
```

`route()` returns a `RoutingDecision` with `model`, `confidence`, `reason`,
`decided_by` (`"rule"` or `"llm"`), `p_extractive` and `latency_ms`.

## Files

| File | Purpose |
|---|---|
| `router.py` | The router — library + CLI |
| `requirements.txt` | Portable CPU dependencies |
| `requirements-npu.txt` | Optional `onnxruntime-qnn` for `--npu` (win-arm64 only) |
| `run.ps1` | Arch-aware venv bootstrap |
