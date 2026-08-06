"""Extractive question answering on the Hexagon NPU.

Reference text + question in, an answer extracted from that text out —
via ONNX Runtime's QNN Execution Provider (onnxruntime-qnn), targeting the
Hexagon NPU on a Snapdragon X-Elite / X2-Elite Windows laptop.

Model: distilbert-base-cased-distilled-squad (Apache-2.0, no gating), run
as a plain fp32 ONNX graph. QNN EP dispatches supported ops to the NPU and
falls back the rest to CPU automatically.

Setup
-----
This MUST run under a native arm64 Python interpreter — onnxruntime-qnn has
no wheel for emulated x86-on-ARM (Prism) or genuine x86_64. Use run.ps1,
which detects/creates the right venv automatically:

    .\\run.ps1
    .venv\\Scripts\\python.exe npu_qa.py --question "..." --context "..."

First run downloads and prepares the model (~260 MB); later runs reuse the
cached file next to this script.

Usage
-----
    # Use the built-in example
    python npu_qa.py

    # Your own text
    python npu_qa.py --question "Who wrote Hamlet?" \\
        --context "William Shakespeare wrote Hamlet around 1600."

    # Context from a file
    python npu_qa.py --question "..." --context-file notes.txt

    # Force CPU (for comparison, or on a non-arm64 dev host)
    python npu_qa.py --cpu --question "..." --context "..."
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "distilbert_qa_static.onnx"
HF_REPO = "Xenova/distilbert-base-cased-distilled-squad"
HF_FILE = "onnx/model.onnx"
TOKENIZER_REPO = "distilbert-base-cased-distilled-squad"

SEQ_LEN = 384
MAX_ANSWER_LEN = 30

MAX_ANSWER_LEN = 256

EXAMPLE_CONTEXT = (
    "Marie Curie was a Polish and naturalised-French physicist and chemist "
    "who conducted pioneering research on radioactivity. She was the first "
    "woman to win a Nobel Prize, the first person to win a Nobel Prize "
    "twice, and the only person to win a Nobel Prize in two scientific "
    "fields."
)
EXAMPLE_QUESTION = "How many Nobel Prizes did Marie Curie win?"


# ── model prep (one-time, cached) ───────────────────────────────────────────


def ensure_model(model_path: Path) -> Path:
    """Download the QA model and patch its dynamic ONNX shapes to static.

    QNN/NPU compilation needs fixed tensor shapes; the upstream export has
    dynamic (batch_size, sequence_length) axes, so this pins them to
    (1, SEQ_LEN) once and caches the result.
    """
    if model_path.exists():
        return model_path

    import onnx
    from huggingface_hub import hf_hub_download
    from onnx.tools import update_model_dims

    print(f"[setup] downloading {HF_REPO}/{HF_FILE} ...")
    src = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)

    model = onnx.load(src)
    static = {"batch_size": 1, "sequence_length": SEQ_LEN}

    def dims(value_infos):
        return {
            v.name: [static.get(d.dim_param, d.dim_value) for d in v.type.tensor_type.shape.dim]
            for v in value_infos
        }

    model = update_model_dims.update_inputs_outputs_dims(
        model, dims(model.graph.input), dims(model.graph.output)
    )
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    print(f"[setup] saved static-shape model -> {model_path.name}")
    return model_path


# ── inference session ───────────────────────────────────────────────────────


def build_session(model_path: Path, use_npu: bool = True):
    """Create an InferenceSession, optionally pinned to the Hexagon NPU.

    onnxruntime-qnn 2.x is a *plugin* execution provider: it must be
    registered and the NPU device found explicitly via get_ep_devices() —
    it never appears in get_available_providers(), and the older
    providers=[('QNNExecutionProvider', {...})] argument silently falls
    back to CPU instead of raising.
    """
    import onnxruntime as ort

    if not use_npu:
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

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
            "No Hexagon NPU device found via QNNExecutionProvider. Run with --cpu, "
            "or confirm this is a Snapdragon X-Elite/X2-Elite host with a native "
            "arm64 Python (see the module docstring)."
        )

    session_options = ort.SessionOptions()
    session_options.add_provider_for_devices(npu_devices, {})
    sess = ort.InferenceSession(str(model_path), sess_options=session_options)

    active = sess.get_providers()[0]
    if active != "QNNExecutionProvider":
        raise RuntimeError(
            f"NPU (QNNExecutionProvider) did not become the active provider "
            f"(got '{active}'). Inference would run on the CPU instead."
        )
    return sess


# ── question answering ──────────────────────────────────────────────────────


def answer_question(sess, tokenizer, context: str, question: str) -> tuple[str, dict]:
    """Return (answer_text, debug_info) for `question` over `context`.

    Picks the (start, end) span maximizing start_logit + end_logit subject
    to start <= end and a max answer length — independent argmax on each
    logit separately can (and does) produce an invalid start > end span.
    """
    enc = tokenizer(
        question,
        context,
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
        return_tensors="np",
    )
    start_logits, end_logits = sess.run(
        None,
        {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        },
    )

    seq_len_used = int(enc["attention_mask"].sum())
    s, e = start_logits[0][:seq_len_used], end_logits[0][:seq_len_used]

    best_score, best_start, best_end = -1e9, 0, 0
    for i in range(len(s)):
        for j in range(i, min(i + MAX_ANSWER_LEN, len(e))):
            score = s[i] + e[j]
            if score > best_score:
                best_score, best_start, best_end = score, i, j

    answer_ids = enc["input_ids"][0][best_start : best_end + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return answer, {"start": best_start, "end": best_end, "score": float(best_score)}


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question", default=EXAMPLE_QUESTION)
    ap.add_argument("--context", default=None, help="Reference text (overrides --context-file)")
    ap.add_argument("--context-file", default=None, help="Path to a text file with the reference text")
    ap.add_argument("--model", default=str(MODEL_PATH), help="Path to the static-shape ONNX model")
    ap.add_argument("--cpu", action="store_true", help="Force CPU execution instead of the NPU")
    args = ap.parse_args()

    if args.context:
        context = args.context
    elif args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")
    else:
        context = EXAMPLE_CONTEXT

    from transformers import AutoTokenizer

    model_path = ensure_model(Path(args.model))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)

    print(f"Loading model on {'CPU' if args.cpu else 'Hexagon NPU'}: {model_path.name}")
    sess = build_session(model_path, use_npu=not args.cpu)
    print(f"Active execution providers: {sess.get_providers()}")

    answer, info = answer_question(sess, tokenizer, context, args.question)

    print(f"\nQuestion: {args.question}")
    print(f"Answer:   {answer!r}")
    print(f"(span {info['start']}:{info['end']}, score {info['score']:.2f})")


if __name__ == "__main__":
    main()
