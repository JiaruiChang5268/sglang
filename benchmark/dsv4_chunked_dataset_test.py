#!/usr/bin/env python3
"""DSV4 chunked-prefill dataset test: token-parity + accuracy on real data.

Complements dsv4_chunked_parity.py (which uses synthetic prompts). This one
drives a REAL dataset (jsonl/json with question/answer keys, same shape as
benchmark/reasoning_benchmark), so prompts are naturally long and actually
exercise chunked prefill end-to-end. Two checks:

  * token parity  : chunked vs baseline produce identical greedy tokens.
  * accuracy      : extracted answer matches the dataset's gold answer
                    (rough heuristic; a sanity signal, not a rigorous eval).

Workflow:

    # server WITH chunking. chunked_prefill_size must be a multiple of page_size
    # (and is divided by dp_size under DP-attention), so the per-DP chunk is
    # always 128-aligned; use page_size*dp_size for the smallest legal chunk.
    python dsv4_chunked_dataset_test.py run --host H --port P \
        --data-path /home/cjr/dataset/aime2025.jsonl \
        --question-key question --answer-key answer \
        --num-prompts 16 --max-new-tokens 2048 --out chunked.jsonl

    # restart server WITHOUT chunking (--chunked-prefill-size -1), re-run --out baseline.jsonl

    python dsv4_chunked_dataset_test.py compare --a chunked.jsonl --b baseline.jsonl   # token parity
    python dsv4_chunked_dataset_test.py acc     --a chunked.jsonl --b baseline.jsonl   # accuracy each + delta

Greedy only (temperature=0). See dsv4_chunked_parity.py for the determinism /
coverage notes (128-aligned chunk sizes skip c128 cross-chunk gather, etc.).
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import random
import re
import sys
import time
from typing import List, Optional, Tuple

import requests

GPQA_DIAMOND_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
)

GPQA_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


# --------------------------------------------------------------------------- #
# server helpers
# --------------------------------------------------------------------------- #
def _wait_for_server(host: str, port: int, timeout: float) -> None:
    url = f"http://{host}:{port}/get_model_info"
    deadline = time.time() + timeout
    waited = False
    while True:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                if waited:
                    print(" ready.")
                return
        except requests.exceptions.RequestException:
            pass
        if time.time() >= deadline:
            sys.exit(f"ERROR: server {host}:{port} not reachable within {timeout:.0f}s")
        if not waited:
            print(f"waiting for server {host}:{port} ", end="", flush=True)
            waited = True
        print(".", end="", flush=True)
        time.sleep(3)


def _get_chunk_size(host: str, port: int):
    try:
        r = requests.get(f"http://{host}:{port}/get_server_info", timeout=10)
        info = r.json() if r.status_code == 200 else None
    except (requests.exceptions.RequestException, ValueError):
        return None

    def _find(d):
        if isinstance(d, dict):
            if "chunked_prefill_size" in d:
                return d["chunked_prefill_size"]
            for v in d.values():
                got = _find(v)
                if got is not None:
                    return got
        return None

    return _find(info)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
_PADDING_LINE = (
    "This line is neutral padding for a prefill parity test and should be ignored.\n"
)


def _prepend_padding(prompt: str, pad_repeat: int) -> str:
    padding = ""
    if pad_repeat > 0:
        padding = (
            "The following padding is only for an inference-system parity test. "
            "Ignore it and solve the actual problem after the padding.\n"
            + _PADDING_LINE * pad_repeat
            + "\nActual problem:\n"
        )
    return padding + prompt


def build_prompt(question: str, pad_repeat: int = 0) -> str:
    prompt = (
        question
        + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    return _prepend_padding(prompt, pad_repeat)


def _load_csv_rows(path: str) -> List[dict]:
    if path == "gpqa":
        path = GPQA_DIAMOND_URL
    if path.startswith(("http://", "https://")):
        r = requests.get(path, timeout=60)
        r.raise_for_status()
        return list(csv.DictReader(io.StringIO(r.text)))
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _format_gpqa_rows(rows: List[dict]) -> List[dict]:
    rng = random.Random(0)
    out = []
    for idx, row in enumerate(rows):
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        permutation = rng.sample(range(4), 4)
        choices = [choices[i] for i in permutation]
        correct_answer = "ABCD"[choices.index(row["Correct Answer"])]
        prompt = GPQA_TEMPLATE.format(
            Question=row["Question"],
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
        out.append(
            {
                "question": row["Question"],
                "prompt": prompt,
                "gold": correct_answer,
                "dataset_idx": idx,
            }
        )
    return out


def build_row_prompt(row: dict, pad_repeat: int) -> str:
    if row.get("prompt") is not None:
        return _prepend_padding(row["prompt"], pad_repeat)
    return (
        build_prompt(row["question"], pad_repeat)
    )


def load_dataset(
    path: str,
    qkey: str,
    akey: Optional[str],
    start: int,
    limit: int,
    dataset_type: str,
) -> List[dict]:
    rows: List[dict] = []
    if dataset_type == "auto":
        dataset_type = "gpqa" if path == "gpqa" or path.endswith(".csv") else "qa"
    if dataset_type == "gpqa":
        rows = _format_gpqa_rows(_load_csv_rows(path))
        end = len(rows) if limit is None else start + limit
        return rows[start:end]

    files = (
        sorted(glob.glob(os.path.join(path, "*.jsonl")))
        if os.path.isdir(path)
        else [path]
    )
    for file in files:
        if file.endswith(".jsonl"):
            with open(file, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        rows.append(json.loads(ln))
        else:  # .json array
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
            rows.extend(data if isinstance(data, list) else data.get("data", []))
    out = []
    end = len(rows) if limit is None else start + limit
    for r in rows[start:end]:
        q = r.get(qkey)
        if q is None:
            continue
        out.append({"question": str(q), "gold": str(r.get(akey)) if akey else None})
    return out


# --------------------------------------------------------------------------- #
# answer extraction (rough heuristic — sanity only)
# --------------------------------------------------------------------------- #
def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    # 1) \boxed{...}
    m = list(re.finditer(r"\\boxed\{([^{}]*)\}", text))
    if m:
        return m[-1].group(1).strip()
    # 2) "answer is X" / "answer: X"
    m = list(re.finditer(r"answer\s*(?:is|:)\s*([^\n.]+)", text, re.IGNORECASE))
    if m:
        return m[-1].group(1).strip().rstrip(".")
    # 3) last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _norm(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"[\s,$]", "", str(s)).lower()


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    rows = load_dataset(
        args.data_path,
        args.question_key,
        args.answer_key,
        args.start_index,
        args.num_prompts,
        args.dataset_type,
    )
    if not rows:
        sys.exit(f"no rows loaded from {args.data_path} (check --question-key)")

    _wait_for_server(args.host, args.port, args.wait)
    chunk_size = _get_chunk_size(args.host, args.port)
    print(f"server (per-DP) chunked_prefill_size = {chunk_size}; {len(rows)} prompts")
    # The per-DP chunk is always a multiple of page_size (enforced by server_args,
    # and divided by dp_size under DP-attention) -> boundaries are 128-aligned ->
    # c128 cross-chunk never fires (clean by construction); c4 overlap is the only
    # cross-chunk path and IS exercised here. Smallest legal chunk for max c4
    # boundary stress: --chunked-prefill-size = page_size * dp_size.

    url = f"http://{args.host}:{args.port}/generate"
    results = []
    for i, row in enumerate(rows):
        prompt = build_row_prompt(row, args.pad_repeat)
        body = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": args.max_new_tokens,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
            "stream": False,
        }
        d = requests.post(url, json=body, timeout=args.timeout).json()
        meta = d.get("meta_info", {}) or {}
        token_ids = [t[1] for t in (meta.get("output_token_logprobs") or [])]
        ptok = meta.get("prompt_tokens")
        chunked = bool(chunk_size is not None and chunk_size > 0 and (ptok or 0) > chunk_size)
        results.append(
            {
                "idx": i,
                "dataset_idx": row.get("dataset_idx", args.start_index + i),
                "prompt": prompt,
                "question": row["question"],
                "gold": row["gold"],
                "text": d.get("text", ""),
                "token_ids": token_ids,
                "prompt_tokens": ptok,
                "completion_tokens": meta.get("completion_tokens"),
                "chunk_size": chunk_size,
                "chunked": chunked,
            }
        )
        print(f"[{i}] prompt_tokens={ptok} out={len(token_ids)} "
              f"{'CHUNKED' if chunked else 'not-chunked'}")

    with open(args.out, "w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    chunked_n = sum(1 for x in results if x["chunked"])
    print(f"wrote {len(results)} -> {args.out} ({chunked_n} actually chunked)")


def _load(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _match_prefix_len(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare(args: argparse.Namespace) -> None:
    A, B = _load(args.a), _load(args.b)
    n = min(len(A), len(B))
    cs_a = next((x.get("chunk_size") for x in A if x.get("chunk_size") is not None), None)
    chunked_n = sum(1 for x in A if x.get("chunked"))
    print(f"coverage: {args.a} chunk_size={cs_a}, {chunked_n}/{len(A)} actually chunked")
    if chunked_n == 0:
        print("  WARNING: no prompt chunked -> PASS is trivial (use smaller chunk size).")

    all_ok = True
    for i in range(n):
        ta, tb = A[i].get("token_ids") or [], B[i].get("token_ids") or []
        m, exact = _match_prefix_len(ta, tb), ta == tb
        tag = "CHUNKED" if A[i].get("chunked") else "not-chunked"
        if exact:
            print(f"[{i}] OK {tag} len={len(ta)} identical")
        else:
            all_ok = False
            print(f"[{i}] DIVERGE {tag} matched_prefix={m} lenA={len(ta)} lenB={len(tb)}")
    print("=" * 48)
    print("TOKEN PARITY PASS" if all_ok else "TOKEN PARITY FAIL")
    sys.exit(0 if all_ok else 1)


def _accuracy(rows: List[dict]) -> Tuple[int, int]:
    correct = total = 0
    for r in rows:
        if r.get("gold") in (None, "None"):
            continue
        total += 1
        if _norm(extract_answer(r.get("text", ""))) == _norm(r["gold"]):
            correct += 1
    return correct, total


def acc(args: argparse.Namespace) -> None:
    for name in (args.a, args.b):
        if name is None:
            continue
        rows = _load(name)
        c, t = _accuracy(rows)
        pct = 100.0 * c / t if t else 0.0
        chunked_n = sum(1 for x in rows if x.get("chunked"))
        print(f"{name}: accuracy {c}/{t} = {pct:.1f}%  ({chunked_n}/{len(rows)} chunked)")
    print("Accuracy is a rough heuristic-extracted sanity signal. The two runs' "
          "accuracy should match; both should be in the expected ballpark for the dataset.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="run a dataset through a server (greedy), save outputs")
    r.add_argument("--host", required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--data-path", required=True, help=".jsonl/.json/.csv dataset, URL, or 'gpqa'")
    r.add_argument(
        "--dataset-type",
        choices=["auto", "qa", "gpqa"],
        default="auto",
        help="qa uses --question-key/--answer-key; gpqa reads GPQA-diamond CSV columns",
    )
    r.add_argument("--question-key", default="question")
    r.add_argument("--answer-key", default=None, help="gold answer key (for acc mode)")
    r.add_argument("--start-index", type=int, default=0)
    r.add_argument("--num-prompts", type=int, default=16)
    r.add_argument(
        "--pad-repeat",
        type=int,
        default=0,
        help="prepend neutral padding lines to force prompt_tokens > chunk size",
    )
    r.add_argument("--max-new-tokens", type=int, default=2048)
    r.add_argument("--out", required=True)
    r.add_argument("--timeout", type=float, default=3600.0)
    r.add_argument("--wait", type=float, default=1800.0)
    r.set_defaults(func=run)

    c = sub.add_parser("compare", help="token-parity diff of two captures")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.set_defaults(func=compare)

    a = sub.add_parser("acc", help="accuracy (vs gold) of one or two captures")
    a.add_argument("--a", required=True)
    a.add_argument("--b", default=None)
    a.set_defaults(func=acc)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
