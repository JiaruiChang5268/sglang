#!/usr/bin/env python3
"""DSV4 chunked-prefill parity checker.

Chunked prefill is mathematically equivalent to single-shot prefill, so the
SAME prompt under greedy decoding must produce the SAME output tokens whether
or not the prefill was chunked. This script captures greedy outputs from a
running SGLang server and diffs two captures token-by-token.

Workflow (one server at a time; restart between captures):

    # A) server WITH chunking, e.g. launch with: --chunked-prefill-size 256
    python dsv4_chunked_parity.py run  --host 192.168.25.209 --port 9527 \
        --out chunked.jsonl --max-new-tokens 256 --concurrency 4

    # B) restart server WITHOUT chunking:        --chunked-prefill-size -1
    python dsv4_chunked_parity.py run  --host 192.168.25.209 --port 9527 \
        --out baseline.jsonl --max-new-tokens 256

    # C) compare
    python dsv4_chunked_parity.py compare --a chunked.jsonl --b baseline.jsonl

Determinism notes
-----------------
* Greedy only (temperature=0, top_p=1). Anything else is not reproducible.
* Cross-run float nondeterminism (TP/EP reductions, atomics) can make even two
  IDENTICAL-config runs differ late in a long output. So first run the SAME
  config twice and ``compare`` them to learn your noise floor:
      - identical                -> exact-token parity is valid, use it.
      - diverge only after many   -> that suffix is fp noise; a chunked-vs-baseline
        identical tokens             divergence at the SAME late point is OK,
                                     an EARLY divergence (first few tokens) is a bug.
  ``HCCL_DETERMINISTIC=true`` (already in your env) helps a lot here.

Coverage notes
--------------
* A prompt only exercises chunked prefill if its token count exceeds the
  per-DP --chunked-prefill-size. The script prints prompt_tokens per request
  so you can confirm chunking actually triggered.
* server_args enforces chunked_prefill_size % page_size == 0 (and divides it by
  dp_size under DP-attention), so the per-DP chunk is ALWAYS a multiple of
  page_size -> chunk boundaries are always 128-aligned. Consequence: c128 (HCA)
  blocks never straddle a boundary, so c128 cross-chunk gather never fires (it
  is clean by construction). c4 (CSA) overlap is the only cross-chunk path and
  fires on every boundary. For the most c4 boundary stress use the smallest
  legal chunk: --chunked-prefill-size = page_size * dp_size.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import requests

# A few fixed prompts. The long ones are padded with filler so they exceed a
# typical --chunked-prefill-size (256) and span multiple chunks (covering c4
# multi-block + c128). Keep these BYTE-IDENTICAL across the two captures.
_FILLER = "The mitochondria is the powerhouse of the cell. "


def default_prompts() -> List[str]:
    return [
        # short: below any chunk size -> never chunks (sanity; must always match)
        "What is 17 * 23? Think step by step, then give the final number.",
        # medium: ~a few hundred tokens -> a few chunks at chunk_size=256
        "Read the passage and answer.\n\n"
        + _FILLER * 60
        + "\n\nIn one sentence, what is the passage about?",
        # long: ~1.5k+ tokens -> many chunks, exercises c128 + many c4 blocks
        "Summarize the following passage in exactly one sentence.\n\n"
        + _FILLER * 220
        + "\n\nSummary:",
        # long, different length -> different chunk-boundary alignment
        "Continue the list logically: 2, 4, 8, 16, ... explain the rule.\n\n"
        + _FILLER * 130
        + "\n\nNow answer the question above.",
    ]


def _wait_for_server(host: str, port: int, timeout: float) -> None:
    """Poll /get_model_info until the server is up (handles 'still loading')."""
    url = f"http://{host}:{port}/get_model_info"
    deadline = time.time() + timeout
    waited = False
    while True:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                if waited:
                    print(" ready.")
                return
        except requests.exceptions.RequestException:
            pass
        if time.time() >= deadline:
            sys.exit(
                f"ERROR: server {host}:{port} not reachable within {timeout:.0f}s.\n"
                f"  - is it still loading? watch the server log for 'ready to roll'.\n"
                f"  - did it crash? check the server terminal for a traceback.\n"
                f"  - right host/port? (you used {host}:{port})"
            )
        if not waited:
            print(f"waiting for server {host}:{port} to be ready ", end="", flush=True)
            waited = True
        print(".", end="", flush=True)
        time.sleep(3)


def _get_chunk_size(host: str, port: int):
    """Best-effort: read the server's chunked_prefill_size from /get_server_info.

    Returns the int, or None if unavailable. -1 means chunking is OFF.
    """
    try:
        r = requests.get(f"http://{host}:{port}/get_server_info", timeout=10)
        if r.status_code != 200:
            return None
        info = r.json()
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


def _format_metric(value) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _print_acceptance_summary(label: str, results) -> None:
    rates = [x.get("spec_accept_rate") for x in results]
    rates = [float(x) for x in rates if x is not None]
    lengths = [x.get("spec_accept_length") for x in results]
    lengths = [float(x) for x in lengths if x is not None]
    if not rates and not lengths:
        print(f"acceptance: {label}: server did not return per-request MTP metrics")
        return

    parts = [f"acceptance: {label}: requests={max(len(rates), len(lengths))}"]
    if rates:
        parts.append(
            f"rate mean/min/max={sum(rates) / len(rates):.3f}/"
            f"{min(rates):.3f}/{max(rates):.3f} zero={sum(x == 0 for x in rates)}"
        )
    if lengths:
        parts.append(
            f"len mean/min/max={sum(lengths) / len(lengths):.3f}/"
            f"{min(lengths):.3f}/{max(lengths):.3f}"
        )
    print(" ".join(parts))


def run(args: argparse.Namespace) -> None:
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
    else:
        prompts = default_prompts()

    if args.concurrency <= 0:
        sys.exit("ERROR: --concurrency must be positive")
    if args.repeat <= 0:
        sys.exit("ERROR: --repeat must be positive")

    prompt_jobs = [
        (prompt_idx, replica_idx, prompt)
        for prompt_idx, prompt in enumerate(prompts)
        for replica_idx in range(args.repeat)
    ]
    if not prompt_jobs:
        sys.exit("ERROR: no prompts to send")

    _wait_for_server(args.host, args.port, args.wait)

    chunk_size = _get_chunk_size(args.host, args.port)
    print(f"server (per-DP) chunked_prefill_size = {chunk_size}")
    # NOTE: server_args enforces chunked_prefill_size % page_size == 0 (and under
    # DP-attention it is first divided by dp_size). So the per-DP chunk is ALWAYS
    # a multiple of page_size (128) -> chunk boundaries are always 128-aligned ->
    # c128 blocks never straddle a boundary, so c128 cross-chunk gather never
    # fires (it is clean by construction). The only cross-chunk path that fires
    # is c4 overlap (reads the previous 4-block across the boundary), which this
    # test does exercise. To stress MORE c4 boundary crossings, use the smallest
    # legal chunk: --chunked-prefill-size = page_size * dp_size (per-DP = 128).

    url = f"http://{args.host}:{args.port}/generate"

    def capture_one(job):
        i, (prompt_idx, replica_idx, p) = job
        body = {
            "text": p,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": args.max_new_tokens,
            },
            # return_logprob gives us the exact output token ids (stricter than
            # comparing detokenized text).
            "return_logprob": True,
            "logprob_start_len": 0,
            "stream": False,
        }
        resp = requests.post(url, json=body, timeout=args.timeout)
        resp.raise_for_status()
        d = resp.json()
        meta = d.get("meta_info", {}) or {}
        otl = (
            meta.get("output_token_logprobs") or []
        )  # [[logprob, token_id, text], ...]
        token_ids = [t[1] for t in otl]
        ptok = meta.get("prompt_tokens")
        chunked = bool(
            chunk_size is not None and chunk_size > 0 and (ptok or 0) > chunk_size
        )
        return {
            "idx": i,
            "prompt_idx": prompt_idx,
            "replica_idx": replica_idx,
            "prompt": p,
            "text": d.get("text", ""),
            "token_ids": token_ids,
            "prompt_tokens": ptok,
            "completion_tokens": meta.get("completion_tokens"),
            "chunk_size": chunk_size,
            "chunked": chunked,
            "concurrency": args.concurrency,
            "spec_accept_length": meta.get("spec_accept_length"),
            "spec_accept_rate": meta.get("spec_accept_rate"),
            "spec_verify_ct": meta.get("spec_verify_ct"),
            "spec_num_correct_drafts": meta.get("spec_num_correct_drafts"),
            "spec_num_proposed_drafts": meta.get("spec_num_proposed_drafts"),
            "spec_correct_drafts_histogram": meta.get("spec_correct_drafts_histogram"),
        }

    indexed_jobs = list(enumerate(prompt_jobs))
    workers = min(args.concurrency, len(indexed_jobs))
    if workers == 1:
        results = [capture_one(job) for job in indexed_jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(capture_one, indexed_jobs))

    for result in results:
        print(
            f"[{result['idx']}] prompt={result['prompt_idx']} "
            f"replica={result['replica_idx']} "
            f"prompt_tokens={result['prompt_tokens']} "
            f"completion_tokens={result['completion_tokens']} "
            f"out_token_ids={len(result['token_ids'])} "
            f"accept_len={_format_metric(result['spec_accept_length'])} "
            f"accept_rate={_format_metric(result['spec_accept_rate'])} "
            f"{'CHUNKED' if result['chunked'] else 'not-chunked'}"
        )

    _print_acceptance_summary(args.out, results)

    with open(args.out, "w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} results -> {args.out}")


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _match_prefix_len(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare(args: argparse.Namespace) -> None:
    A = _load(args.a)
    B = _load(args.b)
    if len(A) != len(B):
        print(f"WARN: different prompt counts: {args.a}={len(A)} {args.b}={len(B)}")
    n = min(len(A), len(B))

    # Coverage: how many prompts in the chunked capture actually chunked. A PASS
    # over prompts that never chunked is trivial and validates nothing.
    cs_a = next(
        (x.get("chunk_size") for x in A if x.get("chunk_size") is not None), None
    )
    chunked_n = sum(1 for x in A if x.get("chunked"))
    print(
        f"coverage: {args.a} chunk_size={cs_a}, {chunked_n}/{len(A)} prompts actually chunked"
    )
    if chunked_n == 0:
        print(
            "  WARNING: NO prompt chunked in capture A -> a PASS here is TRIVIAL.\n"
            "  Relaunch the chunked server with a smaller --chunked-prefill-size "
            "(< your prompt lengths) and re-capture."
        )
    # NOTE: the per-DP chunk is always a multiple of page_size (server_args
    # enforces it), so c4 overlap is the only cross-chunk path that fires; c128
    # cross-chunk is clean by construction (boundaries are 128-aligned). No need
    # for a non-128 chunk size (it is rejected at startup anyway).

    _print_acceptance_summary(args.a, A)
    _print_acceptance_summary(args.b, B)

    all_ok = True
    for i in range(n):
        a, b = A[i], B[i]
        # prefer token-id parity; fall back to text if logprobs were off
        ta, tb = a.get("token_ids") or [], b.get("token_ids") or []
        if ta and tb:
            la, lb = len(ta), len(tb)
            m = _match_prefix_len(ta, tb)
            exact = ta == tb
            kind = "tokens"
        else:
            sa, sb = a.get("text", ""), b.get("text", "")
            la, lb = len(sa), len(sb)
            m = _match_prefix_len(list(sa), list(sb))  # char prefix
            exact = sa == sb
            kind = "chars"

        ptok = a.get("prompt_tokens")
        accept = (
            "accept_len(A/B)="
            f"{_format_metric(a.get('spec_accept_length'))}/"
            f"{_format_metric(b.get('spec_accept_length'))} "
            "accept_rate(A/B)="
            f"{_format_metric(a.get('spec_accept_rate'))}/"
            f"{_format_metric(b.get('spec_accept_rate'))}"
        )
        if exact:
            print(
                f"[{i}] OK  ({kind}) prompt_tokens={ptok} len={la} "
                f"fully identical {accept}"
            )
        else:
            all_ok = False
            print(
                f"[{i}] DIVERGE ({kind}) prompt_tokens={ptok} "
                f"matched_prefix={m} lenA={la} lenB={lb} "
                f"-> first diff at {kind[:-1]} #{m} {accept}"
            )
            if kind == "tokens":
                lo = max(0, m - 2)
                print(f"      A[{lo}:{m+3}]={ta[lo:m+3]}")
                print(f"      B[{lo}:{m+3}]={tb[lo:m+3]}")

    print("=" * 48)
    print(
        "PARITY PASS — all prompts identical"
        if all_ok
        else "PARITY FAIL — see DIVERGE above"
    )
    print(
        "Tip: an EARLY divergence (matched_prefix small) is a chunked-path bug; "
        "a LATE one after a long identical prefix is likely fp nondeterminism "
        "(confirm by comparing two SAME-config runs)."
    )
    sys.exit(0 if all_ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="capture greedy outputs from a server")
    r.add_argument("--host", required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--out", required=True, help="output .jsonl path")
    r.add_argument("--max-new-tokens", type=int, default=256)
    r.add_argument("--prompts", default=None, help="optional file, one prompt per line")
    r.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="number of simultaneous requests",
    )
    r.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt N times (use one prompt + repeat=N for fixed-input concurrency)",
    )
    r.add_argument(
        "--timeout", type=float, default=3600.0, help="per-request timeout (s)"
    )
    r.add_argument(
        "--wait",
        type=float,
        default=1800.0,
        help="seconds to wait for the server to become ready before sending (0 = no wait)",
    )
    r.set_defaults(func=run)

    c = sub.add_parser("compare", help="diff two captures token-by-token")
    c.add_argument("--a", required=True, help="e.g. chunked.jsonl")
    c.add_argument("--b", required=True, help="e.g. baseline.jsonl")
    c.set_defaults(func=compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
