"""Generate exact input-id datasets for prefix-cache benchmarks."""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def _positive_int(value: str) -> int:
    value_int = int(value)
    if value_int <= 0:
        raise argparse.ArgumentTypeError(f"expected positive int, got {value}")
    return value_int


def _load_valid_token_ids(model_path: str, seed: int) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    valid_ids = sorted(
        {
            int(token_id)
            for token_id in tokenizer.get_vocab().values()
            if isinstance(token_id, int) and int(token_id) not in special_ids
        }
    )
    if not valid_ids:
        raise ValueError("tokenizer has no non-special token ids")

    # Shuffle once so generated prompts do not concentrate on low token ids.
    rng = random.Random(seed)
    rng.shuffle(valid_ids)
    return valid_ids


def _sample_ids(valid_ids: list[int], length: int, rng: random.Random) -> list[int]:
    return [valid_ids[rng.randrange(len(valid_ids))] for _ in range(length)]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate warmup/formal JSONL files whose formal requests share an "
            "exact token-id prefix with the warmup requests."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix-len", type=_positive_int, default=990080)
    parser.add_argument("--tail-len", type=_positive_int, default=9920)
    parser.add_argument("--num-prompts", type=_positive_int, default=2)
    parser.add_argument("--dp-size", type=_positive_int, default=2)
    parser.add_argument("--output-len", type=_positive_int, default=1024)
    parser.add_argument("--warmup-output-len", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_ids = _load_valid_token_ids(args.model_path, args.seed)
    rng = random.Random(args.seed)

    prefix_ids = _sample_ids(valid_ids, args.prefix_len, rng)
    tails = [
        _sample_ids(valid_ids, args.tail_len, rng) for _ in range(args.num_prompts)
    ]

    for dp_rank in range(args.dp_size):
        _write_jsonl(
            out_dir / f"warmup_dp{dp_rank}.jsonl",
            [
                {
                    "input_ids": prefix_ids,
                    "output_len": args.warmup_output_len,
                    "routed_dp_rank": dp_rank,
                }
            ],
        )

    formal_records = []
    for index, tail_ids in enumerate(tails):
        formal_records.append(
            {
                "input_ids": prefix_ids + tail_ids,
                "output_len": args.output_len,
                "routed_dp_rank": index % args.dp_size,
            }
        )
    _write_jsonl(out_dir / "formal.jsonl", formal_records)

    manifest = {
        "prefix_len": args.prefix_len,
        "tail_len": args.tail_len,
        "prompt_len": args.prefix_len + args.tail_len,
        "num_prompts": args.num_prompts,
        "dp_size": args.dp_size,
        "target_cache_hit_rate": args.prefix_len / (args.prefix_len + args.tail_len),
        "files": {
            "formal": str(out_dir / "formal.jsonl"),
            "warmup": [
                str(out_dir / f"warmup_dp{dp_rank}.jsonl")
                for dp_rank in range(args.dp_size)
            ],
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
