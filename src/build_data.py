"""Download and split harmful (AdvBench) and harmless (Alpaca) instructions.

Writes JSONL files under data/:
    data/harmful_train.jsonl    used to compute the direction
    data/harmful_val.jsonl      used to select layer/position
    data/harmful_test.jsonl     used for final refusal-rate eval
    data/harmless_train.jsonl   matched to harmful_train (direction class B)
    data/harmless_val.jsonl     used as capability sanity check (optional)

Each line is `{"instruction": "..."}`.
"""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from pathlib import Path


# Pulling raw files directly from the original repos avoids two issues:
#  1. The HF AdvBench mirrors are gated.
#  2. Importing `datasets` after `torch` segfaults on Python 3.13 + Windows
#     because pyarrow conflicts with torch's bundled MKL/OMP. Fetching with
#     urllib means we only depend on torch/transformers in the main pipeline.
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Sizes (small, faithful to the paper's setup).
N_TRAIN = 64
N_VAL = 32
N_TEST = 64


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main(seed: int = 0) -> None:
    rng = random.Random(seed)

    print("downloading AdvBench (~80 KB)...")
    raw = urllib.request.urlopen(ADVBENCH_URL).read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))
    harmful_prompts = [r["goal"] for r in reader if r.get("goal")]
    rng.shuffle(harmful_prompts)
    needed = N_TRAIN + N_VAL + N_TEST
    if len(harmful_prompts) < needed:
        raise RuntimeError(f"AdvBench has {len(harmful_prompts)}, need {needed}")
    h_train = harmful_prompts[:N_TRAIN]
    h_val = harmful_prompts[N_TRAIN:N_TRAIN + N_VAL]
    h_test = harmful_prompts[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]

    print("downloading Alpaca (~22 MB)...")
    alpaca = json.loads(urllib.request.urlopen(ALPACA_URL).read().decode("utf-8"))
    # Restrict to single-turn instructions with no required input context.
    candidates = [r["instruction"] for r in alpaca if not r.get("input")]
    rng.shuffle(candidates)
    a_train = candidates[:N_TRAIN]
    a_val = candidates[N_TRAIN:N_TRAIN + N_VAL]

    _write_jsonl(DATA_DIR / "harmful_train.jsonl", [{"instruction": x} for x in h_train])
    _write_jsonl(DATA_DIR / "harmful_val.jsonl", [{"instruction": x} for x in h_val])
    _write_jsonl(DATA_DIR / "harmful_test.jsonl", [{"instruction": x} for x in h_test])
    _write_jsonl(DATA_DIR / "harmless_train.jsonl", [{"instruction": x} for x in a_train])
    _write_jsonl(DATA_DIR / "harmless_val.jsonl", [{"instruction": x} for x in a_val])

    print(f"wrote {N_TRAIN}/{N_VAL}/{N_TEST} harmful and {N_TRAIN}/{N_VAL} harmless to {DATA_DIR}")


def load_split(name: str) -> list[str]:
    path = DATA_DIR / f"{name}.jsonl"
    return [json.loads(line)["instruction"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
