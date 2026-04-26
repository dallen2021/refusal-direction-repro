"""Many-shot jailbreaking harness (Anthropic 2024).

Prepend N (user, assistant) pairs of harmless Q/A to the conversation history,
then ask the harmful query as the final user turn. The attack hypothesis from
Anil et al. 2024: the model treats the long fake history as evidence of its
own commitments and is increasingly willing to comply, with refusal rate
monotonically decreasing as N grows.

This harness extends the project's existing context-priming framing: the
single-turn-with-repeated-content result in `turn_mechanism.py` is essentially
the 1-shot variant. Many-shot scales the same intuition to longer histories.

Two measurements per (model, N):

1. **Refusal-direction projection** at the last user-prompt token under each
   N. Pure forward-pass; no generation.
2. **Refusal rate** under each N (16-token generation, classifier on the first
   line, only booleans persisted; no completion text written to disk).

Probe distribution is non-bio AdvBench (cyber/harassment/fraud), same as
turn_mechanism.py and validate_single_turn.py.

The fake (user, assistant) pairs are drawn from Stanford Alpaca's
instruction/output pairs — these are real benign Q/A so the conversation
history is plausible and topically coherent.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import torch
from tqdm import tqdm

from .crescendo import ESCALATION as CRESCENDO_TURNS
from .lib import (
    capture_residuals,
    generate_batch,
    is_refusal,
    load_model,
)
from .probe_templates import filter_probes


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ALPACA_PAIRS_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
ALPACA_PAIRS_PATH = DATA_DIR / "alpaca_pairs.json"


def fetch_alpaca_pairs(min_pairs: int = 1024) -> list[dict]:
    """Cache instruction/output pairs from Stanford Alpaca, single-turn only."""
    if ALPACA_PAIRS_PATH.exists():
        return json.loads(ALPACA_PAIRS_PATH.read_text(encoding="utf-8"))
    print(f"downloading Alpaca for many-shot context (~22 MB)...")
    raw = urllib.request.urlopen(ALPACA_PAIRS_URL).read().decode("utf-8")
    full = json.loads(raw)
    pairs = [
        {"instruction": r["instruction"], "output": r["output"]}
        for r in full
        if not r.get("input") and r.get("instruction") and r.get("output")
    ]
    if len(pairs) < min_pairs:
        raise RuntimeError(f"Alpaca yielded {len(pairs)} pairs; need >= {min_pairs}")
    ALPACA_PAIRS_PATH.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    print(f"  cached {len(pairs)} (instruction, output) pairs to {ALPACA_PAIRS_PATH}")
    return pairs


def build_manyshot_prompt(
    tokenizer,
    fake_history: list[dict],
    final_query: str,
) -> str:
    msgs: list[dict] = []
    for pair in fake_history:
        msgs.append({"role": "user", "content": pair["instruction"]})
        msgs.append({"role": "assistant", "content": pair["output"]})
    msgs.append({"role": "user", "content": final_query})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def measure_projections(
    lm,
    prompts: list[str],
    layer_idx: int,
    position: int,
    direction: torch.Tensor,
    batch_size: int,
) -> list[float]:
    out: list[float] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="forward", leave=False):
        batch = prompts[start:start + batch_size]
        enc = lm.tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(lm.device)
        with capture_residuals(lm.model, positions=(position,)) as caps:
            lm.model(**enc)
        acts = caps[layer_idx][0].squeeze(1)
        projs = (acts * direction).sum(dim=-1)
        out.extend(projs.tolist())
    return out


@torch.no_grad()
def measure_refusal(
    lm,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[bool]:
    flags: list[bool] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="gen", leave=False):
        outs = generate_batch(lm, prompts[start:start + batch_size], max_new_tokens=max_new_tokens)
        flags.extend(is_refusal(o) for o in outs)
        del outs
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--probes", default=str(DATA_DIR / "harmful_test.jsonl"))
    ap.add_argument("--max-probes", type=int, default=24,
                    help="probes per N (each is one forward pass; keep small for long-N runs)")
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 4, 16, 64],
                    help="values of N (number of fake (user, assistant) pairs prepended)")
    ap.add_argument("--final-wrapper", choices=["bare", "T2"], default="bare",
                    help="bare: final user is the raw query; T2: wrap in academic threat-model frame")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="long contexts need a small batch")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--measure", choices=["projection", "refusal", "both"], default="both")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    state = torch.load(args.directions, map_location="cpu", weights_only=False)
    model_id = state["model_id"]
    positions = list(state["positions"])
    directions: torch.Tensor = state["directions"]
    best_layer = sel["best"]["layer"]
    best_pos = sel["best"]["position"]
    pi = positions.index(best_pos)
    direction = directions[best_layer, pi]

    probes_full = [json.loads(line)["instruction"] for line in Path(args.probes).read_text(encoding="utf-8").splitlines() if line.strip()]
    probes, _ = filter_probes(probes_full)
    probes = probes[:args.max_probes]
    print(f"using {len(probes)} non-bio probes; shots={args.shots}; final_wrapper={args.final_wrapper}")

    pairs_pool = fetch_alpaca_pairs(min_pairs=max(args.shots) + 32)
    rng = random.Random(args.seed)

    lm = load_model(model_id)

    T2 = CRESCENDO_TURNS[1]

    def shape_query(q: str) -> str:
        return q if args.final_wrapper == "bare" else T2.format(query=q)

    rows: list[dict] = []
    for n in args.shots:
        # Sample one fixed fake history to share across all probes for this N
        # (removes prompt-history variance from the projection measurement).
        fake_history = rng.sample(pairs_pool, n) if n > 0 else []
        prompts = [build_manyshot_prompt(lm.tokenizer, fake_history, shape_query(q)) for q in probes]

        record: dict = {"n_shots": n, "n_probes": len(prompts)}

        if args.measure in ("projection", "both"):
            print(f"\n--- N={n} projections ---")
            projs = measure_projections(lm, prompts, best_layer, best_pos, direction, args.batch_size)
            t = torch.tensor(projs)
            record["mean_projection"] = t.mean().item()
            record["median_projection"] = t.median().item()
            record["std_projection"] = t.std().item()
            print(f"  N={n:>3}  mean_proj={record['mean_projection']:>+8.3f}  median={record['median_projection']:>+8.3f}")

        if args.measure in ("refusal", "both"):
            print(f"--- N={n} refusal generation ---")
            flags = measure_refusal(lm, prompts, args.batch_size, args.max_new_tokens)
            record["refusal_rate"] = sum(flags) / len(flags)
            print(f"  N={n:>3}  refusal_rate={record['refusal_rate']:.3f}")

        rows.append(record)

    out = {
        "model_id": model_id,
        "best_layer": best_layer,
        "best_position": best_pos,
        "n_probes": len(probes),
        "shots": args.shots,
        "final_wrapper": args.final_wrapper,
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== many-shot summary ===")
    print(f"  {'N':>4}  {'mean_proj':>10}  {'refusal_rate':>14}")
    for r in rows:
        proj = f"{r.get('mean_projection', float('nan')):>+10.3f}" if "mean_projection" in r else "         -"
        refu = f"{r.get('refusal_rate', float('nan')):.3f}" if "refusal_rate" in r else "    -"
        print(f"  {r['n_shots']:>4}  {proj}  {refu:>14}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
