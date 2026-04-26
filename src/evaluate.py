"""Stage 3: final evaluation — baseline vs ablated refusal rate on test split.

Loads the winning (layer, position) from selection.json and the corresponding
direction tensor from directions.pt. Generates short continuations (~16 tokens,
just enough to classify refusal vs. non-refusal) on harmful_test in two
conditions (no ablation, ablation) and writes ONLY the boolean refusal flag
per prompt plus summary statistics. Full text completions are deliberately
not persisted: the claim being measured is behavioral (refusal rate delta),
not capabilities, so storing payload text is unnecessary and irresponsible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .build_data import load_split
from .lib import ablate_direction, format_chat, generate_batch, is_refusal, load_model


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def refusal_flags(lm, prompts_chat: list[str], batch_size: int, max_new_tokens: int) -> list[bool]:
    flags: list[bool] = []
    for start in tqdm(range(0, len(prompts_chat), batch_size), desc="gen"):
        outs = generate_batch(lm, prompts_chat[start:start + batch_size], max_new_tokens=max_new_tokens)
        # Classify each output for refusal, then drop the text.
        flags.extend(is_refusal(o) for o in outs)
        del outs
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", default=str(RESULTS_DIR / "directions.pt"))
    ap.add_argument("--selection", default=str(RESULTS_DIR / "selection.json"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Generation length: keep small. The refusal classifier only inspects "
        "the first line, so 16 tokens is sufficient and minimizes the volume of "
        "post-ablation content produced in memory before classification.",
    )
    ap.add_argument("--out", default=str(RESULTS_DIR / "evaluation.json"))
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

    lm = load_model(model_id)
    test = load_split("harmful_test")
    test_chat = [format_chat(lm.tokenizer, x) for x in test]

    print(f"baseline run on {len(test)} harmful test prompts")
    base_flags = refusal_flags(lm, test_chat, args.batch_size, args.max_new_tokens)
    base_rate = sum(base_flags) / len(base_flags)

    print(f"ablated run (layer={best_layer}, position={best_pos})")
    d = direction.to(lm.device, dtype=lm.model.dtype)
    with ablate_direction(lm.model, d):
        abl_flags = refusal_flags(lm, test_chat, args.batch_size, args.max_new_tokens)
    abl_rate = sum(abl_flags) / len(abl_flags)

    summary = {
        "model_id": model_id,
        "n_test": len(test),
        "best_layer": best_layer,
        "best_position": best_pos,
        "max_new_tokens": args.max_new_tokens,
        "baseline_refusal_rate": base_rate,
        "ablated_refusal_rate": abl_rate,
        "delta": base_rate - abl_rate,
    }
    # Per-prompt: record the prompt instruction (already public in AdvBench)
    # and only the boolean refusal flag for each condition. No completion text
    # is persisted in either condition.
    per_prompt = [
        {"instruction": inst, "baseline_refused": bool(b), "ablated_refused": bool(a)}
        for inst, b, a in zip(test, base_flags, abl_flags)
    ]
    Path(args.out).write_text(
        json.dumps({"summary": summary, "per_prompt": per_prompt}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
