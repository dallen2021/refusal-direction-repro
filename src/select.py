"""Stage 2: select the best (layer, position) direction by ablation effect.

For each candidate direction r[L, p] from extract.py, install ablation hooks
that project r out of the residual stream at every block, then generate on the
harmful validation split. The candidate that drives the refusal rate lowest
wins. Ties are broken toward middle-to-late layers, where the paper finds the
strongest refusal-mediating directions.
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


def evaluate_candidate(
    lm,
    direction: torch.Tensor,
    val_prompts_chat: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> tuple[float, list[str]]:
    """Returns (refusal_rate, completions) under ablation of `direction`."""
    completions: list[str] = []
    with ablate_direction(lm.model, direction):
        for start in range(0, len(val_prompts_chat), batch_size):
            batch = val_prompts_chat[start:start + batch_size]
            completions.extend(generate_batch(lm, batch, max_new_tokens=max_new_tokens))
    refusals = sum(1 for c in completions if is_refusal(c))
    return refusals / len(completions), completions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", default=str(RESULTS_DIR / "directions.pt"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--layer-min-frac", type=float, default=0.4,
                    help="skip the earliest fraction of layers (paper finds them noisy)")
    ap.add_argument("--out", default=str(RESULTS_DIR / "selection.json"))
    args = ap.parse_args()

    state = torch.load(args.directions, map_location="cpu", weights_only=False)
    model_id = state["model_id"]
    positions: tuple[int, ...] = tuple(state["positions"])
    directions: torch.Tensor = state["directions"]  # (L, P, D)
    n_layers, n_pos, _ = directions.shape

    lm = load_model(model_id)
    val = load_split("harmful_val")
    val_chat = [format_chat(lm.tokenizer, x) for x in val]

    layer_lo = int(args.layer_min_frac * n_layers)

    best = {"layer": -1, "position": 0, "refusal_rate": 1.01}
    grid: list[dict] = []

    for L in tqdm(range(layer_lo, n_layers), desc="layer"):
        for pi, p in enumerate(positions):
            d = directions[L, pi].to(lm.device, dtype=lm.model.dtype)
            rate, _ = evaluate_candidate(lm, d, val_chat, args.batch_size, args.max_new_tokens)
            grid.append({"layer": L, "position": int(p), "refusal_rate": rate})
            tqdm.write(f"  L={L:>2} p={int(p):>3}  refusal_rate={rate:.3f}")
            if rate < best["refusal_rate"]:
                best = {"layer": L, "position": int(p), "refusal_rate": rate}

    out = {
        "model_id": model_id,
        "n_val": len(val),
        "best": best,
        "grid": grid,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nbest: layer={best['layer']} position={best['position']} refusal_rate={best['refusal_rate']:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
