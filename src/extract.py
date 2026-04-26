"""Stage 1: extract candidate refusal directions.

For each transformer block L and each prompt-token position p in POSITIONS,
compute the unit-norm mean-difference direction between harmful and harmless
class activations:

    r[L, p] = normalize( mean(act_harmful[L, p]) - mean(act_harmless[L, p]) )

Saves a tensor of shape (n_layers, len(POSITIONS), d_model) to
    results/directions.pt

along with a small metadata dict (model id, positions, layer count).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .build_data import load_split
from .lib import LoadedModel, capture_residuals, format_chat, load_model


# Last 5 prompt-token positions, per Arditi et al.'s "last instruction position
# and recent neighbors" sweep.
POSITIONS: tuple[int, ...] = (-5, -4, -3, -2, -1)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@torch.no_grad()
def collect_class_activations(
    lm: LoadedModel,
    instructions: list[str],
    positions: tuple[int, ...],
    batch_size: int = 8,
) -> torch.Tensor:
    """Run forward passes and return activations of shape
    (n_layers, len(positions), d_model) — averaged across `instructions`."""
    n_layers = lm.n_layers
    sums = torch.zeros(n_layers, len(positions), lm.d_model, dtype=torch.float32)
    count = 0

    prompts = [format_chat(lm.tokenizer, x) for x in instructions]

    for start in tqdm(range(0, len(prompts), batch_size), desc="forward"):
        batch = prompts[start:start + batch_size]
        enc = lm.tokenizer(batch, return_tensors="pt", padding=True).to(lm.device)
        with capture_residuals(lm.model, positions=positions) as caps:
            lm.model(**enc)
        for layer_idx, layer_caps in enumerate(caps):
            # layer_caps is a list of length 1 (one forward pass): shape
            # (batch, len(positions), d_model).
            acts = layer_caps[0]  # cpu, float32
            sums[layer_idx] += acts.sum(dim=0)
        count += len(batch)

    return sums / count


def compute_directions(
    harmful_means: torch.Tensor,
    harmless_means: torch.Tensor,
) -> torch.Tensor:
    diff = harmful_means - harmless_means
    norms = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return diff / norms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=str(RESULTS_DIR / "directions.pt"))
    args = ap.parse_args()

    lm = load_model(args.model)
    print(f"loaded {args.model}: layers={lm.n_layers}, d_model={lm.d_model}")

    harmful = load_split("harmful_train")
    harmless = load_split("harmless_train")

    print(f"collecting activations on {len(harmful)} harmful + {len(harmless)} harmless")
    h_means = collect_class_activations(lm, harmful, POSITIONS, batch_size=args.batch_size)
    a_means = collect_class_activations(lm, harmless, POSITIONS, batch_size=args.batch_size)

    directions = compute_directions(h_means, a_means)  # (n_layers, n_pos, d_model)
    print(f"directions shape: {tuple(directions.shape)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": args.model,
            "positions": POSITIONS,
            "n_layers": lm.n_layers,
            "d_model": lm.d_model,
            "directions": directions,  # float32 cpu
        },
        out_path,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
