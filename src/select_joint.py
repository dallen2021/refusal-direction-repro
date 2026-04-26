"""Stage 2b: joint-criterion direction selection.

The pure refusal-rate sweep in select.py picks the (layer, position) that
maximally suppresses refusal. As the MMLU capability check shows, that
criterion can also chip at general capability (~6 pp on 7B). This script
re-scores the same candidate grid with a joint criterion:

    keep candidates with val_refusal_rate <= REFUSAL_TARGET, then pick the
    one that minimizes mean KL(baseline || ablated) of next-token logits on
    harmless prompts.

KL is a much cheaper capability proxy than MMLU — one forward pass per
candidate per harmless prompt, no generation. The hypothesis: candidates
that achieve near-zero refusal at the cost of low harmless-KL preserve
capability better than the refusal-only argmax.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .build_data import load_split
from .lib import ablate_direction, format_chat, load_model


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REFUSAL_TARGET = 0.05  # require <=5% val refusal to qualify as a viable candidate


@torch.no_grad()
def last_token_logprobs(lm, prompts_chat: list[str], batch_size: int) -> torch.Tensor:
    """Returns log-softmax logits at the last input position, shape (n_prompts, vocab)."""
    chunks: list[torch.Tensor] = []
    for start in range(0, len(prompts_chat), batch_size):
        batch = prompts_chat[start:start + batch_size]
        enc = lm.tokenizer(batch, return_tensors="pt", padding=True).to(lm.device)
        logits = lm.model(**enc).logits[:, -1, :]  # left-padded → last position is final input token
        chunks.append(F.log_softmax(logits.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", required=True)
    ap.add_argument("--selection", required=True, help="existing selection.json with refusal-rate grid")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--refusal-target", type=float, default=REFUSAL_TARGET)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    state = torch.load(args.directions, map_location="cpu", weights_only=False)
    model_id = state["model_id"]
    positions: list[int] = list(state["positions"])
    directions: torch.Tensor = state["directions"]

    qualifiers = [c for c in sel["grid"] if c["refusal_rate"] <= args.refusal_target]
    if not qualifiers:
        raise RuntimeError(
            f"no candidates achieve refusal_rate <= {args.refusal_target}; "
            f"loosen --refusal-target or re-run select.py"
        )
    print(f"{len(qualifiers)} of {len(sel['grid'])} candidates meet refusal_rate <= {args.refusal_target}")

    lm = load_model(model_id)
    harmless = load_split("harmless_val")
    harmless_chat = [format_chat(lm.tokenizer, x) for x in harmless]

    print(f"computing baseline logprobs on {len(harmless)} harmless prompts...")
    baseline_logp = last_token_logprobs(lm, harmless_chat, args.batch_size)

    rows: list[dict] = []
    for c in tqdm(qualifiers, desc="candidate"):
        L, p = c["layer"], c["position"]
        pi = positions.index(p)
        d = directions[L, pi].to(lm.device, dtype=lm.model.dtype)
        with ablate_direction(lm.model, d):
            ablated_logp = last_token_logprobs(lm, harmless_chat, args.batch_size)
        # KL(baseline || ablated) = sum_x P_base(x) * (logP_base(x) - logP_abl(x))
        p_base = baseline_logp.exp()
        kl = (p_base * (baseline_logp - ablated_logp)).sum(dim=-1).mean().item()
        rows.append({"layer": L, "position": p, "refusal_rate": c["refusal_rate"], "harmless_kl": kl})
        tqdm.write(f"  L={L:>2} p={p:>3}  refusal={c['refusal_rate']:.3f}  KL={kl:.4f}")

    rows.sort(key=lambda r: r["harmless_kl"])
    best = rows[0]

    out = {
        "model_id": model_id,
        "criterion": "min(harmless_kl) subject to refusal_rate <= refusal_target",
        "refusal_target": args.refusal_target,
        "n_candidates_considered": len(qualifiers),
        "best": {"layer": best["layer"], "position": best["position"], "refusal_rate": best["refusal_rate"], "harmless_kl": best["harmless_kl"]},
        "ranked": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\njoint-best: layer={best['layer']} position={best['position']} "
          f"refusal_rate={best['refusal_rate']:.3f} KL={best['harmless_kl']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
