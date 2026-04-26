"""Stage 4: capability sanity check — does ablation tank general capability?

For each MMLU question, build a chat-formatted prompt that ends positioned
for the assistant to emit a single letter (A/B/C/D). Forward pass once
(no generation). Compare the logits at the four candidate letter tokens;
argmax is the model's prediction.

Run twice: once with no hooks (baseline), once with `ablate_direction(r*)`
installed. Report MMLU accuracy in both conditions. The headline number is
the delta — if ablation collapsed capability, accuracy would crash; if the
direction is narrowly localized to refusal, accuracy stays close to baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .lib import ablate_direction, format_chat, load_model


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LETTERS = ("A", "B", "C", "D")


def build_prompt(q: dict) -> str:
    a, b, c, d = q["choices"]
    return (
        "The following is a multiple choice question.\n\n"
        f"Question: {q['question']}\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n\n"
        "Answer with a single letter: A, B, C, or D."
    )


def get_letter_token_ids(tokenizer) -> dict[str, int]:
    """Return {letter: token_id} for A/B/C/D as the model would emit them at
    the start of an assistant turn (no leading space)."""
    out: dict[str, int] = {}
    for L in LETTERS:
        ids = tokenizer.encode(L, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"letter {L!r} encodes to {len(ids)} tokens: {ids}")
        out[L] = ids[0]
    return out


@torch.no_grad()
def score_questions(lm, questions: list[dict], letter_ids: dict[str, int], batch_size: int) -> tuple[float, list[bool]]:
    correct: list[bool] = []
    letter_id_list = [letter_ids[L] for L in LETTERS]

    prompts = [format_chat(lm.tokenizer, build_prompt(q)) for q in questions]
    golds = [q["answer"] for q in questions]
    gold_idx = [LETTERS.index(g) for g in golds]

    for start in tqdm(range(0, len(prompts), batch_size), desc="mmlu"):
        batch_prompts = prompts[start:start + batch_size]
        enc = lm.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(lm.device)
        logits = lm.model(**enc).logits  # (B, T, V)
        # Last token logits per row (left-padded, so just position -1).
        last = logits[:, -1, :]  # (B, V)
        # Restrict to the four letter tokens.
        letter_logits = last[:, letter_id_list]  # (B, 4)
        preds = letter_logits.argmax(dim=-1).tolist()
        for j, p in enumerate(preds):
            correct.append(p == gold_idx[start + j])

    acc = sum(correct) / len(correct)
    return acc, correct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--mmlu", default=str(DATA_DIR / "mmlu_test.jsonl"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="cap number of questions for a fast run")
    ap.add_argument("--out", default=str(RESULTS_DIR / "capability.json"))
    args = ap.parse_args()

    sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    state = torch.load(args.directions, map_location="cpu", weights_only=False)
    model_id = state["model_id"]
    positions = list(state["positions"])
    directions: torch.Tensor = state["directions"]
    pi = positions.index(sel["best"]["position"])
    direction = directions[sel["best"]["layer"], pi]

    questions: list[dict] = []
    with Path(args.mmlu).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit:
        questions = questions[:args.limit]
    print(f"scoring {len(questions)} MMLU questions on {model_id}")

    lm = load_model(model_id)
    letter_ids = get_letter_token_ids(lm.tokenizer)
    print(f"letter token ids: {letter_ids}")

    print("baseline...")
    base_acc, base_correct = score_questions(lm, questions, letter_ids, args.batch_size)

    print("ablated...")
    d = direction.to(lm.device, dtype=lm.model.dtype)
    with ablate_direction(lm.model, d):
        abl_acc, abl_correct = score_questions(lm, questions, letter_ids, args.batch_size)

    # Per-subject breakdown.
    by_subject: dict[str, dict] = {}
    for q, b, a in zip(questions, base_correct, abl_correct):
        s = q["subject"]
        bs = by_subject.setdefault(s, {"n": 0, "base": 0, "abl": 0})
        bs["n"] += 1
        bs["base"] += int(b)
        bs["abl"] += int(a)
    for s, bs in by_subject.items():
        bs["base_acc"] = bs["base"] / bs["n"]
        bs["abl_acc"] = bs["abl"] / bs["n"]
        bs["delta"] = bs["base_acc"] - bs["abl_acc"]

    summary = {
        "model_id": model_id,
        "n_questions": len(questions),
        "best_layer": sel["best"]["layer"],
        "best_position": sel["best"]["position"],
        "baseline_accuracy": base_acc,
        "ablated_accuracy": abl_acc,
        "delta_pp": (base_acc - abl_acc) * 100,
        "by_subject": by_subject,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
