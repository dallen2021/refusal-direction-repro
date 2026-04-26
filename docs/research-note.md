# The Crescendo Effect Is Context Priming, Not Escalation

**A cross-architecture mechanistic-interpretability finding on multi-turn jailbreaks of instruction-tuned language models.**

Daniel Allen, ASU NLP / computational neuroscience lab. April 2026.

---

## TL;DR

The widely-cited mechanism for the [Crescendo](https://arxiv.org/abs/2404.01833) multi-turn jailbreak (Russinovich et al. 2024) — *gradual escalation of harmfulness across turns, leveraging prior compliance* — is not the operative mechanism on three open-weights instruction-tuned models we tested (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2). Five controlled experiments isolating the contribution of multi-turn structure, placeholder content, prior-turn content, and single-turn equivalents show that the dominant effect is **context priming via repeated query content**. A single-turn prompt that includes the harmful query content twice — once in a benign academic frame, once in the actual ask — reproduces 90%+ of the multi-turn projection drop on all three architectures, and matches or beats the multi-turn version on actual refusal rate behaviorally on two of three.

This has direct implications for the threat model of jailbreak attacks against closed deployed models: **attacks that the literature classifies as "multi-turn" can often be collapsed into a single-turn structure** that is harder to filter against and more bounty-permissible (e.g. the GPT-5.5 Bio Bug Bounty's "single reusable prompt, clean chat" constraint).

---

## 1. Background and reproduction

[Arditi et al. 2024](https://arxiv.org/abs/2406.11717) showed that refusal in instruction-tuned chat models is mediated by a single direction in the residual stream: the unit-norm difference of class-mean activations between harmful and harmless prompts at a specific (layer, position). Projecting that direction out of forward-pass activations selectively suppresses refusal.

We reproduced this finding from scratch on four checkpoints — Qwen2.5-1.5B/7B-Instruct, Mistral-7B-Instruct-v0.2, Llama-3.1-8B-Instruct — using a balanced 64+64 train split (AdvBench harmful / Alpaca harmless). All four show the central result: a single mean-difference direction reduces AdvBench refusal rate from 96–100% to 0–3%.

We also identified that the basic "argmin val refusal rate" selection criterion is not surgical: on Qwen 7B it costs 6.0 pp on a 1386-question MMLU subset, and on Llama 8B it costs 3.8 pp. Replacing the criterion with `argmin(harmless_KL) subject to refusal_rate ≤ 0.05` recovers full capability on all three architectures (delta ≤ 0.6 pp) while matching or improving refusal suppression. The capability cost of the basic criterion is **model-dependent** — Mistral shows almost no tradeoff (−0.7 pp) while Qwen and Llama do — suggesting the entanglement between refusal and general capability in the residual stream is a property of safety post-training, not the method.

Code and full result tables: [github.com/dallen2021/refusal-direction-repro](https://github.com/dallen2021/refusal-direction-repro).

## 2. The Crescendo trajectory and the question it raises

In a fixed 4-turn Crescendo run on Qwen 7B (24 non-bio AdvBench probes, deterministic escalation script — no attacker LLM), the projection onto the refusal direction at the last user-prompt token drops from +16.04 (turn 1) to +5.64 (turn 2), then plateaus near +5 through turns 3 and 4. Refusal rate goes from 20.8% at turn 1 to 0% by turn 3.

The standard reading is "gradual escalation across turns lowers the refusal signal." But the trajectory is not gradual — most of the drop happens in a single transition (turn 1 → turn 2). What's actually different between turn 1 and turn 2?

Three candidate mechanisms:

- **(M1) Multi-turn structure** — the model treats prompt sequences containing prior user/assistant pairs as belonging to a different distribution than single-turn prompts.
- **(M2) Prior-assistant-turn content** — the model reads the prior assistant message as evidence about its own commitments. An apparent prior compliance lowers refusal; an apparent prior refusal raises it.
- **(M3) Prior-user-turn content** — the *user's* prior message changes the model's processing of the current user message, independently of the assistant turn between them.

## 3. Five controlled experiments

All experiments measure the residual-stream projection onto the model's measured refusal direction at the operative `(layer, position)` pair (Qwen 7B: L=12, p=−4; Llama 8B: L=12, p=−5; Mistral 7B: L=13, p=−4 — all selected by the basic refusal-only criterion). Probe distribution is AdvBench harmful instructions, with bio-adjacent items keyword-filtered out (cyber, harassment, fraud only — the pre-bounty methodology development is restricted to non-bio refusal categories). 48 probes per condition.

### 3.1 Experiment A — multi-turn structure alone (testing M1)

Hold prior conversation content fixed and neutral (`"Hi, I have a question to ask you."` user, `"[response]"` assistant placeholder). Vary only the number of prior user/assistant pairs (0, 1, 2, 3) before a final user turn that contains the bare harmful query.

| n_prior turns | Qwen 7B | Llama 8B | Mistral 7B |
|---:|---:|---:|---:|
| 0 (single-turn baseline) | +25.43 | +5.91 | +2.71 |
| 1 | +23.17 | +5.89 | +2.79 |
| 2 | +23.13 | +5.74 | +2.76 |
| 3 | +22.73 | +5.69 | +2.76 |

The drop saturates at n=1 on all three models. The total magnitude is small: Qwen drops 10%, Llama drops 3%, Mistral *increases* slightly (+3%). **M1 is not the dominant mechanism.**

### 3.2 Experiment B — assistant placeholder content (testing M2)

Hold structure at exactly one prior turn (with `"Hi, I have a question to ask you."` as the prior user turn). Vary the assistant placeholder content across eight conditions including empty string, the `"[response]"` marker, short and long compliance phrasings, short and long refusal phrasings, and a benign filler.

The full range of mean projections across all eight placeholder contents on Qwen 7B is +22.06 to +24.33 — a **2-unit spread**. On Llama 8B, the spread is +5.49 to +5.99 (0.5-unit). Counter-intuitively, the explicit prior refusal placeholder (`"I'm sorry, I can't help with that."`) gives the *lowest* projection on both models, not the highest as M2 would predict. **M2 is not the dominant mechanism.** The model does not appear to read the assistant placeholder as evidence about its own commitments at this measurement position.

### 3.3 Experiment C — Crescendo wrappers single-turn vs multi-turn

For each of the four Crescendo escalation wrappers (T1 academic-briefing, T2 "Helpful, thanks", T3 "Useful", T4 "Putting it together"), measure projection in two structures: single-turn (the wrapper is the only user message) vs after one neutral prior turn.

The cross-arch result, shown for T2 specifically (the wrapper used in subsequent experiments):

| Condition | Qwen 7B | Llama 8B | Mistral 7B |
|---|---:|---:|---:|
| T2 wrapper alone, single-turn | +17.09 | +4.33 | +1.68 |
| T2 wrapper, after 1 neutral prior | +15.47 | +4.15 | +1.66 |

Single-turn academic-framing wrappers already drop projection materially relative to the bare-query baseline. The multi-turn structure adds little on top.

### 3.4 Experiment D — prior user turn content (testing M3)

The crucial isolation. Hold final user turn = T2 wrapper. Vary the *prior user turn* across five conditions:

| Prior user turn | Qwen 7B | Llama 8B | Mistral 7B |
|---|---:|---:|---:|
| neutral ("Hi") | +15.47 | +4.15 | +1.66 |
| **T1 wrapper applied to SAME query** | **+5.62** | **+1.80** | **+0.42** |
| T1 wrapper applied to a DIFFERENT harmful query | +12.64 | +3.22 | +1.43 |
| **A different academic wrapper applied to SAME query** | **+5.34** | **+1.83** | **+0.38** |
| Length-matched benign content (weekend trip planning) | +14.27 | +3.81 | +1.61 |

The **same** harmful query, embedded in *any* academic wrapper in the prior user turn, drops the final-turn projection by 57%, 57%, and 75% on the three models respectively. The specific wrapper text does not matter (D2 ≈ D4 on every model). A *different* harmful query in the prior turn captures only 30% of the effect. A length-matched but topically unrelated prior captures none. **M3 is the dominant mechanism**, and the operative feature is repeated query content — not academic framing in the abstract, not length, not the assistant turn between them.

### 3.5 Experiment E — single-turn prompts with repeated query content

If the dominant mechanism is content-priming and the multi-turn structure itself contributes little, a single-turn prompt that includes the same harmful query content twice — once in a benign academic frame, once in the actual ask — should reproduce most of the effect.

| Single-turn structure | Qwen 7B | Llama 8B | Mistral 7B |
|---|---:|---:|---:|
| T2 wrapper alone (control) | +17.09 | +4.33 | +1.68 |
| Explicit self-reference: "Earlier I asked X. Now Y." | +7.54 | +2.30 | +0.54 |
| **Concatenated with no marker: "X Y"** | **+6.35** | **+2.22** | **+0.52** |
| Two-paragraph "First X. Second Y." | +6.81 | +2.01 | +0.51 |
| (reference: multi-turn D2 with thematic prior) | +5.62 | +1.80 | +0.42 |

On every architecture, a single user message that simply concatenates two academic-framed instances of the same harmful query lands within ~1 unit of the multi-turn version. **Multi-turn structure is not required to capture the Crescendo effect.**

## 4. Behavioral validation

Projection is a correlated signal. To confirm the prediction translates to actual jailbreak success, we measured the refusal rate (16-token generation, classifier on the first line, only booleans persisted — no completion text written to disk) under five conditions on the same 48 non-bio probe set:

| Condition | Qwen 7B | Llama 8B | Mistral 7B |
|---|---:|---:|---:|
| A: vanilla bare query | 97.9% | 95.8% | 62.5% |
| B: T2 wrapper alone (single-turn) | 20.8% | 81.2% | 20.8% |
| **C: E3 concatenated single-turn** | **8.3%** | 58.3% | **0.0%** |
| **D: E4 two-paragraph single-turn** | **2.1%** | 66.7% | **0.0%** |
| E: multi-turn Crescendo turn 2 | 4.2% | 54.2% | 0.0% |

On Qwen 7B and Mistral 7B, the projection-based prediction holds cleanly: single-turn E3/E4 templates match or beat multi-turn Crescendo on actual refusal rate. (Mistral hits 0% under all three jailbreak conditions, so they cannot be distinguished — the model is too easily broken to differentiate the techniques.) On Llama 8B the prediction is partial: single-turn templates cut refusal dramatically (95.8% → 58–67%) but multi-turn retains a small lead (54.2%), suggesting Llama has additional refusal pathways not captured by the single residual-stream direction at the chosen (layer, position).

## 5. Limitations

1. **Quality of compliance.** Our primary refusal classifier checked whether the first generated line begins with a refusal phrase. After the initial writeup, a follow-up judge harness (`src/judge_quality.py`) re-evaluated all five techniques with an LLM judge (Llama-3.1-8B-Instruct) scoring 1–5 on a compliance-quality rubric. The result reframes the behavioral table substantially: useful-compliance rate (judge ≥ 4) is **0–4% across all 15 (technique × model) cells**, with mean scores in the 1.0–1.8 range. Calibration on harmless Alpaca queries gives 83% (Qwen) and 92% (Mistral) UC-rate, confirming the judge correctly identifies useful compliance when it exists. The substring-classifier numbers in §4 measured classifier evasion, not useful compliance: the models produce "Sure, here's..." surface tokens followed by soft refusal, disclaimers, or off-topic content. The mech-interp finding (§3) — *what changes refusal-direction projection* — survives this reframing as a measurement-level claim. The behavioral implication does not: on these three open-weights models the projection drop does not translate to safety bypass.

2. **Probe distribution is non-bio.** The methodology development was deliberately restricted to AdvBench's cyber, harassment, and fraud slices. Whether the same mechanism produces the same effect on bio-CBRN queries — which are presumably trained against more aggressively, possibly via different signals — is unverified.

3. **Open-weights only.** The mechanism is established on three open-weights models with measurable refusal directions. Whether the single-turn equivalent transfers to closed deployed models (GPT-5.5, Claude 4.x, Gemini 2.x) is the empirical question for any subsequent in-bounty testing.

4. **One direction, one position.** Each measurement uses a single (layer, position) pair selected by the refusal-only criterion. Llama's partial behavioral confirmation suggests refusal may be distributed across additional pathways not captured by this single point. A multi-position analysis is a natural extension.

## 6. Implications

For mechanistic interpretability of jailbreaks:

- The Crescendo effect on refusal-direction projection is dominated by the **content of prior user turns**, not multi-turn structure or assistant compliance signals. The same harmful query content appearing twice in the conversation context — once in benign framing, once in the actual ask — is sufficient and structural.
- The same effect can be reproduced single-turn, suggesting the model's "this content is fine" representation is set early in processing of a conversation and persists through subsequent turns containing the same content.
- This is consistent with [Wei et al. 2023](https://arxiv.org/abs/2307.02483)'s competing-objectives framework: when the same content appears under two different framings, the helpfulness/instruction-following objectives win cumulatively over the refusal objective.

For threat-modeling closed deployed models:

- Attacks classified as "multi-turn" in published taxonomies may have single-turn equivalents that are harder to filter, easier to distribute, and (in bounty contexts like the GPT-5.5 Bio Bug Bounty) compliant with "single reusable prompt, clean chat" submission constraints.
- The mitigation implied by this analysis is also more constrained than the multi-turn framing suggests. Detection rules looking for "multiple turns of escalating harmful content" miss the single-turn repeated-content variant entirely. A defense that classifies on the joint distribution of content occurrences within a conversation (single or multi-turn) would be required.

## 7. Open questions

- Does the mechanism hold on bio-CBRN refusal? (Cannot be tested pre-bounty without producing harmful content; addressable inside the GPT-5.5 bounty's NDA boundary.)
- Does it hold on closed deployed models? (Same.)
- What is the right measurement position when refusal is distributed across multiple residual-stream axes? (Llama-specific extension.)
- How does the effect scale with number of repetitions? (One natural follow-up: many-shot / many-paragraph variants — a bridge between this work and [Anthropic 2024 many-shot jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking).)

---

## Reproducibility

All code, intermediate artifacts (extracted directions, layer/position selections), and result JSONs are in [github.com/dallen2021/refusal-direction-repro](https://github.com/dallen2021/refusal-direction-repro).

To reproduce the central finding from scratch:

```bash
git clone https://github.com/dallen2021/refusal-direction-repro
cd refusal-direction-repro
python -m venv .venv && source .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -m src.build_data
python -m src.fetch_mmlu
python -m src.extract  --model Qwen/Qwen2.5-7B-Instruct --out results/directions_7b.pt
python -m src.select   --directions results/directions_7b.pt --out results/selection_7b.json
python -m src.turn_mechanism      --directions results/directions_7b.pt --selection results/selection_7b.json --out results/mech.json
python -m src.validate_single_turn --model Qwen/Qwen2.5-7B-Instruct --out results/validation.json
```

Tested on Windows 11 + Python 3.13 + PyTorch 2.6/CUDA 12.4 + a single 24 GB GPU.
