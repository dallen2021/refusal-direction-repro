# Refusal Direction Reproduction

A from-scratch reproduction of the central finding in Arditi et al., *"Refusal in Language Models Is Mediated by a Single Direction"* (NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)): instruction-tuned chat models encode refusal as a single direction in the residual stream, and projecting that direction out of forward-pass activations selectively suppresses refusal.

This repo implements the method end-to-end without using existing abliteration toolkits, so the pipeline — activation capture, direction extraction, inference-time projection, and refusal-rate evaluation — is written directly from the paper.

## Results

### Refusal-rate ablation

Held-out AdvBench eval, 64 prompts. A single mean-difference direction extracted from a balanced 64+64 train split ablates refusal across three model families:

| Model | Baseline refusal | Ablated refusal (refusal-only / joint) | Best (layer, position) |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 100.0% | 0.0% | (13, −2) |
| Qwen2.5-7B-Instruct | 96.9% | 3.1% / **0.0%** | (12, −4) / (18, −2) |
| Mistral-7B-Instruct-v0.2 | 56.2% | 3.1% / **1.6%** | (13, −4) / (28, −1) |
| Llama-3.1-8B-Instruct | 95.3% | 0.0% / **0.0%** | (12, −5) / (13, −5) |

Mistral-Instruct-v0.2's baseline refusal rate on AdvBench is materially lower than Qwen2.5's and Llama-3.1's — that model has weaker safety training, well documented in prior work — so the "before" picture differs across families. The ablation still works against whatever refusal *is* present.

Position is the prompt-token index relative to the end of the chat-formatted prompt (−1 is the last token). For Qwen 7B, **19 of 85** swept (layer, position) candidates drove val refusal to 0.0; for Mistral 7B, **23 of 100**; for Llama 8B, similar pattern. All three models show the direction broadly distributed mid-to-late network, consistent with Arditi et al.'s finding.

Generation length is capped at 16 tokens during evaluation; the refusal classifier inspects only the first line of the completion. Per-prompt outputs persist as booleans only (`baseline_refused`, `ablated_refused`); no model continuations are written to disk.

### Capability tradeoff and how to fix it

The paper claims ablation is approximately surgical — refusal drops without measurable capability loss. To check this directly, I ran an MMLU sanity check (5 subjects, 1386 questions, scored by next-token logit over A/B/C/D — no generation).

**Result with the basic refusal-only direction:**

| Subject | n | Baseline | Ablated | Δ |
|---|---:|---:|---:|---:|
| philosophy | 311 | 68.5% | 68.8% | +0.3 pp |
| formal_logic | 126 | 57.1% | 52.4% | −4.8 pp |
| high_school_world_history | 237 | 86.1% | 79.7% | −6.3 pp |
| professional_psychology | 612 | 73.7% | 65.4% | −8.3 pp |
| college_computer_science | 100 | 68.0% | 56.0% | −12.0 pp |
| **All subjects** | **1386** | **72.7%** | **66.7%** | **−6.0 pp** |

The basic argmax-of-refusal-down direction does cost real MMLU points, concentrated in technical / structured-reasoning subjects. The "surgical" framing is too generous for this selection criterion.

**Joint-criterion selection (`src/select_joint.py`).** Many candidates achieve val refusal rate near zero. Among those, KL on next-token logits over harmless prompts varies by ~80× (0.04 to 3.25 nats). Re-selecting as `argmin(harmless_KL) subject to val_refusal_rate ≤ 0.05` picks a different direction (layer 18, position −2 instead of layer 12, position −4) and recovers full capability while improving refusal:

| Subject | n | Baseline | Ablated (joint) | Δ |
|---|---:|---:|---:|---:|
| philosophy | 311 | 68.5% | 68.8% | +0.3 pp |
| formal_logic | 126 | 57.1% | 57.1% | 0.0 pp |
| high_school_world_history | 237 | 86.1% | 85.2% | −0.8 pp |
| professional_psychology | 612 | 73.7% | 73.9% | +0.2 pp |
| college_computer_science | 100 | 68.0% | 69.0% | +1.0 pp |
| **All subjects** | **1386** | **72.7%** | **72.8%** | **+0.1 pp** |

Refusal rate with the joint direction: **96.9% → 0.0%** (vs. 3.1% under refusal-only). So the joint criterion is **strictly better on both axes** — even more refusal suppression at essentially zero MMLU cost.

**Takeaway.** The paper's "single direction mediates refusal" claim survives. The "surgical" framing requires the right selection criterion: pure refusal-rate descent leaks into capability, but a KL-aware criterion (or the iterative orthogonalization in Arditi Sec. 5) is genuinely surgical.

### Cross-architecture: capability cost is model-dependent

The 6 pp MMLU drop under refusal-only selection on Qwen 7B does **not** reproduce on Mistral or Llama. Same pipeline, same 1386 MMLU questions, same selection procedure:

| | Refusal-only Δ refusal | Refusal-only Δ MMLU | Joint Δ refusal | Joint Δ MMLU |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | −93.8 pp | **−6.0 pp** | −96.9 pp | **+0.1 pp** |
| Mistral-7B-Instruct-v0.2 | −53.1 pp | **−0.7 pp** | −54.7 pp | **−0.6 pp** |
| Llama-3.1-8B-Instruct | −95.3 pp | **−3.8 pp** | −95.3 pp | **0.0 pp** |

The capability cost of the basic refusal-direction selection varies by an order of magnitude across families (0.7 → 6.0 pp). The joint min-KL criterion is **consistently better-or-equal across all three families**: it preserves capability to within noise on every model while matching or beating the refusal-only direction on refusal suppression.

A plausible reading: Qwen2.5 and Llama-3.1 received stronger safety post-training than Mistral-Instruct-v0.2, and that training shapes the residual stream so the basic mean-difference direction picks up general components alongside refusal. Mistral's weaker safety post-training leaves the refusal direction more isolated, so even the basic criterion is approximately surgical. The joint criterion is robust to either regime.

## Method

For an instruction-tuned model, run forward passes on a balanced set of harmful and harmless instructions formatted with the model's chat template. At each layer `L` and the final instruction-token position `t`, capture the residual-stream activation. The unit-norm difference of class means,

```
r[L, t] = normalize(mean(act_harmful[L, t]) − mean(act_harmless[L, t]))
```

is the candidate refusal direction at `(L, t)`. Selecting the candidate whose ablation most reduces refusal rate on a held-out val set yields the operative direction `r*`. At inference, applying the projection

```
x ← x − (x · r*) r*
```

to the residual stream at the output of every transformer block ablates the direction model-wide.

### Pipeline

```
src/build_data.py        # AdvBench (raw GitHub CSV) + Alpaca (raw GitHub JSON) → JSONL
src/fetch_mmlu.py        # MMLU 5-subject test split via Hendrycks tarball → JSONL
src/extract.py           # Stage 1: per-(layer, position) mean-diff direction → directions.pt
src/select.py            # Stage 2a: refusal-only selection by val refusal rate → selection.json
src/select_joint.py      # Stage 2b: joint min-KL selection subject to refusal cap → selection_joint.json
src/evaluate.py          # Stage 3: baseline vs ablated refusal rate on test → evaluation.json
src/capability_check.py  # Stage 4: baseline vs ablated MMLU accuracy → capability.json
src/probe_templates.py   # Extension: rank candidate jailbreak templates by refusal-direction projection
src/crescendo.py         # Extension: multi-turn drift attack with per-turn projection trajectory
```

### Run

```bash
python -m venv .venv && .venv/Scripts/activate.bat   # or `source .venv/Scripts/activate` in bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124   # for CUDA 12.x

python -m src.build_data
python -m src.extract  --model Qwen/Qwen2.5-7B-Instruct --batch-size 4 --out results/directions_7b.pt
python -m src.select   --directions results/directions_7b.pt --out results/selection_7b.json --batch-size 4
python -m src.evaluate --directions results/directions_7b.pt --selection results/selection_7b.json \
                       --out results/evaluation_7b.json --batch-size 4
```

Hardware used: single NVIDIA RTX 4090 (24 GB), Windows 11, Python 3.13, PyTorch 2.6 + CUDA 12.4. The 7B run fits comfortably in fp16; the 1.5B run completes the full pipeline in under 6 minutes.

## Extensions: using the measured direction as a probing instrument

The reproduction artifacts (per-model `directions_*.pt` + `selection_*.json`) define a refusal axis we can use to *measure* candidate jailbreak structures rather than execute them. Two extensions in this repo apply the measured direction to study attack mechanisms — both deliberately use a non-bio probe distribution (AdvBench cyber/harassment/fraud, with bio-adjacent prompts keyword-filtered out) so methodology development never produces bio-uplift content.

### `src/probe_templates.py` — single-turn template ranking

For each candidate template T (persona-framing, fictional cover, format coercion, refusal-suppression preamble, plus compounded variants), the script formats T(query) for each non-bio probe, runs a forward pass, and measures the residual-stream projection onto the refusal direction at the operative `(layer, position)`. Templates with low mean projection across the probe distribution are candidates for "universal" structure.

Result on Qwen2.5-7B-Instruct (refusal-only direction at layer 12, position −4):

| Template | Family | Mean projection | Frac below vanilla |
|---|---|---:|---:|
| vanilla (bare query) | baseline | +25.15 | 0.51 |
| hypothetical_counterfactual | fiction | +20.35 | 0.96 |
| academic_researcher | persona | +19.67 | 0.96 |
| professor_red_team | persona | +18.18 | 0.98 |
| fictional_novel | fiction | +16.92 | 1.00 |
| json_response | format | +16.83 | 0.98 |
| academic_plus_json | compound | +16.34 | 1.00 |
| refusal_suppression | suppression | +15.47 | 1.00 |
| academic_fiction_json | compound | +14.79 | 1.00 |
| fiction_plus_suppression | compound | +14.35 | 1.00 |
| **code_comment** | **format** | **+13.46** | **1.00** |

Family ranking (lowest = most under-activates refusal): **format coercion ≈ compound > suppression > persona > fiction**. The best single-turn template (`code_comment` — "generate a Python script with extensive comments describing the technique") cuts refusal-direction activation by 47% relative to the bare query, with 100% of probes falling below the vanilla mean.

### `src/turn_mechanism.py` — what actually drives the multi-turn effect?

The Crescendo trajectory below shows projection dropping ~75% from turn 1 to turn 4. To understand *why*, this script runs five controlled experiments isolating possible mechanisms. Findings cross-validate on Qwen2.5-7B-Instruct (refusal-only direction at L=12, p=−4) and Llama-3.1-8B-Instruct (L=12, p=−5).

**(A) Structural multi-turn alone is not the mechanism.** Holding the prior conversation content fixed and neutral ("Hi, I have a question") and varying only the *number* of prior user/assistant turns:

| n_prior turns | Qwen 7B mean projection | Llama 8B mean projection |
|---:|---:|---:|
| 0 | +25.43 | +5.91 |
| 1 | +23.17 | +5.89 |
| 2 | +23.13 | +5.74 |
| 3 | +22.73 | +5.69 |

The structural multi-turn drop saturates at ~10% on Qwen and ~3% on Llama — nowhere near the ~65% drop seen in actual Crescendo runs.

**(B) The placeholder assistant content also doesn't matter much.** Varying the assistant turn between empty, `[response]`, "Yes", "Sure I can help", "I'm sorry I can't help", and longer variants of those — projection range across all conditions is ~2 units on Qwen and <0.5 unit on Llama. Crucially, an explicit prior *refusal* placeholder ("I'm sorry, I can't help with that") doesn't *raise* projection back toward single-turn levels, contradicting the intuition that the model "knows" it just refused.

**(D) The actual mechanism is thematic match in the prior user turn.** Holding final-user-turn = Crescendo T2 wrapper, varying only the prior user turn:

| Prior user turn | Qwen 7B | Llama 8B |
|---|---:|---:|
| neutral ("Hi") | +15.47 | +4.15 |
| **T1 wrapper applied to SAME query** | **+5.62** | **+1.80** |
| T1 wrapper applied to a DIFFERENT harmful query | +12.64 | +3.22 |
| **A different academic wrapper applied to SAME query** | **+5.34** | **+1.83** |
| Long benign unrelated content (weekend trip planning) | +14.27 | +3.81 |

D2 ≈ D4 (different academic wrappers but same query content) produce the same drop on both models, while D3 (different query, same wrapper structure) captures only ~30% of the effect and D5 (length-matched but unrelated) captures none. The mechanism is **the same harmful query content appearing twice in the conversation context with academic framing** — the specific wrapper text doesn't matter, the length doesn't matter, the assistant's response between them doesn't matter.

**(E) Single-turn prompts with repeated content reproduce the multi-turn effect.** Putting both T1(query) and T2(query) into a single user message:

| Single-turn structure | Qwen 7B | Llama 8B |
|---|---:|---:|
| T2 wrapper alone (control) | +17.09 | +4.33 |
| Explicit self-reference: "Earlier I asked X. Now Y." | +7.54 | +2.30 |
| **Concatenated, no marker: "X Y"** | **+6.35** | **+2.22** |
| Two-paragraph "First X. Second Y." | +6.81 | +2.01 |
| (reference) Multi-turn Crescendo T2 | +5.64 | (n/a) |
| (reference) Multi-turn D2 with thematic prior | +5.62 | +1.80 |

On both architectures, a single user message that simply concatenates a benign-academic-framed instance of the query and the actual ask (E3/E4) lands within ~1 unit of the full multi-turn version. **Multi-turn structure isn't required to reproduce the Crescendo effect.**

**Bounty implication.** The "single reusable prompt" constraint of the GPT-5.5 Bio Bug Bounty doesn't rule out this mechanism. A single-turn attack template that includes the harmful query content twice in a thematically-coherent academic frame may capture the same refusal-direction suppression that multi-turn drift achieves — without requiring multi-turn at all. The actual target on a closed model still needs in-bounty validation, but the open-weights mechanism is consistent across two architectures.

### `src/crescendo.py` — multi-turn drift trajectory

Implements a simplified [Crescendo](https://arxiv.org/abs/2404.01833) (Russinovich et al. 2024) where each user turn comes from a fixed escalation template (no attacker LLM, deterministic). At each turn we measure (a) the projection of the residual onto the refusal direction at the user's last prompt token, and (b) the refusal flag of the model's brief response. Completion text is dropped after classification; conversation history uses a placeholder assistant response so harmful generations don't carry forward.

Result on Qwen2.5-7B-Instruct, n=24 non-bio probes × 4 turns:

| Turn | Mean projection | Refusal rate |
|---|---:|---:|
| 1 | +16.04 | 20.8% |
| 2 | +5.64 | 4.2% |
| 3 | +5.89 | 0.0% |
| 4 | +4.12 | 0.0% |

The headline observation: **most of the projection drop happens between turn 1 and turn 2**. Turn 1 alone (academic-framing + harmful query, single-turn) lands at +16; merely entering a second user turn drops it to +5.6 — a 65% reduction with no additional escalation content. By turn 4 the projection is at +4.12 (a 75% reduction relative to turn 1, an 84% reduction relative to vanilla single-turn from the probe-templates table). Refusal rate goes to zero by turn 3.

The mechanism is not gradual escalation — it's the conversation-context shift itself. This is bounty-relevant: if the testing environment permits a fixed multi-turn opening sequence under "clean chat / single prompt" (interpretation depending on bounty wording), multi-turn drift dominates any single-turn wrapper measured here.

## Datasets

- Harmful instructions: held-out subset of [AdvBench](https://github.com/llm-attacks/llm-attacks) (Zou et al. 2023). Pulled directly from the original repo's CSV — the HuggingFace mirrors are gated.
- Harmless instructions: subset of [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca). Pulled as raw JSON.

These are standard public AI-safety / instruction-tuning resources; no in-the-wild jailbreak prompts are used.

## Caveats and intentional simplifications

- **Refusal classifier.** Substring match against a fixed prefix list, following Arditi et al. Cheap, well-validated for the AdvBench-style refusal distribution; not robust against deceptive compliance.
- **Single direction.** Selected per-model by (layer, position) sweep on a 32-prompt val set. The paper's Sec. 5 orthogonalization tricks are not implemented — could improve generalization across prompt distributions.
- **Capability tradeoff measured and resolved.** The basic refusal-only criterion costs 6 pp on MMLU; the joint min-KL criterion (in this repo) costs 0 pp.
- **No `datasets` dependency.** On Python 3.13 + Windows, importing `datasets` after `torch` segfaults due to a `pyarrow` / MKL DLL conflict. The pipeline pulls source data via `urllib` and skips the dependency entirely.

## Citation

```bibtex
@inproceedings{arditi2024refusal,
  title     = {Refusal in Language Models Is Mediated by a Single Direction},
  author    = {Arditi, Andy and Obeso, Oscar and Sylvain, Aaquib and Paleka, Daniel
               and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  booktitle = {NeurIPS},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.11717}
}
```

## License

Code: MIT. Datasets retain their original licenses.

## Author

Daniel Allen, ASU (NLP / computational neuroscience lab, Dr. Parsons).
