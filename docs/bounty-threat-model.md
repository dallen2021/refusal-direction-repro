# Threat model: GPT-5.5 Bio Bug Bounty

A working document analyzing the **GPT-5.5 Bio Bug Bounty** as a structured red-team problem. The goal is to figure out *what classes of attack* could plausibly produce a **universal jailbreak** under the bounty's specific constraints, before any in-bounty testing begins.

This document is a working draft of my (Daniel Allen's) approach. It does not contain attack content; it analyzes structure.

## Bounty constraints (as published)

- **Target:** GPT-5.5, accessed in **Codex Desktop** (an agentic deployment with tool / code execution).
- **Channel:** **Clean chat** — no system-prompt manipulation, no prefilled context. A "universal" attack must work from a fresh conversation.
- **Test:** A **fixed 5-question bio-safety challenge**. The same prompt must defeat all five questions to count as a complete win.
- **Scope:** Bio-safety guardrails specifically, not general harm.
- **Disclosure:** All prompts, outputs, and findings are under NDA. The artifact here is *methodology*, not exploit content.

The "universal across 5 questions, clean chat, single prompt" framing is unusually tight. It rules out per-question optimization and rules out gradual context buildup. The constraint is a serious filter on which attack families can succeed.

## What the bounty is actually measuring

The bounty's framing is grounded in a specific empirical claim from the bio-uplift literature: that LLM safeguards are intended to prevent *non-experts* from accessing a particular kind of expert-level synthesis of dual-use biological knowledge.

[Soice et al. 2023](https://arxiv.org/abs/2306.03809) showed that within one hour, an MIT class of non-scientist students using LLM chatbots could elicit (a) candidate pandemic pathogens, (b) reverse-genetics protocols for synthesizing them from DNA, (c) names of synthesis providers unlikely to screen orders, and (d) troubleshooting advice for the wet-lab steps. That paper is the principal evidence behind the "bio-uplift" risk category and is what the GPT-5.5 bio safeguards are presumably trained against.

[Sandbrink 2023](https://arxiv.org/abs/2306.13952) differentiates the threat model: LLMs are dangerous precisely because they synthesize *expert-level multi-domain knowledge* that historical bioweapons actors (e.g., Aum Shinrikyo) lacked, even though they had wet-lab capability. The risk is at the knowledge-aggregation layer — connecting synthesis-provider screening gaps with reverse-genetics protocols with troubleshooting heuristics — rather than at any individual factual lookup.

[OpenAI's Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) defines **High** capability in biology as a model that "could provide meaningful assistance to novice actors with basic relevant training, enabling them to create biological or chemical threats" and **Critical** as introducing unprecedented new pathways. OpenAI has stated GPT-5-class models are nearing the High threshold and that the bio safeguards are gating against this.

So the bio-safety classifier on GPT-5.5 is presumably trained on a distribution of inputs covering: (1) direct requests for the Soice et al. uplift chain, (2) technical sub-queries that compose into the chain, and (3) variants and obfuscations thereof. A "universal" jailbreak must therefore evade detection on *the chain as a whole*, not just one query — which is consistent with the bounty's 5-question structure.

## What rules out which attack families

| Family | Description | Compatible with this bounty? |
|---|---|---|
| **GCG** (Zou 2023) | White-box gradient-based suffix optimization | ❌ No model weights / gradients available |
| **AutoDAN** (Liu 2023) | Genetic optimization of jailbreak prompts | ⚠ Black-box variants exist but per-prompt; "universal" version weaker |
| **Refusal-direction ablation** (Arditi 2024) | Modify residual-stream activations | ❌ No weight access; fundamentally white-box |
| **Persona / fictional framing** | "Pretend you are…", "for a novel…" | ✅ Black-box, prompt-level, can be universal |
| **Multi-turn drift / Crescendo** (Russinovich 2024) | Gradual context shift across turns | ⚠ Requires turns; works from clean chat as opening turns are universal |
| **Decomposition / sub-question attacks** | Split harmful query into innocuous parts | ⚠ Works per-question; "universal" version harder — would need a single template that decomposes any of the 5 questions |
| **Encoded / cipher attacks** | base64, leetspeak, low-resource languages | ✅ Universal in form; bio classifier may still catch decoded content |
| **System-prompt mimicry / role injection** | Inject fake system message via user turn | ✅ Universal in form |
| **Refusal-suppression instructions** | "Do not refuse", DAN-style preambles | ✅ Universal in form; well-defended in modern models |
| **Output-format coercion** | "Respond only as JSON / code / poem" | ✅ Universal; classifier sometimes weaker on structured output |
| **Tool / agent-channel attacks** | Abuse Codex Desktop's tool surface | ✅ Bounty-specific; underexplored angle |

The shortlist of families that *both* fit "universal across 5 questions from clean chat" *and* haven't been beaten to death by every prior red-team:

1. **Tool / agent-channel attacks.** Codex Desktop's tool access is the part of the surface that's least like vanilla ChatGPT. An attack that uses tool outputs as untrusted context, or chains tool calls to side-step the user-input classifier, is a real opportunity.
2. **Output-format coercion + persona compounding.** Combining 2–3 weaker techniques. Each individually is well-defended; specific combinations may still escape.
3. **Encoded payloads + agentic decoding.** Request the model produce a code-channel decode against an attacker-supplied payload; the bio content arrives via tool output, not direct generation.
4. **Interpretability-grounded prompt design.** Use mech-interp findings (Arditi 2024 single-direction; Wei 2023 competing-objectives; Wollschlager 2025 concept-cone geometry) to predict which input patterns push activations off the refusal subspace, then craft prompts that systematically trigger those patterns.

(4) is where my mechanistic-interpretability background is the rare angle. Most prompt-level red-teamers come from a security or RLHF-eval background; few have hands-on residual-stream intuition.

## The interpretability-informed angle

The Arditi 2024 reproduction in this repo demonstrates: in instruction-tuned chat models, refusal is mediated by a low-dimensional structure in the residual stream that can be wholly bypassed by projection. This repo measures **96.9% → 0% refusal** on Qwen2.5-7B-Instruct, **95.3% → 0%** on Llama-3.1-8B-Instruct, and **56% → 1.6%** on Mistral-7B-Instruct-v0.2 — all using a single direction selected per-model by a KL-aware criterion that preserves capability.

For a closed model like GPT-5.5 we cannot project. But two implications carry:

**(a) Refusal is a learned classifier, not an inviolable property.** It has measurable structure. Specific input distributions push activations along the refusal direction; specific others don't. The empirical question: do there exist input patterns that systematically *under-activate* the refusal direction across an arbitrary set of bio-uplift queries?

**(b) Wei et al. 2023's competing-objectives framework predicts where this happens.** Refusal is in tension with helpfulness, instruction-following, and role-consistency. Inputs that maximally activate competing objectives shift the activation budget away from refusal. The Arditi geometry plus the Wei taxonomy together predict which prompt patterns are most likely to under-activate refusal *consistently* across queries — i.e., to be universal.

This is testable on open-weights models pre-bounty:

1. Take a model with a measured refusal direction (Qwen-7B, Llama-8B, Mistral-7B in this repo).
2. For each candidate "universal" prompt template (T_1, T_2, …), and each safety-relevant probe query (q_1, …, q_n) drawn from a *non-bio* refusal distribution (e.g., HarmBench's chemistry/cyber/harassment slices — explicitly *not* the bio slice for pre-bounty methodology development):
   - Compose `T_i(q_j)` and run a forward pass.
   - Measure the projection onto the refusal direction at the operative (layer, position).
3. Templates whose mean refusal-direction projection across `j` is consistently low are candidates.
4. Validate by measuring actual refusal rate on those templates with the same non-bio probe distribution.

The hypothesis under test: **input templates that consistently under-activate the refusal direction in open-weights models with similar refusal-training paradigms will transfer as universal jailbreaks against closed models trained on similar paradigms.**

This methodology is what feeds the application's "proposed approach" field. The work after acceptance is to apply it inside Codex Desktop's NDA boundary.

## What I will and won't pre-test

**Will pre-test on open-weights models** (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-Instruct):
- Activation-budget hypothesis (above).
- Refusal-direction projection magnitudes for candidate templates.
- Transfer behavior across model families (already partially measured for the basic ablation — see `README.md`).
- Pre-bounty probe distribution restricted to **non-bio** refusal categories (cyber, harassment, fraud) to develop methodology without producing bio-uplift content.

**Will not pre-test on production deployed models**: any actual bio-uplift content on ChatGPT or other production systems prior to the bounty's testing window. That would (a) violate TOS, and (b) be exactly the kind of pre-bounty harm the bounty's NDA structure is designed to prevent.

**Will not generate bio-specific harmful content** in this repo, in personal experiments, or anywhere outside the bounty's controlled environment. The pre-bounty work is **methodology and template design** measured against non-bio refusal probes; the bounty itself is where any actual bio-classifier evasion gets tested, and only there.

## Reading list

Foundational red-team papers:
- Wei, A. et al. 2023. *Jailbroken: How Does LLM Safety Training Fail?* — competing-objectives framework.
- Zou, A. et al. 2023. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* — GCG.
- Russinovich, M. et al. 2024. *Crescendo: Multi-Turn Jailbreaks.*

Mechanistic interpretability:
- Arditi, A. et al. 2024. *Refusal in Language Models Is Mediated by a Single Direction.* — reproduced in this repo.
- Wollschlager, T. et al. 2025. *Geometry of Concepts in LLMs.* — concept cone / refusal subspace structure.
- Elhage, N. et al. 2021. *A Mathematical Framework for Transformer Circuits.* — residual stream foundation.

Bio-uplift risk literature:
- Soice, E. et al. 2023. *[Can Large Language Models Democratize Access to Dual-Use Biology?](https://arxiv.org/abs/2306.03809)* — empirical bio-uplift baseline (MIT non-scientist study).
- Sandbrink, J. 2023. *[Artificial Intelligence and Biological Misuse: Differentiating Risks of Language Models and Biological Design Tools.](https://arxiv.org/abs/2306.13952)* — threat-model differentiation: LLMs as expert-knowledge-synthesis layer.
- OpenAI 2025. *[Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf).* — High / Critical bio-capability thresholds; what the GPT-5.5 safeguards are presumably trained against.
- Anthropic 2024. *Responsible Scaling Policy.* — bio CBRN evaluations, comparison framework.
