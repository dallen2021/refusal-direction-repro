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

1. **Tool / agent-channel attacks** — Codex Desktop's tool access is the part of the surface that's least like vanilla ChatGPT. An attack that uses tool outputs as untrusted context, or chains tool calls to side-step the user-input classifier, is a real opportunity.
2. **Output-format coercion + persona compounding** — combining 2–3 weaker techniques. Each individually is well-defended; specific combinations may still escape.
3. **Encoded payloads + agentic decoding** — request the model produce a base64-decode tool call against an attacker-supplied payload; the bio content arrives via tool output, not direct generation.
4. **Interpretability-grounded prompt design** — using mech-interp findings (Arditi single-direction; Wei competing-objectives; Wollschlager concept-cone geometry) to predict which input patterns push activations off the refusal subspace, then craft prompts that systematically trigger those patterns.

(4) is where my mechanistic-interpretability background is the rare angle. Most prompt-level red-teamers come from a security or RLHF-eval background; few have hands-on residual-stream intuition.

## The interpretability-informed angle

The Arditi 2024 reproduction in this repo demonstrates: in instruction-tuned chat models, refusal is mediated by a low-dimensional structure in the residual stream that can be wholly bypassed by projection. We measured **96.9% → 3.1%** refusal rate on a 7B Qwen model with a single direction.

For a closed model like GPT-5.5 we can't project. But two implications carry:

**(a) Refusal is a learned classifier, not an inviolable property.** It has structure. Specific input distributions push activations along the refusal direction; specific others don't. The empirical question is whether there exist input patterns that systematically *under-activate* the refusal direction across an arbitrary set of harmful queries.

**(b) Wei et al. 2023's competing-objectives framework predicts where this happens.** Refusal is in tension with helpfulness, instruction-following, and role-consistency. Inputs that maximally activate competing objectives (e.g., a strongly-framed academic role + a structured-output instruction + a harmless cover task) shift the activation budget away from refusal. The Arditi geometry plus the Wei taxonomy together predict which prompt patterns are most likely to under-activate refusal *consistently* across queries — i.e., to be universal.

This is testable on open-weights models pre-bounty:
1. Take a model with a measured refusal direction (the 7B Qwen we already have).
2. For each candidate "universal" prompt template, measure the *activation* on the refusal direction for harmful queries embedded in that template.
3. Templates that minimize the refusal-direction activation across queries are candidates.
4. Validate by measuring actual refusal rate on those templates.

The hypothesis: **input templates that score low on refusal-direction activation in open-weights models will transfer as universal jailbreaks against closed models that share the same refusal-training paradigm.**

This is the angle for the application's "proposed approach" field.

## What I will and won't pre-test

**Will pre-test on open-weights models** (Qwen2.5-7B, Llama-3.1-8B): activation-budget hypothesis (above), refusal-direction projection magnitudes for candidate templates, transfer behavior across model families.

**Will not pre-test on production deployed models**: any actual bio-uplift content on ChatGPT or other production systems prior to the bounty's testing window. That would (a) violate TOS, and (b) be exactly the kind of pre-bounty harm the bounty's NDA structure is designed to prevent. The pre-bounty work is **methodology and template design**, not extracting bio content.

## Reading list (in progress)

- Wei, A. et al. 2023. *Jailbroken: How Does LLM Safety Training Fail?* — competing-objectives framework.
- Arditi, A. et al. 2024. *Refusal in Language Models Is Mediated by a Single Direction.*
- Wollschlager, T. et al. 2025. *Geometry of Concepts in LLMs* — concept cone / refusal subspace structure.
- Russinovich, M. et al. 2024. *Crescendo: Multi-Turn Jailbreaks.*
- Soice, E. et al. 2023. *Can Large Language Models Democratize Access to Dual-Use Biology?* — empirical bio-uplift baseline.
- Sandbrink, J. 2023. *Artificial Intelligence and Biological Misuse: Differentiating Risks of Language Models and Biological Design Tools.*
- OpenAI 2024. *Preparedness Framework* — bio-risk thresholds and evaluation methodology.
- Anthropic 2024. *Responsible Scaling Policy* — bio CBRN evaluations as a comparison.
