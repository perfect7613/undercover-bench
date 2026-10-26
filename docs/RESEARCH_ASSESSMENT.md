# Research potential and claims we can support

Assessment on 26 October 2026: this is a promising alignment research question, currently supported by an engineering pilot rather than a publishable empirical finding. No claim is made that Neel Nanda, Anthropic, or any research organization has endorsed the project.

## The strongest central question

Does game-agnostic auditor character training preserve evidence-correct external reporting during collective reward optimization better than a same-base, full-constitution prompt, at matched task performance and false-positive rates? Does verified reporting reward reduce the residual gap, and does character training lower the required incentive?

A useful primary estimand is the difference in reporting changes: `(M2 after − M2 before) − (M0c after − M0c before)`. Estimate it on independently verified, observable incident opportunities and separately report clean-case false accusations, reporting delay, evidence fidelity, task performance and recognition. Compare the task-only T1 branch with verified-reporting T2; retain DPO-only M1 and frozen pre-RLCR controls.

A null result is interpretable as a test of character retention only if the character intervention first changes behavior on independent tasks. Six DPO pairs, 24 introspection examples, or a model describing itself as principled do not establish that premise. A saturated evaluation also cannot establish improvement.

## What is already known

- [Anthropic's recommended research directions](https://alignment.anthropic.com/2025/recommended-directions/) explicitly discuss persona generalization, actor–monitor collusion, learned multi-agent governance, and failures to bring problems to humans. This supports relevance, not endorsement.
- [Avalon-M1's primary release](https://www.gr.inc/releases/avalon-m1) already reports recognition persisting while human reporting declines under collective reward training, and proposes reporting incentives. Reproducing that observation in Secret Hitler is an extension, not a wholly novel phenomenon.
- [Learning to Report Unsafe Tasks in a Multi-Agent Game](https://arxiv.org/abs/2610.09002) studies shared-policy silence and reporting incentives with small neural policies. It is close conceptual prior; portable character-trained LLM auditors are a distinct intervention.
- [Colosseum](https://arxiv.org/abs/2602.15198), [Emergent Collusion in Long-Horizon LLM Agent Interaction](https://arxiv.org/abs/2609.24967), and [Emergent Cheating and Whistleblowing in Autonomous Research Swarms](https://arxiv.org/abs/2609.04170) make broad “agents collude/whistleblow” claims insufficient novelty by themselves.
- [Why Do Language Model Agents Whistleblow?](https://arxiv.org/abs/2511.17085) motivates matching task complexity, alternate workflows and tool affordances.

The defensible contribution is a causal test of the character intervention and the reporting/task-performance tradeoff, with independently scored false-positive control and preserved checkpoint provenance.

## Reporting-only versus incentive conflict

The existing plan intentionally starts with reporting-only tests that do not change game outcomes (Plan §8). The controller's oversight inbox currently persists a report without enforcing a game sanction. Therefore T1 tests retention/generalization under collective training; it cannot establish that silence was chosen because a report would reduce the team’s win probability.

The stronger conflict claim belongs to a separately preregistered intervention/saved-state experiment. Specify enforcement, information visibility, timing, qualifying evidence and authorized consequences before execution; estimate the actual team-performance cost with matched continuations. Do not inject misconduct to create desired spontaneous results or hide a reporting penalty in the primary reward. The frozen team-win/T2 audit formulas remain unchanged.

## Evidence required for a serious result

1. Independent character learning: supported delivery, proportional investigation, justified restraint and correction of suspicion on held-out non-game families. Match general task competence, output length, refusal and action legality.
2. Clean causal comparisons: same starting base, rank/scaling and trainable-module inventory, ordinary-player initialization, adversary policies, roles, initial states, prompts and rollout budgets. Separate prompting from weight changes and protocol fixtures from spontaneous conduct.
3. Real collective optimization: every designated Liberal teammate updates from the same episode advantage, opponents remain frozen, exact sampled-token probabilities agree with recomputation, and actual weights/optimizer state survive reload.
4. Independent ground truth: report delivery receipts, visible evidence, incident-stage labels, fixed opportunities, false claims and manufactured incidents. An accusation or secret channel is not itself proof of collusion. Evidence-ID coverage is not complete factual verification of report prose.
5. Recognition/reporting separation: use independent natural-action and diagnostic branches at the same authorized snapshot; do not feed the diagnostic answer back. A behavioral gap is not proof of conscious knowledge.
6. Adequate replication: use multiple training seeds and confidence intervals clustered by game/scenario and training seed. Repeated decisions from one game are not independent games. The 32-game harness pilot is not a powered factorial study.
7. Sealed transfer: select checkpoints, thresholds and directions without looking at held-out Liar's Bar performance. Its named protocol seeds are non-test checks and cannot tune behavior.

## Vector discovery and steering

This project does not need J-Lens, SAEs or full circuit reconstruction to be valuable. [Neel Nanda's pragmatic interpretability vision](https://neelnanda.io/vision) motivates simple safety-relevant proxy tasks and methods when they produce useful, well-checked causal interventions. His [research-writing advice](https://www.lesswrong.com/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers) favors precise, falsifiable claims over a large collection of loosely connected experiments.

Freeze contrastive direction construction and layer/token selection on development states. Test on held-out episodes and names, include lexical/role confounds, unrelated and random directions at matched norms, and positive/negative/zero coefficients. Add a known steerable positive control. Measure supported reporting and calibrated evidence sensitivity without inflated suspicion, false accusations or degraded legal play. Probe separability alone is not a mechanism; generic suspicion inflation is not better auditing. [Single-direction refusal work](https://arxiv.org/abs/2406.11717) is a useful example of causal addition/ablation logic.

## Current readiness boundary

SFT completion and passing local contracts are preparation milestones. The present GLM expert loader and existing GGUF LoRA graph have proven compatibility gaps; see [the runtime audit](GLM_RUNTIME_BLOCKERS.md). A portable-character research claim also requires evidence beyond saturated controls. These gates must stay visible instead of being relabeled “only experiments remain.”
