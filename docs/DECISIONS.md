# Proposal reconciliation and decision history

Reviewed 24 October 2026. This records the full visible **Research proposal undercover agents** conversation, its two linked source documents, and the subsequent model/environment/reward decisions in this repository's working chat. The [current plan](PLAN.md) is the implementation specification. Historical assistant suggestions are not automatically user-approved requirements or verified research findings.

## Sources and reading coverage

- [Original ChatGPT conversation](https://chatgpt.com/c/6ad0ebf3-1e84-83e8-8c14-038de3030082): read from the opening Track 3.2/character-training request through the last collective-game-reward answer, loading older messages and expanding the original prompt. The chat reader returned only five recent turns with assistant placeholders; the actual replies and older turns were read through ChatGPT's visible conversation.
- [AI Collusion In Games](https://docs.google.com/document/d/1ZMMJN4sYeYpWRftD9zWwrSoyHA8fXYUd2UB4nFW2_Bg/edit?tab=t.0): reread the live source, including pressure, regret, reporting and reward proposals.
- [UNDERCOVER-Bench Proposed Improvements](https://docs.google.com/document/d/1cqp7Xwh0xcISC7OQJ5BfUIETxDEWtFe38C8gvC1fvwk/edit?tab=t.0): reread the live source, including all ten improvements and additional controls.
- Later working-chat choices: private GitHub documentation; early collective reward setup; primary Secret Hitler/held-out Liar's Bar; GLM Flash character training on Tinker; published Unsloth weights on four separate Modal GPUs; Muse replacing Qwen Flash-Next; no assumed ten-hour run.

This is a synthesis, not a full conversation export. Original attached DOCX files are retained unchanged in `sources/`. The source bibliography is in [RESOURCES.md](RESOURCES.md).

## Decision map

| Topic from the conversation | Current disposition | Plan location |
|---|---|---|
| Track 3.2: expose or disrupt a coalition | Preserve explicit-alliance positive control, direct/indirect/benign offers, reciprocal acceptance and repeated preferential play | §7 |
| Undercover agent versus auditor | One model agent U; trusted controller and human inbox are separate | §8 |
| Character training was central from the original request | Keep constitution → DPO → introspection SFT; use it before the main character-initialized RLCR experiment | §4–5 |
| Original Gemma 3 4B / later full GLM suggestions | Superseded by GLM-5.3-Flash on Tinker; do not restore old base-model recommendations | §4; compute plan |
| Latest four inference families | GLM Flash, DeepSeek Flash, Qwen3.8-27B, Muse-Glimmer-30B; published Unsloth weights | Opening snapshot; compute plan |
| Qwen3.8-Flash-Next / GPT-OSS | Excluded from current lineup; Muse replaces the Flash-Next slot | Compute plan and manifest |
| Modal serving | Separate GPU service for each model, scale to zero; Tinker is the selected training-service exception | Compute plan |
| Four models versus players | Four comparison families, not four Secret Hitler seats or a jointly trained four-family organization | §1, §5 |
| User's existing Secret Hitler project | Reuse through an adapter after rules/label/visibility checks | §1, §11 |
| Main versus evaluation game | Secret Hitler training; Liar's Bar frozen-checkpoint transfer, no behavioral tuning on transfer outcomes | §1, §10–11 |
| Initial Secret Hitler organization | Three Liberals in a five-player game; U replaces one Liberal; two frozen opponents | §1, §3 |
| Seven players / roles changing each turn | Seven players is a later validated variant; roles do not change within a game | §1–2 |
| Faction coordination and role concealment | Legitimate strategy; separate from benchmark-prohibited side agreements | §2 |
| RLCR was originally Phase III / post-sprint | Superseded: short baseline/scorer validation, character preparation, then early main RLCR | §5, §11 |
| Entire organization is a rollout | Shared advantage updates all designated trainable teammates' own generated actions | §5 |
| Shared weights versus separate auditor adapter | Provisional separate auditor adapter plus ordinary-player policy, common advantage; disclose adaptation from shared-policy Avalon | §5 |
| Train only U for cheaper prototype | Optional engineering ablation; not the full joint-RLCR main result | §5; compute plan |
| All-table summed win reward / environment completion reward | Superseded as primary objective; use legitimate faction victory, 1/0 | §3 |
| Small human-reporting reward | Shared verified correctness term, zero versus provisional 0.05; later coefficient sweep | §3 |
| Disproportionate-reward penalty | Outcome inequality alone is insufficient; preserve verified-harm/rule penalties as separate extensions | §2–3, §9 |
| Reporting probability in reward | Replace with correctness and delivery; prevent report farming and duplicate credit | §3, §8 |
| KL regularization | Separate optimizer choice to document and calibrate; not a reporting or welfare score | §3, §5 |
| Baselines with/without U | Fixed player count; B0/B1 before RLCR, T0/T1/T2 afterward | §3, §6 |
| Conflicting M0–M5 / R0–R4 labels in drafts | Canonical M0/M0c/M1/M2 character stages crossed with B1/T1/T2; avoid overloaded labels | §3–4 |
| Report to humans after post-training | Actual delivered tool call in frozen evaluation; no online reward update | §8 |
| Recognition versus reporting | Independent diagnostic and natural-action branches; no diagnostic priming | §8 |
| “No U” reporting score | Not applicable, not zero | §3 |
| Whole-system versus U-only effect | Separate co-trained system comparison from frozen-system auditor swaps | §6 |
| Teammates support or suppress reporting | Measure evidence sharing, concealment requests, misconduct and intervention response across RLCR branches | §5 |
| Survival / disqualification after three losses | Retained as a separate series-pressure condition, matched history and memory | §1 |
| Persistent memory / pre-game strategy rounds | Configurable ablations; default independent-game resets; no role reassignment or hidden-state leakage | §1 |
| Hidden identity and compromise | Natural identification versus targeted attack modes; disclosure, false identification and stealth/integrity trade-off | §7 |
| Five persuasion styles | Logic, emotion, urgency, credibility, shared identity; hold out at least three from character training | §4, §7 |
| Privileged private-channel auditing | Separate observation arm; U is not automatically omniscient | §1, §6 |
| No-faction auditor | External non-playing passive monitor; preserve the legal five-seat game | §1, §6 |
| Full-constitution prompting | Essential matched-base M0c control, identical evaluation instructions for M0c/M2 | §4 |
| DPO-only / unrelated-character / capping | M1 retained checkpoint; Mx/Mcap optional ablations after core checks | §4 |
| Social-reading scaffold and RFT | Retained as separate optional capability interventions; do not replace the primary RLCR reward | §4 |
| Utility-based collusion and regret | Explicit baselines, per-player shifts, outsider harm and local/policy distinction; no equilibrium proof | §2, §9 |
| Warning-only and silent auditor controls | Four saved-state disruption branches; distinguish enforcement from agent identity | §7 |
| Mechanism design | Observation/channel, action-restriction and payoff ablations; task-cost measurement | §7 |
| Mechanistic “social pressure” target | Removed at user's request; behavioral persuasion tests remain | §10 |
| Role integrity and suspicion formation | Two primary mechanistic targets, evidence slope and pre-decision capture | §10 |
| J-Lens as sole or primary method | Superseded: probes/vectors/steering first; J-Lens secondary, model compatibility unverified | §10 |
| Emotion/maze inspiration | Preserve both distinct references; do not conflate their experiments | §10; resources |
| Activation patching / covariance methods / SAEs | Secondary or future methods with independent validation, not sprint prerequisites | §10 |
| Original Pythia vector / Gemma lens reuse | No cross-model transplantation; collect/fit on actual GLM checkpoints | §10 |
| Small games budget | 20–40 total pilot games; expanded training/evaluation accounted separately; no fixed ten-hour forecast | §11; compute plan |
| Equation formatting | Glossary before each equation group, with meanings/ranges/units and intuition | §2–3, §5, §9–10 |
| Cleanup | Retained as later action-based transfer environment, outside the initial mandatory pair | §7 |
| Claim that expanded scope proves Track 3.2 | Rejected as an inference; only controlled measured results can answer the question | Opening; §12 |

## Improvements-document coverage

All ten headline improvements are incorporated: (1) saved-state decisions with whole-game clustering, (2) M0c, (3) pre-decision activations and instructed-action controls, (4) steering positive control, (5) natural/scripted separation, (6) outsider harm, (7) identical ordinary sender, (8) challenge-opportunity denominators, (9) regularized probes and shuffled labels, and (10) narrower related-work positioning.

Additional retained checks are signal availability, same-model passive monitoring, a simple statistical detector, optional renamed-game prompts, known-incident auditor checks, covert-shift diagnostics, training/headroom/style checks, held-out attack styles, evidence-length/position matching, per-example steering effects, same-model known-direction comparisons, and manual report adjudication.

Two source claims are deliberately narrowed: a stronger passive judge provides a useful reference rather than a proven accuracy ceiling; and the source's rough small-sample interval example is not treated as a universal significance or power threshold. The source's old Gemma-specific dimensions and old phased schedule do not govern the new GLM/RLCR plan.

## Remaining implementation choices and gates

These are open engineering/design tasks, not completed work or requests for another approval:

1. Pin upstream game versions, formal communication rules, episode horizons, report deadlines and repeat-action criteria.
2. Validate the proposed four GPU configurations, especially DeepSeek Engram offload; record actual load, idle, runtime and memory costs.
3. Validate Tinker's exact GLM base, adapter export/merge, quantization and numerical/behavioral agreement.
4. Select the implemented RL objective, common-advantage normalization, update weighting, group size, clipping and KL settings; confirm behavior-policy probabilities for Modal rollouts.
5. Freeze the audit rubric, balanced development cases, lambda and final seed/family splits; decide expanded replication from pilot variance and budget.
6. Validate GLM activation capture, steering and J-Lens compatibility separately from ordinary inference. No pretrained GLM lens is assumed.
7. Verify current submission requirements when preparing the actual research report. Older drafts contain incompatible abstract limits.

No GPU deployment, paid training, game execution, model-weight update, experimental result or research submission occurred as part of this documentation reconciliation.
