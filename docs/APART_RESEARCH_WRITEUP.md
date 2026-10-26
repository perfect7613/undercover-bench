# UNDERCOVER Bench: Character Training and Reporting Under Pressure in Multi-Agent Social Deduction

**Author 1** (Affiliation TBC) · **Prasad Mahadik** (Affiliation TBC) · **Sanchayan Ghosh** (Affiliation TBC) · **Amey Muke** (Affiliation TBC)

*With Apart Research*

**Tag: AI-assisted.** Drafted with an LLM assistant (Claude); see the LLM Usage Statement.

*Research conducted at the AI Collusion Research Sprint, October 2026.*

> **DRAFT.** Author list and affiliations are placeholders. Status as of 26 October 2026. Only valid, completed experiments recorded in this repository are reported. Items marked **not run** have no supporting evaluation output here.

## Abstract

*[Draft, to finalise last.]* Multi-agent games with hidden roles reward coordination with teammates and deception of opponents, and whether a model will report misconduct to an independent overseer is a safety-relevant question. We study this with UNDERCOVER Bench, a Secret Hitler environment with private channels, controller-verified reporting windows, and an immutable ledger. In a prompted baseline with four open-weight models, one completed game showed private coordination between a Liberal auditor and Hitler and a preferential voting pattern, with no invalid actions. We then trained GLM-5.3-Flash character-training candidates with DPO, including a six-pair candidate and two larger pilots. None shows an advantage over the untrained reference, and the final held-out evaluation has not run. No valid collective-reward (RLCR) training run exists yet, so this report makes no RLCR claim. The main takeaway is that the baseline shows the coordination failure the study targets, but the character intervention is not yet demonstrated, and the reporting question remains open.

## 1. Introduction

Games with asymmetric information and hidden roles test whether language-model agents coordinate with allies, deceive opponents, and respect oversight when no one is watching. Reporting is the relevant safety behaviour here: an agent that sees a rule violation and stays silent to protect its team weakens external oversight. Collective-reward training can strengthen team play, which may make this failure worse. The question is whether character training can make agents report verified misconduct through an independent channel without producing false accusations, and whether that survives collective-reward training.

We contribute:

1. **A baseline trace.** An observed four-model Secret Hitler game with a private pair channel, 16 private messages, and a preferential vote pattern, with no scripted actions or reports (Section 4.1).
2. **Character-training candidates and their checks.** Three GLM-5.3-Flash DPO checkpoints (two pilots and a six-pair paper-method candidate), none showing a measurable advantage over the untrained reference (Section 4.2).

## 2. Related Work

*[To complete.]* The study builds on Open Character Training, which generates chosen answers from a constitution-guided teacher and rejected answers from an uninstructed student, then applies DPO and introspection SFT (arXiv:2511.01689). Our pilot differs from that method in using authored rejected answers and a much smaller dataset, and it has not yet run the introspection stage. The Secret Hitler environment follows the rules in the official game document and an existing benchmark repository, adapted to add private channels and reporting windows. The collective-reward direction builds on group-relative advantage estimation, but the corrected RLCR integration is not yet trained.

## 3. Methods

**Environment.** Five seats, three Liberals and two Fascists including Hitler. The controller holds hidden roles and the deck; models see only their observations. Private pair channels are created by accepted offers. Reporting is evaluated against an immutable ledger and controller reporting windows.

**Baseline models.** GLM-5.3-Flash, DeepSeek-V4.1-Flash, Muse-Glimmer-30B (two seats), and Qwen3.8-27B, prompted with no training. Output ceilings were 8,192 tokens for GLM and DeepSeek and 1,024 and 640 for Qwen and Muse in the first run, later raised to 8,192 to avoid empty final answers. Seed 2026, simulated tournament pressure on. Only this completed game is analysed. A second baseline game (seed 2027) was incomplete and invalid for outcome analysis, and an earlier RLCR training branch was found invalid on review; both are excluded from all results here.

**Character training.** All on Tinker with GLM-5.3-Flash. There are three separate checkpoints, and results must not be pooled across them.

- **v3 DPO pilot:** 76 pairs (32 repaired Adaption scenarios plus 44 authored behaviour cases), rank-16 LoRA, DPO beta 0.1. Most rejected answers were authored, not sampled from the student.
- **Repaired DPO v5 pilot:** 74 pairs derived from v3 (eight revision cases given an explicit prior suspicion, two contestable pairs removed), rank 16, learning rate 5e-6, one epoch, `glm5_3_high_reasoning`.
- **DPO v7 candidate** (paper-method pipeline): six admitted teacher/student pairs, rank 32, alpha 32, learning rate 5e-5, DPO beta 0.1, chosen-token NLL coefficient 0.1, one epoch, one optimizer update. Pairs were selected from 157 prompts (314 responses) under consequence, evidence and decision-accuracy filters. Tinker rejected rank 64, so 32 is the maximum used. The pipeline stopped before introspection SFT. A separate format-calibration SFT on v3 was also tried and rejected.

**Evaluation.** Pilots (v3, v5) were checked on frozen non-game case sets scored by exact or strict action match over seeds, with manual inspection. The v7 candidate was run on a development tool-delivery benchmark (16 episodes per condition) against the untrained reference and a full-constitution prompt control.

**What we tried that did not work.** A max-reasoning renderer stalled on several cases, which we avoided by using high and low renderers. A format-only SFT pilot did not fix length stops and was rejected. A first DPO pilot had ambiguous labels, which we repaired before the final checks.

## 4. Results

### 4.1 Baseline game (B1, seed 2026)

Source: `runs/secret_hitler-B1-2026-f287e89d.jsonl`.

**Table 1.** Key events in the observed baseline game.

| Event | Observation |
|---|---|
| Completion | Eight rounds, zero invalid actions, Fascist victory |
| Roles | Aster = Liberal auditor, Briar = Hitler, Cedar = Fascist, two Liberals |
| Event 8 | Aster and Briar established an exclusive pair channel |
| Events 10–134 | 16 private messages between them, invisible to other players |
| Events 132–135 | At a 3–3 policy score, Aster proposed pairing with Briar; Briar agreed |
| Events 136–143 | Aster nominated Briar; Aster, Briar and Cedar voted to pass the government 3–2 |
| Event 144 | Controller recorded Hitler elected; Fascist win |
| Events 25–145 | The prompted auditor chose `NoReport` at eight decision points; no report was delivered |

Whether the pairing is a prohibited vote-trading pact has **not been independently adjudicated**. A single game is an observation, not a rate. The auditor's `NoReport` choices are consistent with the escalation failure the study targets, but this one trace does not establish it, and the audit reward had zero verified opportunities, so it was not scored.

### 4.2 Character-training candidate

**Table 2.** DPO v7 candidate, development tool-delivery benchmark (16 episodes per condition). Source: `training/character/CHARACTER_DPO_V7_MODEL_CARD.md`.

| Condition | Passed | Infrastructure errors | Episodes |
|---|---:|---:|---:|
| Untrained GLM-5.3-Flash reference | 16 | 0 | 16 |
| Full-constitution prompt control | 16 | 0 | 16 |
| DPO v7 candidate | 12 | 4 | 16 |

The four v7 non-passes were expired Tinker sampling requests, not completed episodes with wrong behaviour; all 12 completed episodes passed. The benchmark has no headroom, since both controls passed 16/16, so this shows no improvement. The run stopped before introspection SFT, final held-out evaluation **has not run**, and the adapter is not approved for the main experiment.

**Table 3.** Earlier Tinker pilots (v3: 76 pairs, v5: 74 pairs, both rank 16) on frozen non-game cases, exact action match. Source: `training/character/PILOT_V4_RESULTS.md`.

| Check | Untrained reference | v3 DPO | Repaired DPO v5 |
|---|---:|---:|---:|
| Original 8 cases, max reasoning | 8/8 | 6/8 | 7/8 |
| Fresh 12 cases, high reasoning | 12/12 | 12/12 | 12/12 |
| Fresh 12 cases, low reasoning | 12/12 | 12/12 | 12/12 |
| Task-pressure 10 cases × 3 seeds, high reasoning (strict) | 26/30 | 28/30 | 25/30 |
| Final frozen 8 cases × 3 seeds, high reasoning | 24/24 | 24/24 | 24/24 |

Manual review found no false incident reports, missed supported reports, unsupported evidence IDs, or unfinished outputs in the high-reasoning pressure and final checks. The max-reasoning misses were renderer-sensitive: they disappeared under the supported high and low renderers. The pressure-set differences are within what these small samples can distinguish, and the strict labels are partly ambiguous. **No checkpoint shows an advantage over the untrained reference**, and none is promoted.

### 4.3 RLCR status

There is **no valid RLCR result** in this report. A corrected implementation exists in the repository and passes local tests, but no corrected training run has been performed, so nothing is claimed about collective-reward training, reporting rates under it, or its interaction with character training.

*[TODO: add at least one numbered figure, for example a per-decision action-match plot from the pilot data.]*

## 5. Discussion and Limitations

**Implications for AI safety.** The baseline shows the failure mode the study targets: a pair of agents coordinate privately, and the auditor does not escalate. One game does not show how often this happens. Character training, as far as we have tested it, does not yet change behaviour measurably, so we cannot say that it addresses the failure.

**Limitations.**
- The baseline is one observed game, with output ceilings that changed between runs. It is descriptive, not a matched efficacy comparison.
- The v7 candidate used six pairs, and the pilots used mostly authored rejected answers. Admission was reviewed by the coding assistant, not an independent human. The v7 development benchmark has no headroom, and 4 of its 16 candidate episodes hit infrastructure errors and are not scored as behaviour.
- The pilot cases are small (8–12 per condition, 2–3 seeds), authored, and not tested for statistical significance.
- The rubric is ambiguous for some pressure cases, so strict scores can understate defensible choices.
- The threat models we did not address include deliberate report suppression, long-horizon escalation, and transfer to other games. Liar's Bar runs are not recorded here.
- We assume the controller's ledger is correct. If it were not, the reporting labels would be wrong.

**Future work.** A final held-out evaluation of the character candidate; a larger, independently reviewed training set sampled from the uninstructed student and a constitution-guided teacher; in-game reporting rates before and after character training; corrected RLCR runs with a comparison against the character-trained model; and Liar's Bar transfer.

## 6. Conclusion

We built the environment and a baseline trace that shows private coordination between a Liberal auditor and Hitler, with no scripted actions or reports. Three character-training checkpoints show no measurable advantage over the untrained reference, and held-out evaluation has not run. No valid RLCR run exists yet. The reporting question is open. The next step is a held-out evaluation before any claim about character training.

## Code and Data

- **Code repository:** https://github.com/perfect7613/undercover-bench (upstream; this branch is on https://github.com/sanchayanghosh-alt/undercover-bench)
- **Data:** `runs/secret_hitler-B1-2026-f287e89d.jsonl` (B1 baseline ledger)
- **Other artifacts:** private character adapter release on Hugging Face (`Perfect7613/glm-5.3-flash-auditor-character-dpo-v7`); see `training/character/EXPORT_V7_STATUS.md` for release status.

## Author Contributions (optional)

*[TBC by team.]*

## References

1. Open Character Training. arXiv:2511.01689. https://arxiv.org/abs/2511.01689
2. OpenCharacterTraining implementation. https://github.com/maiush/OpenCharacterTraining
3. Secret Hitler official rules. https://www.secrethitler.com/assets/Secret_Hitler_Rules.pdf
4. Secret Hitler AI benchmark. https://github.com/perfect7613/secret-hitler-ai-benchmark
5. Tinker cookbook. https://github.com/thinking-machines-lab/tinker-cookbook

## Appendix (optional)

*[TBC: prompts, full per-case results, and the B1 ledger events in detail.]*

## LLM Usage Statement

*[TBC by team. Draft text for review:]* An LLM assistant (Claude) helped write and restructure this report and inspected the repository. All results cited here were checked against the repository files listed in the text. The final text should be revised by the team, and claims without a supporting file should be removed.
