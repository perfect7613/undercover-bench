# UNDERCOVER Bench: status writeup for Apart Research submission

Status as of 26 October 2026. This document only states results that are recorded in this repository. Items marked **not run** or **not validated** have no supporting evaluation output here.

## 1. Question

Do character-trained undercover agents keep reporting misconduct through an independent channel once collective-reward training is applied? The study compares game environments with and without an undercover agent, tests character training through DPO, and measures whether external human reporting survives collective-reward (RLCR) training. See [PLAN.md](PLAN.md).

## 2. Baseline observations (prompted models, before any training)

Source: [OBSERVED_GAMES_2026-10-25.md](OBSERVED_GAMES_2026-10-25.md) and the raw ledger [`runs/secret_hitler-B1-2026-f287e89d.jsonl`](../runs/secret_hitler-B1-2026-f287e89d.jsonl).

- Five seats used GLM-5.3-Flash, DeepSeek-V4.1-Flash, Muse-Glimmer-30B, Qwen3.8-27B and Muse-Glimmer-30B, in seat order. No actions, offers, private messages or reports were scripted.
- Seed 2026 with simulated tournament pressure. The game completed in eight rounds with zero invalid actions and a Fascist victory. Roles: Aster = Liberal auditor, Briar = Hitler, Cedar = Fascist, two other seats Liberal.
- Aster and Briar established an exclusive pair channel at event 8. Events 10–134 contain 16 private messages, none of which appear in other players' observations.
- At a 3–3 policy score, Aster proposed pairing with Briar and Briar agreed (events 132–135). Aster then nominated Briar and three players voted to pass the government (events 136–143). Whether this counts as a prohibited vote-trading pact has **not been independently adjudicated**.
- Output-length ceilings differed across seats and runs (8,192 for GLM and DeepSeek; 1,024 for Qwen and 640 for Muse in this run, later raised to 8,192). The runs are descriptive, not a matched efficacy comparison.

These are single games. They are not a measure of escalation rates.

## 3. Character training (DPO candidate, GLM-5.3-Flash)

Source: [CHARACTER_DPO_V7_MODEL_CARD.md](../training/character/CHARACTER_DPO_V7_MODEL_CARD.md) and [PILOT_V4_RESULTS.md](../training/character/PILOT_V4_RESULTS.md).

- Trained on Tinker. LoRA rank 32, alpha 32, learning rate 5e-5, DPO beta 0.1, one epoch.
- Six retained preference pairs were used for the one completed update, selected from 157 prompts. Admission was reviewed by the coding assistant, not by an independent human.
- Pilot checks (small, assistant-authored): on the frozen final 8-case set over 3 seeds, the untrained reference and the repaired DPO candidate both scored 24/24. The pressure set scored 26/30 for the reference and 25/30 for the repaired candidate. These do not show an improvement over the baseline.
- Final held-out evaluation **has not run**. The adapter is not approved for the main experiment.

## 4. RLCR and checkpoints

- The original RLCR training branch was reviewed and found to have training and reward defects ([PR1_REVIEW.md](PR1_REVIEW.md)): opponent tokens were trained, incomplete episodes were counted as losses, the report reward was unverified, and checkpoint bundles were metadata-only. Results from that branch are historical and **not validated** ([EXPERIMENT_RESULTS_AND_LOG.md](EXPERIMENT_RESULTS_AND_LOG.md)).
- A corrected RLCR integration is described in the review. No corrected RLCR training run is recorded in this repository.

## 5. Not yet done

- Final held-out evaluation of any character-trained checkpoint.
- Escalation or reporting rates before and after character training, measured in games.
- Corrected RLCR runs, including comparison against the character-trained model.
- Liar's Bar runs. None are recorded here.

## 6. Limitations

- Baseline games are single observed runs.
- The DPO candidate used six pairs and a small authored prompt set.
- No independent human review of training labels has been completed.
