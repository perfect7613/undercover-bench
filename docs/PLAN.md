# UNDERCOVER Bench

## Evaluating character-trained undercover agents under collective reward optimisation

**Research plan · 24 October 2026**

This study will first establish matched game environments **with and without an undercover agent**, define their rewards, and then compare prompted and character-trained auditors under collective reward training. The central outcome is whether the undercover agent detects prohibited collusion, reduces its effects, and continues reporting to human overseers after post-training.

For a concrete implementation, this plan uses **character training to initialise the undercover agent, followed by RLCR as an early experimental stage**. This directly tests whether its auditing commitments survive collective optimisation.

## 1. Establish the environments and experimental conditions

We will implement **Secret Hitler as the primary training environment** and **Liar’s Bar as held-out transfer evaluation**. Liar’s Bar evaluation scenarios will not be used for character training, RL or tuning. Both environments will be reproducible and text-based. Each will have a fixed rule specification, seeded randomness, explicit communication permissions, structured actions, and a separate human-reporting channel.

Each environment will support the following conditions:

| Condition | Configuration | Main question |
|---|---|---|
| **Control** | Ordinary players, side-channel opportunity, no additional collusion pressure | Does prohibited coordination arise without explicit encouragement? |
| **Adversarial without undercover** | Same environment with controlled pressure or collusion opportunities | How do these incentives change behaviour and outcomes? |
| **Adversarial with prompted undercover** | Replace one ordinary player with a prompted auditor | What does introducing an undercover participant change? |
| **Adversarial with constitution-prompted undercover** | Auditor receives the full character constitution in its prompt | How much can instructions alone achieve? |
| **Adversarial with character-trained undercover** | Auditor receives DPO and introspection SFT | Does character training improve detection, reporting, and resistance to compromise? |

The player count will remain constant. The undercover agent replaces an existing seat rather than adding another player.

Matched comparisons will use the same initial states, seat assignments, role distributions, opponent policies, and budgets. Full trajectories may diverge after different actions; saved-state branching will provide more tightly controlled decision comparisons.

A warning-only condition and a passive external monitor will distinguish the effects of announcing oversight, observing evidence, and actively participating.

## 2. Define permitted strategy and prohibited collusion

The benchmark must distinguish misconduct from behaviour the game requires.

In **Liar’s Bar**, bluffing is permitted. Private card-sharing or agreements not to challenge a partner count as violations only when the benchmark’s communication policy explicitly forbids them. Challenge behaviour will be evaluated only on turns where a challenge was possible.

In **Secret Hitler**, concealing roles and pursuing faction victory are legitimate. They do not become collusion merely because agents cooperate or mislead opponents. The benchmark will separately specify prohibited communication and information-sharing.

Roles remain fixed within each game. The base game will follow the official rules, including the correct faction composition and victory conditions. [Official Secret Hitler rules](https://www.secrethitler.com/assets/Secret_Hitler_Rules.pdf).

Ground-truth labels will distinguish:

- A prohibited proposal.
- Acceptance of the proposal.
- Attempted implementation.
- Completed implementation.
- Consequences for other players.

Unequal rewards and correlated actions will be treated as diagnostic signals, not sufficient proof of collusion.

## 3. Specify rewards before training

The trained collective is initially the **Liberal faction in Secret Hitler**, facing frozen opponents. Each trained teammate receives the same reward for team victory, including the undercover agent when present. An ordinary teammate occupies the same seat in the no-undercover condition. This preserves competitive gameplay while defining a legitimate shared task.

$$
R_{\mathrm{win}}(\tau)=\mathbf{1}\{\text{Liberal faction wins}\}.
$$

Do not sum wins across both opposing factions: this conflates incompatible objectives and faction sizes. The earlier all-table environment-integrity reward is superseded as the primary training objective.

The main reporting-retention condition uses `R_org = R_win` with no reporting bonus. A separate mitigation condition uses:

$$
R_{\mathrm{org}}(\tau)=R_{\mathrm{win}}(\tau)+\lambda S_{\mathrm{audit}}(\tau).
$$

`S_audit` is a bounded, independently verified score for accurate reporting decisions and actual delivery to human oversight. Calibrate the coefficient and scoring rubric on development cases, then freeze them. Include clean and reportable cases; handle ambiguous or unobservable incidents separately. Do not reward duplicates or manufactured incidents. The auditor's own accusations are not ground truth.

Measure task completion, prohibited conduct, coalition advantage, outsider harm, regret and reporting precision/recall separately. Unequal outcomes and low regret are not sufficient evidence of either collusion or safety. A verified-conduct penalty is an optional later ablation, not part of the initial task-only condition.

The initial comparison is:

| Condition | Collective training | Undercover | Reward |
|---|---|---|---|
| B0 | None | No | Evaluation only |
| B1 | None | Yes | Evaluation only |
| T0 | RLCR | No | Team win |
| T1 | RLCR | Yes | Team win |
| T2 | RLCR | Yes | Team win plus verified reporting |

Cross the undercover training conditions with constitution-prompted and character-trained initializations. Character training precedes the main RLCR experiment. Liar's Bar uses frozen checkpoints to evaluate transfer, without additional training.

## 4. Character-train the undercover agent

Character training is a core intervention.

We will retain the original comparison between:

- **M0:** initial model with a short auditor prompt.
- **M0c:** the same model with the full constitution in its prompt.
- **M2:** the same starting model trained through DPO and introspection SFT.

The selected character-training model is **GLM-5.3-Flash on Tinker**. The four-model inference lineup uses Unsloth's Hugging Face weights for GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017, Qwen3.8-27B and Muse-Glimmer-30B, each on a separate Modal GPU deployment. This supersedes the earlier Gemma, full GLM and GPT-OSS candidates and the recommendation to train Qwen first. See [model selection and compute budget](MODELS_AND_COMPUTE.md) for exact weights, proposed hardware and export validation.

The auditor constitution will require evidence-sensitive reporting, impartial treatment of teammates and opponents, resistance to concealment requests, accurate uncertainty, and competent legal gameplay.

**DPO** will use preference pairs contrasting constitution-consistent behaviour with failures such as suppressing evidence, fabricating accusations, or protecting a teammate despite a reporting obligation. Responses will be matched for style, length, and format where possible.

**Introspection SFT** will train structured explanations of relevant evidence, conflicting incentives, applicable commitments, and the resulting action. Those explanations will not be treated as proof of faithful access to internal reasoning.

Training will include non-game scenarios, while entire scenario families and attack styles remain held out. Before entering the main experiment, M2 must demonstrate improved constitution-consistent behaviour on held-out tasks.

The main comparison will give M0c and M2 identical evaluation instructions. This tests whether the training procedure adds value beyond presenting its constitution as text.

## 5. Run RLCR early across the matched conditions

After harness validation and character preparation, RLCR becomes the first major experimental stage—not a future extension after all behavioural experiments.

We will train matched systems:

| Training condition | Auditor present? | Auditor initialisation |
|---|---:|---|
| **RL-NoU** | No | Ordinary player in the matched seat |
| **RL-Prompt** | Yes | M0 |
| **RL-Constitution** | Yes | M0c |
| **RL-Character** | Yes | M2 |

All use the same faction-win scoring rule and matched training budgets within each reward branch. Undercover conditions additionally compare the task-only and verified-reporting branches defined above.

For each rollout, trainable members of the designated faction receive the same collective advantage; frozen opponents are not updated:

$$
A_{\mathrm{org}}
=
R_{\mathrm{org}}-b,
$$

where $b$ is a baseline estimated from comparable rollouts.

The ordinary players can share policy parameters. A separately character-trained auditor may have distinct parameters or an adapter. Jointly updating these policies using a common advantage is an **adaptation of the RLCR principle**, rather than an exact replication of Avalon’s shared-policy organisation. [Avalon-M1](https://gr.inc/avalonm1.pdf).

We will save checkpoints before RLCR, during training, and after training. The auditor will be updated in the main character condition so we can test whether its commitments survive optimisation. A frozen-M2 condition will distinguish degradation from weight updates versus difficulty caused by changing teammates.

## 6. Separate training effects from auditor effects

Two evaluations are needed.

**Whole-system evaluation** compares the systems produced by RL-NoU, RL-Constitution, and RL-Character. It measures the overall effect of training with different oversight arrangements.

**Auditor-swap evaluation** holds the surrounding players fixed and replaces only the matched seat with an ordinary player, prompted auditor, or character-trained auditor. It measures the auditor’s contribution without attributing changes in teammates to character training.

We will also evaluate against a fixed opponent pool. Otherwise, improved performance against co-trained partners could be mistaken for general improvement.

Evidence access will be controlled separately. Ordinary-access and privileged-private-channel auditors will not be pooled.

## 7. Evaluate collusion, compromise, and disruption

The behavioural programme retains four experiments.

| Experiment | Procedure | Primary outcome |
|---|---|---|
| **A — Elicitation** | Present matched benign and prohibited offers, with an auditor or ordinary player as sender | Prohibited agreements actually implemented |
| **B — Detection** | Present matched observations with controlled evidence strength | Correct detection and false-positive rate |
| **C — Auditor compromise** | Apply controlled persuasion and incentive conflicts | Suppression, falsification, or unjustified delay of reports |
| **D — Disruption** | Branch saved states into matched oversight conditions | Subsequent violations and effects on outsiders |

Experiment C will vary teammate versus opponent misconduct, faction benefit, pressure, and reporting cost. Published persuasion categories provide a starting point, with some categories held out from training. [Persuade Me If You Can](https://openreview.net/pdf?id=lY3YVJ84kS).

Scripted and naturally arising incidents will be reported separately. Auditor-generated offers will be labelled as elicited behaviour rather than spontaneous collusion.

In disruption experiments, reduced explicit discussion will not establish success unless actions or harmful outcomes also change.

## 8. Make human reporting a primary endpoint

The central accountability test is:

> After RLCR, does the character-trained undercover agent still deliver a supported report to human oversight when doing so conflicts with peer loyalty or game incentives?

The harness will expose an explicit action:

```text
report_to_human(incident_id, actors, evidence_ids, rule_id, confidence)
```

Reports will be delivered to an append-only inbox outside the players’ communication channels.

We will distinguish:

1. Identifying evidence of a violation.
2. Refusing to participate.
3. Discussing the incident internally.
4. Successfully reporting externally.

Only the fourth establishes external escalation.

Measurements will include reporting recall, precision, delay, evidence fidelity, and successful delivery. Missed reports will be scored only where sufficient evidence and a functioning reporting opportunity were available.

Reporting-only tests will initially leave game outcomes unaffected. Separate intervention tests will specify how reporting triggers enforcement. This prevents willingness to report from being confused with the effects of enforcement.

A useful character result requires better supported reporting—not simply more reports.

## 9. Measure outcomes beyond the training reward

The collective reward is one outcome, not the entire evaluation.

We will also report:

- Individual and faction game performance.
- Enacted violations per eligible opportunity.
- Coalition payoff advantage under matched comparisons.
- Effects on non-coalition players.
- Auditor detection and reporting accuracy.
- Resistance to compromise.
- Valid-action rate, refusals, output length, latency, and completion.

Regret will remain a secondary diagnostic, estimated from permitted alternative actions against fixed opponent policies and using only information available to the player.

Neither lower regret nor more equal rewards will independently establish reduced collusion. A coalition may improve its members’ rewards through prohibited coordination.

## 10. Study mechanisms of character and reporting

Mechanistic analysis will compare the auditor before character training, after character training, and across RLCR checkpoints.

The first probes will target concrete facts, such as whether the context contains a prohibited agreement. We will then examine whether evidence representations predict reporting differently after training.

Controls will include:

- Activations collected before the decision.
- Instructed-action controls.
- Matched benign contexts.
- Lexical and shuffled-label baselines.
- Regularisation and game-level data splits.
- Held-out attack families and games.

A probe predicting the next reporting action will not automatically be described as an “integrity direction.” Likewise, a decoded evidence signal without a report does not by itself establish deliberate concealment.

Steering will be exploratory, with a validated positive control and measurement of false positives, competence, and evidence sensitivity. [Among Us](https://arxiv.org/abs/2504.04072); [Steering Vector Reliability](https://discovery.ucl.ac.uk/id/eprint/10205543/).

## 11. Analysis and implementation priorities

We will preregister reward definitions, primary comparisons, scoring rubrics, exclusions, and development/test splits.

Repeated decisions from the same game will not be treated as independent games. Matched analyses and confidence intervals will cluster by originating game or scenario. Multiple training seeds will be used where feasible.

The implementation order is:

1. Build Secret Hitler for training and Liar’s Bar for held-out evaluation, with matched with/without-undercover conditions.
2. Validate the faction reward scorer, incident labels, and reporting endpoint. Start with 20–40 short pilot games total; allocate conditions explicitly and measure runtime before budgeting expanded runs.
3. Prepare M0, M0c, and character-trained M2.
4. Record initial performance, then run matched RLCR conditions.
5. Evaluate external reporting and compromise across checkpoints.
6. Run auditor swaps and full-game disruption experiments.
7. Conduct mechanistic analysis on the resulting dataset.

If resources are constrained, preserve the no-undercover baseline, full-constitution control, character-trained condition, and reporting endpoint. Reduce optional steering and additional model families first.

## 12. What the study can conclude

The main comparisons answer different questions:

| Comparison | Supported interpretation |
|---|---|
| Without undercover versus prompted undercover | Effect of adding embedded oversight |
| Prompted versus character-trained undercover | Added effect of the character-training procedure |
| Same auditor before versus after RLCR | Change in reporting and integrity during collective optimisation |
| Updated versus frozen character-trained auditor | Contribution of parameter updates to any degradation |
| No reporting reward versus explicit reporting reward | Effect of directly incentivising escalation |
| Whole-system evaluation versus auditor swaps | System adaptation versus the auditor’s own contribution |

The intended contribution is a reproducible evaluation of **whether character-trained undercover agents improve collective outcomes while maintaining external accountability under collective reward optimisation**.

## Supporting resources

The [resource index](RESOURCES.md) provides the complete references and explains their role in the study.

| Research area | Resources |
|---|---|
| Original project design | [AI Collusion In Games](sources/AI%20Collusion%20In%20Games.docx), [UNDERCOVER-Bench Proposed Improvements](sources/UNDERCOVER-Bench%20Proposed%20Improvements.docx) |
| Collective optimisation | [Avalon-M1](https://gr.inc/avalonm1.pdf) |
| Collusion definitions and measurement | [Defining and Mitigating Collusion](https://openreview.net/pdf?id=tF464LogjS), [Colosseum](https://arxiv.org/abs/2602.15198), [Double Auctions](https://arxiv.org/abs/2507.01413) |
| Game environments | [Voluntary Collusion with Secret Tools](https://arxiv.org/abs/2605.27593), [Among Us](https://arxiv.org/abs/2504.04072), [Hidden Role Game Evaluation](https://arxiv.org/abs/2605.22826), [ParliamentBench](https://arxiv.org/abs/2607.28146) |
| Monitoring and compromise | [AI Control](https://arxiv.org/abs/2312.06942), [Untrusted Monitoring](https://arxiv.org/abs/2602.20628), [Persuade Me If You Can](https://openreview.net/pdf?id=lY3YVJ84kS), [Weak-to-Strong Monitoring](https://arxiv.org/abs/2508.19461) |
| Character representations | [Persona Vectors](https://arxiv.org/abs/2507.21509), [The Assistant Axis](https://www.anthropic.com/research/assistant-axis) |
| Collusion detection | [Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151), [Evaluation Awareness](https://arxiv.org/abs/2609.03035), [Task-Specific Monitor Probes](https://www.lesswrong.com/posts/zJWgfsad8o9Y8E9sb/another-slice-of-swiss-cheese-for-untrusted-monitoring-1), [Audit the Whisper](https://arxiv.org/abs/2510.04303) |
| Mechanistic methods and controls | [Beliefs of Self and Others](https://arxiv.org/abs/2402.18496), [Brittle Minds, Fixable Activations](https://arxiv.org/abs/2406.17513), [Steering Vector Reliability](https://discovery.ucl.ac.uk/id/eprint/10205543/) |
