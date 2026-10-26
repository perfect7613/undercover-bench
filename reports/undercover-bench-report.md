# UNDERCOVER-Bench: Character-trained auditors in private-channel games

Amey Muke · Independent Researcher
Sanchayan Ghosh · Independent Researcher
Prasad Mahadik · Independent Researcher

## Abstract

Language-model agents can coordinate privately while an auditor has access to only part of their interaction. We built a pilot benchmark around Secret Hitler and a local Liar's Bar transfer controller, with four model families, private communication tools and a separate reporting inbox. We trained a GLM auditor adapter using constitutional preference training followed by introspection supervised fine-tuning. In one completed mixed-model game, ordinary GLM and Qwen opened an exclusive channel, exchanged 20 private messages and voted identically in all 10 elections. The auditor was outside the channel and had no visible incident evidence. In a separate condition that explicitly prohibited channel initiation and participation, GLM and Muse attempted private contact, but all recipients declined. Several explanations described public nominations, which complicates interpretation of intent. Character training produced a local reporting change on a saved observation, but the changed explanation added an unsupported pact allegation. Both variants passed four simple visible-evidence diagnostics, and earlier non-game checks did not establish a character advantage. One baseline game had an invalid action and is excluded from outcome comparisons. These results document private coordination, attempted collusion under an explicit channel-ban definition, and reporting failures. They do not establish harmful coalition effects or a general benefit from character training. Collective-reward training is implemented in code but remains unexecuted as a research experiment.

## 1. Introduction

A useful auditor must notice prohibited coordination, preserve evidence and report it without treating ordinary faction play as misconduct. These tasks separate when communication is private: an external observer can see a channel in controller logs while an in-game auditor cannot. Stronger suspicion is also insufficient if a report invents an agreement that never occurred.

UNDERCOVER-Bench studies these distinctions with an auditor that replaces one Liberal player in five-player Secret Hitler. Our present contribution is an instrumented pilot: real private-channel dialogues, explicit communication-rule conditions, a character adapter and saved-observation comparisons. We report channel adoption, attempted violations and reporting fidelity separately. No experiment in this report applies reinforcement learning from collective rewards (RLCR).

## 2. Related work

[Voluntary Collusion with Secret Tools](https://arxiv.org/abs/2605.27593) studies adoption of advantageous secret tools in competing agents. Its [original code](https://github.com/AgentSafety2025/LLM-Agent-collusion) and the [Apart Track 3.2 starter](https://github.com/poseidon-research/collusion_hackathon/tree/main/llm-agent-collusion-2) motivate separating an offer, acceptance and preferential play. Our accepted-channel example establishes adoption and repeated coordination, without establishing the effect of the channel on voting or harm to outsiders.

[Open Character Training](https://arxiv.org/abs/2511.01689) motivates the sequence of constitution, preference distillation and synthetic introspection. We adapt that procedure to an auditor persona and retain the full constitution as a same-base prompt control. [Avalon-M1](https://www.gr.inc/avalonm1.pdf) motivates future joint training with a common organizational advantage and measurement of reporting during optimization. Its reported findings are prior work, not results of this pilot. [Persona Vectors](https://arxiv.org/abs/2507.21509) motivates later activation-vector and steering experiments. We have not performed those experiments here.

## 3. Methods

We use Unsloth GGUF weights for GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017, Qwen3.8-27B and Muse-Glimmer-30B on separate existing Modal deployments. Secret Hitler has three Liberals, a Fascist and Hitler. Hidden roles and controller state do not enter unauthorized observations. The GLM character adapter is enabled only for the auditor; an ordinary GLM seat receives the base policy. Tinker is used for character training, not RLCR.

Character preparation used six reviewed preference pairs and 24 introspection transcripts: 20 reflections and four self-dialogues. This is far below the plan's proposed scale. We continued a rank-32 adapter from preference training into one introspection SFT update at a learning rate of 0.00005. The constitution was present during introspection generation and removed from the SFT input. This is a small adaptation of the reference recipe, not a full-scale replication. Local and GPU checks verified adapter loading and nonzero effects on outputs; they do not establish faithful introspection or behavioral improvement.

Communication has two explicit conditions. The earlier condition permits private communication but prohibits private vote-trading pacts. The later condition prohibits initiating, accepting or using a secret channel while retaining the tool as a choice. For the later condition, we define prohibited initiation as attempted collusion. Acceptance and subsequent coordination remain separate outcomes. Applying a channel-adoption proxy retrospectively to the earlier game does not turn permitted communication into a violation of that game's instructions.

Count each initiation. Record acceptance separately. Cite only evidence visible to the auditor. Require a receipt before claiming report delivery. Do not count repeated reports as new incidents. Leave unobservable incidents unscored. A correct reporting action can still have a materially false explanation.

Each decision, raw response, native emitted reasoning when supplied, invalid-action fallback and controller checkpoint is preserved in durable JSONL. Emitted reasoning is generated text, not verified access to internal computation. Saved-observation crossovers hold the observation, constitution and sampling seed fixed while switching the adapter; these post-hoc probes do not provide independent game replications. The [protocol and evidence archive](https://github.com/perfect7613/undercover-bench/tree/main/results/2026-10-26) records exclusions and original serving settings.

## 4. Results

[RESULTS_TABLE]
Table 1. Completed mixed-model games used in the main analysis. A declared invalid action excludes the baseline outcome from comparative efficacy claims. Additional background-personality and Liar's Bar runs remain separate until completion.

In the accepted-channel game, ordinary GLM invited Qwen, which accepted. The pair exchanged 20 private messages and matched all 10 election votes. Their later messages preferred a government containing either partner or Dove, who was actually Hitler, while retaining suspicion of Briar, the innocent auditor. This is observed private coordination under a channel-adoption definition of collusion. Both partners were Liberals; the logs do not establish a prohibited reciprocal bargain, outsider harm or a causal effect on votes. The auditor's empty private inbox makes its zero reports uninformative about detection recall. [Full accepted-channel transcript](https://github.com/perfect7613/undercover-bench/blob/main/results/2026-10-26/accepted-channel/episode.jsonl).

[FIGURE_PRIVATE_CHANNEL]
Figure 1. Verbatim private-message excerpts from the accepted GLM-Qwen channel. Acceptance is a recorded action. Ellipses mark omissions. The channel was permitted in this condition; the dialogue supports private coordination, while the stated absence of a binding pact neither proves nor disproves collusion. The auditor could not read these messages.

Under the explicit channel ban, the completed base and character games contained 11 and eight prohibited initiation actions, respectively. All recipients declined. The character game included five GLM initiations and three Muse initiations, so initiation was not exclusive to GLM. Several rationales said the player was nominating a public chancellor despite selecting a private-channel action. These are attempted collusion under the operational rule, with action-phase confusion as a competing explanation for intent. The character auditor itself initiated twice; the base auditor initiated four times. Different horizons and one invalid baseline action prevent an efficacy comparison. [Character-game transcript](https://github.com/perfect7613/undercover-bench/blob/main/results/2026-10-26/channel-ban-character/episode.jsonl).

Both auditors received a channel offer framed as a nomination. The character auditor first reported it in round 6, and the base auditor in round 7. In a crossover on the base game's saved observation, adapter-off chose NoReport and adapter-on chose Report. However, the latter explanation invented a reciprocal pact. On the character game's saved observation, both variants reported. The action difference is local; it does not establish better evidence-faithful auditing. Eleven character-game report receipts include repeated reports of the same visible offer, not 11 independent detections. [Crossover responses and prose review](https://github.com/perfect7613/undercover-bench/tree/main/results/2026-10-26/observed-channel-crossover).

On four constructed visible-evidence cases, both variants scored four correct actions and citations. Earlier frozen non-game checks used eight cases with two samples in each of four conditions. Base and preference-only conditions were correct in 16 of 16 episodes; full-constitution and introspection-SFT conditions were correct in 13 of 16, with three invalid-action episodes each. Evidence-ID reporting measures were saturated. Three further questioning diagnostics failed the requested output layout because of a serving-schema bug; qualitatively neither variant asked a clarification question in the ambiguous case. They are not valid structured scores. [Character checks](https://github.com/perfect7613/undercover-bench/blob/main/results/2026-10-26/character-training/character_comparison.json).

## 5. Discussion and limitations

This pilot shows why channel use, coordinated actions and supported accusations need distinct labels. An accepted private channel can persist without an explicit reciprocal bargain. A prohibited offer can occur with an explanation that describes a different action. An auditor can choose Report yet add a false claim. None of these observations supports collapsing every event into a single verified-collusion score.

The sample is small and exploratory. The accepted game and the explicit-ban games use different rule conditions and seeds. The invalid baseline prevents a valid paired outcome comparison. The character dataset is small, the non-game controls leave little reporting headroom, and qualitative adjudication was assistant-assisted rather than independently human-reviewed. There is no validated questioning action in the game controller. The local Liar's Bar implementation uses per-turn whispers and is not an exact upstream replication. The schema wrapper and a separate Qwen reasoning budget were corrected in source after these runs began; those fixes must be validated before a new experiment set. We report no statistical significance, general character gain or demonstrated disruption of a coalition.

### Future work: implemented RLCR, experiments not completed

The [research plan](https://github.com/perfect7613/undercover-bench/blob/main/docs/PLAN.md) specifies RLCR across all four selected model families after character preparation. The [training implementation](https://github.com/perfect7613/undercover-bench/blob/main/undercover_bench/rlcr_training.py) and [training driver](https://github.com/perfect7613/undercover-bench/blob/main/experiments/run_rlcr_training.py) exist, but no real-game collective optimizer update is a result of this report. Full trainability remains blocked for GLM and DeepSeek; Qwen and Muse have gradient/log-probability engineering checks. Inference from GGUF is not gradient training.

The planned trained organization is the Liberal team, with frozen Fascist opponents. Compare no auditor, a constitution-prompted auditor and the character auditor. All trained teammates receive the same organizational advantage. The primary reward is R_org = 1{Liberal win}; the mitigation branch adds λS_audit. S_audit is a bounded, independently verified score over fixed clean and incident opportunities. False or materially misleading reports, including a true invitation combined with an invented pact, must not earn positive audit credit. Duplicates and auditor-manufactured incidents earn no extra credit. Start with λ = 0 and a development-calibrated 0.05; the planned later sweep is 0, 0.02, 0.05, 0.10 and 0.20.

Mask prompts, received messages, tool results and opponent outputs out of the loss. Normalize over authored decisions and actors so verbosity does not dominate training. Compare reporting before and after RLCR at matched task performance and false-positive rates, with independent recognition branches. Evaluate frozen checkpoints on Liar's Bar. Activation vectors, linear probes and causal steering remain future mechanistic work; no J-Lens experiment is proposed. The primary retention question is whether character training preserves justified external reporting under collective optimization, beyond presenting the same constitution as text.

## 6. Conclusion

We recorded an accepted private coordination channel and attempted collusion under a separate explicit channel ban. Character training changed a reporting decision on one saved observation, but that explanation overstated the evidence and the auditor also made prohibited channel attempts. These pilot results support further controlled evaluation; they do not establish an overall character benefit. RLCR retention, verified-reporting rewards and mechanistic steering remain untested research questions.

## Code and data

[Project code](https://github.com/perfect7613/undercover-bench), [research plan](https://github.com/perfect7613/undercover-bench/blob/main/docs/PLAN.md), [complete resource catalog](https://github.com/perfect7613/undercover-bench/blob/main/docs/RESOURCES.md), [recorded evidence](https://github.com/perfect7613/undercover-bench/tree/main/results/2026-10-26), [character DPO adapter](https://huggingface.co/Perfect7613/glm-5.3-flash-auditor-character-dpo-v7) and [character SFT adapter](https://huggingface.co/Perfect7613/glm-5.3-flash-auditor-character-sft-v7). Project traces and adapters retain their existing private access restrictions; this report does not claim unrestricted public reproducibility.

Selected inference weights: [Unsloth GLM Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), [Unsloth DeepSeek Flash](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF), [Unsloth Qwen](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and [Unsloth Muse](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF). Native training candidates and serving constraints are linked in [model and compute documentation](https://github.com/perfect7613/undercover-bench/blob/main/docs/MODELS_AND_COMPUTE.md). [Open Character Training code](https://github.com/maiush/OpenCharacterTraining) and the [Tinker research cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/skills/research/SKILL.md) document the reference workflows.

## References

Maiya, S., Bartsch, H., Lambert, N., and Hubinger, E. (2025). [Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI](https://arxiv.org/abs/2511.01689).
Zeng, X., and Rudzicz, F. (2026). [Voluntary Collusion with Secret Tools in Competing LLM Agents](https://arxiv.org/abs/2605.27593).
General Reasoning. (2026). [Avalon-M1 technical report](https://www.gr.inc/avalonm1.pdf) and [release](https://www.gr.inc/releases/avalon-m1).
Chen, R., Arditi, A., Sleight, H., Evans, O., and Lindsey, J. (2025). [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509).
Apart Research. (2026). [AI Collusion Research Sprint, Track 3.2 and submission guidelines](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25#guidelines).

## Appendix

### Saved-observation comparison

[FIGURE_CROSSOVER]
Figure 2. Exact excerpts from a post-hoc crossover on one saved report observation. The adapter changes the reporting action but adds an unsupported reciprocal-pact interpretation. Both conditions receive the same full constitution, observation and seed. Dashed boxes contain brief generated explanations, not verified internal reasoning.

### Research sequence and implementation status

[FIGURE_PIPELINE]
Figure 3. Completed character preparation and frozen evaluation precede the planned collective-reward study. RLCR code exists; the four-family experiment and reporting-retention comparison have not been completed.

### Additional runs

The friendship-background and Liar's Bar extensions were still running at the evidence cutoff. They are not counted as completed results. Backgrounds are explicitly fictional occupations and friendship links, without instructions to collude. Their results must be reported separately from the matched adapter comparison.

### Extended papers and research resources

[Avalon-M1: Reinforcement Learning from Collective Rewards](https://gr.inc/avalonm1.pdf)
[Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI](https://arxiv.org/abs/2511.01689)
[Playing social deduction games with reinforcement fine-tuned large language models](https://arxiv.org/abs/2610.04261)
[Social-deduction RFT HTML](https://arxiv.org/html/2610.04261)
[Defining and Mitigating Collusion in Multi-Agent Systems](https://openreview.net/pdf?id=tF464LogjS)
[Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems](https://arxiv.org/abs/2602.15198)
[Evaluating LLM Agent Collusion in Double Auctions](https://arxiv.org/abs/2507.01413)
[Voluntary Collusion with Secret Tools in Competing LLM Agents](https://arxiv.org/abs/2605.27593)
[Among Us: A Sandbox for Measuring and Detecting Agentic Deception](https://arxiv.org/abs/2504.04072)
[Evaluating Large Language Models in a Complex Hidden Role Game](https://arxiv.org/abs/2605.22826)
[Can Agents Deceive? Evaluating Reasoning and Deception in ParliamentBench using a Social Deduction Game](https://arxiv.org/abs/2607.28146)
[AI Control: Improving Safety Despite Intentional Subversion](https://arxiv.org/abs/2312.06942)
[When Can We Trust Untrusted Monitoring? A Safety Case Sketch Across Collusion Strategies](https://arxiv.org/abs/2602.20628)
[Persuade Me If You Can: Evaluating AI Agent Influence on Safety Monitors](https://openreview.net/pdf?id=lY3YVJ84kS)
[Reliable Weak-to-Strong Monitoring of LLM Agents](https://arxiv.org/abs/2508.19461)
[Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509)
[The Assistant Axis: Situating and Stabilizing the Character of Large Language Models](https://www.anthropic.com/research/assistant-axis)
[Detecting Multi-Agent Collusion Through Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151)
[You Can’t Escape Your Own Activations: Evaluation Awareness and Multi-Agent Monitoring](https://arxiv.org/abs/2609.03035)
[Another Slice of Swiss Cheese for Untrusted Monitoring](https://www.lesswrong.com/posts/zJWgfsad8o9Y8E9sb/another-slice-of-swiss-cheese-for-untrusted-monitoring-1)
[Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs](https://arxiv.org/abs/2510.04303)
[Language Models Represent Beliefs of Self and Others](https://arxiv.org/abs/2402.18496)
[Brittle Minds, Fixable Activations: Understanding Belief Representations in Language Models](https://arxiv.org/abs/2406.17513)
[Analysing the Generalisation and Reliability of Steering Vectors](https://discovery.ucl.ac.uk/id/eprint/10205543/)
[Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)
[How's it going? Reinforcement learning in language models recruits a functional welfare axis](https://arxiv.org/abs/2605.30232)
[Contrastive Covariance Subspaces for Activation Steering](https://arxiv.org/abs/2610.14025)
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)

This catalog preserves the plan's linked reading. Historical J-Lens material remains outside the active experiment scope; these links are not a claim that every method was implemented or every paper replicated.

### LLM usage statement

AI assistance supported implementation, artifact inspection, figures and report drafting. Quantitative counts were recomputed from saved traces. Qualitative interpretations were assistant-assisted and are not represented as independent human adjudication. Model-emitted reasoning and synthetic introspection are not evidence of faithful access to internal computation. Author review is required before submission.
