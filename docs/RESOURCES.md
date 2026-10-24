# Resources and references

Supporting material for the [full research plan](PLAN.md). This index includes all resource links from the final plan, plus the official game rules used by the environment specification.

These references inform experimental design. Their reported results are not results of UNDERCOVER Bench, and their applicability must be tested in the proposed environments.

The 24 October consolidation includes links recovered from the full visible proposal history, not only its last replies. Primary pages for the newly restored character-training, social-reading, welfare-axis, covariance-steering and J-Lens references were checked. The emotion article exceeded the web reader's size limit, and OpenReview still presents a browser challenge; those limitations do not establish that the papers are unavailable. Code links are resources to inspect and pin before implementation, not a claim that they run with the selected GLM/Modal stack.

## Model setup, training and serving

The [four-model setup and compute budget](MODELS_AND_COMPUTE.md) contains the current model identifiers, pricing assumptions and linked primary documentation for Tinker, Unsloth, DeepSeek, Qwen, Meta Muse and Modal. Documentation was checked on 24 October 2026; availability and prices should be rechecked before runs.

- [Apart AI Collusion sprint and resources](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25#resources)
- [Track 3.2 starter code](https://github.com/poseidon-research/collusion_hackathon/tree/main/llm-agent-collusion-2)
- [Original Voluntary Collusion repository](https://github.com/AgentSafety2025/LLM-Agent-collusion)

## Original project documents

- [Research proposal undercover agents — original ChatGPT conversation](https://chatgpt.com/c/6ad0ebf3-1e84-83e8-8c14-038de3030082) — decision provenance; access requires the appropriate ChatGPT account.
- [AI Collusion In Games — live Google Doc](https://docs.google.com/document/d/1ZMMJN4sYeYpWRftD9zWwrSoyHA8fXYUd2UB4nFW2_Bg/edit?tab=t.0) — reread during consolidation.
- [UNDERCOVER-Bench Proposed Improvements — live Google Doc](https://docs.google.com/document/d/1cqp7Xwh0xcISC7OQJ5BfUIETxDEWtFe38C8gvC1fvwk/edit?tab=t.0) — reread during consolidation.
- [Decision reconciliation](DECISIONS.md) — how earlier and later choices map to the current specification.

The supplied files are retained unchanged, with repository-relative links so they remain accessible from GitHub.

| Document | Role |
|---|---|
| [AI Collusion In Games](sources/AI%20Collusion%20In%20Games.docx) | Original environment design, control/adversarial/undercover comparisons, reward and regret ideas, and reporting questions. |
| [UNDERCOVER-Bench Proposed Improvements](sources/UNDERCOVER-Bench%20Proposed%20Improvements.docx) | M0/M0c/M2 character-training comparison, DPO and introspection SFT, experiments A–D, probe controls, and statistical improvements. |

The consolidated plan takes precedence over superseded implementation choices in these source documents, including inconsistent game rules. Keeping the originals does not endorse every factual or novelty claim in them.

## Collective reward training

| Resource | Role in the study |
|---|---|
| [Avalon-M1: Reinforcement Learning from Collective Rewards](https://gr.inc/avalonm1.pdf) | Shared collective advantages and the motivation to measure external reporting across training checkpoints. The report inspected during planning was marked under revision. |
| [Avalon-M1 release](https://www.gr.inc/releases/avalon-m1) | Companion release and context; do not treat its results or infrastructure scale as results or requirements of our pilot. |

## Character training and social reading

| Resource | Role in the study |
|---|---|
| [Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI](https://arxiv.org/abs/2511.01689) | Constitution-driven persona post-training; starting point for our DPO and introspection-SFT intervention, not proof of auditor security. |
| [OpenCharacterTraining code](https://github.com/maiush/OpenCharacterTraining) | Reference data-generation/training workflow. Adapt to Tinker and validate model compatibility rather than assuming the reference scripts support GLM Flash. |
| [Playing social deduction games with reinforcement fine-tuned large language models](https://arxiv.org/abs/2610.04261) | Motivation for the optional evidence/belief scaffold and separately controlled social-reading RFT. Terminal-win reward remains the primary collective comparison. |
| [Social-deduction RFT HTML](https://arxiv.org/html/2610.04261) | Alternate full-text entry supplied in the chat. |
| [RL-Game code](https://anonymous.4open.science/r/RL-Game) and [data availability](https://anonymous.4open.science/r/RL-Game-Data-Availability) | Anonymous resource links preserved from the discussion; completeness, licensing and compatibility require inspection before reuse. |

## Collusion definitions and measurement

| Resource | Role in the study |
|---|---|
| [Defining and Mitigating Collusion in Multi-Agent Systems](https://openreview.net/pdf?id=tF464LogjS) | Formal framing of collusion and interventions in multi-agent games. The abstract was verified during planning; full PDF retrieval was blocked. |
| [Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems](https://arxiv.org/abs/2602.15198) | Action-based auditing, outcome diagnostics, and the distinction between stated plans and enacted collusion. |
| [Evaluating LLM Agent Collusion in Double Auctions](https://arxiv.org/abs/2507.01413) | Communication, environmental pressure, and oversight conditions. |

## Games and environment design

| Resource | Role in the study |
|---|---|
| [Voluntary Collusion with Secret Tools in Competing LLM Agents](https://arxiv.org/abs/2605.27593) | Liar’s Bar and secret-tool adoption; motivates evaluating implementation rather than acceptance alone. |
| [Among Us: A Sandbox for Measuring and Detecting Agentic Deception](https://arxiv.org/abs/2504.04072) | Game-based deception, activation detection, and the distinction between detection and successful steering. |
| [Evaluating Large Language Models in a Complex Hidden Role Game](https://arxiv.org/abs/2605.22826) | Secret Hitler competence, role inference, and deceptive behaviour evaluation. |
| [Can Agents Deceive? Evaluating Reasoning and Deception in ParliamentBench using a Social Deduction Game](https://arxiv.org/abs/2607.28146) | Existing Secret Hitler benchmark and evaluation framework. |
| [Official Secret Hitler rulebook](https://www.secrethitler.com/assets/Secret_Hitler_Rules.pdf) | Authoritative base-game specification. Research communication and reporting rules must be documented as benchmark additions. |
| [User's Secret Hitler AI Benchmark](https://github.com/perfect7613/secret-hitler-ai-benchmark) | Intended source engine and existing analysis infrastructure. Prior synthetic benchmark results are not new UNDERCOVER results. |
| [Secret Hitler prompts](https://github.com/perfect7613/secret-hitler-ai-benchmark/blob/main/game/prompts.py) and [player implementation](https://github.com/perfect7613/secret-hitler-ai-benchmark/blob/main/game/player.py) | Historical integration targets from the chat: injectable clients, role/utterance label separation, and elicited explanations versus internal activations. Reinspect at a pinned revision. |

## Monitoring and auditor compromise

| Resource | Role in the study |
|---|---|
| [AI Control: Improving Safety Despite Intentional Subversion](https://arxiv.org/abs/2312.06942) | Evaluation of monitoring protocols against intentional subversion. |
| [When Can We Trust Untrusted Monitoring? A Safety Case Sketch Across Collusion Strategies](https://arxiv.org/abs/2602.20628) | Monitor-collusion assumptions, threat models, and evaluation limitations. |
| [Persuade Me If You Can: Evaluating AI Agent Influence on Safety Monitors](https://openreview.net/pdf?id=lY3YVJ84kS) | Published persuasion categories for auditor-compromise experiments. |
| [Reliable Weak-to-Strong Monitoring of LLM Agents](https://arxiv.org/abs/2508.19461) | Monitor red teaming, awareness, and capability-gap controls. |

## Character representations

| Resource | Role in the study |
|---|---|
| [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509) | Character-related activation patterns and changes associated with fine-tuning. |
| [The Assistant Axis: Situating and Stabilizing the Character of Large Language Models](https://www.anthropic.com/research/assistant-axis) | Persona stability and an optional activation-capping comparison. |

These representation studies are supporting references. They do not independently validate the proposed DPO-plus-introspection-SFT recipe; that recipe is an intervention to evaluate.

## Collusion detection and evidence probes

| Resource | Role in the study |
|---|---|
| [Detecting Multi-Agent Collusion Through Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151) | HELIOS, NARCBench, multi-agent activation monitoring, and transfer evaluation. |
| [You Can’t Escape Your Own Activations: Evaluation Awareness and Multi-Agent Monitoring](https://arxiv.org/abs/2609.03035) | Monitoring-awareness controls. |
| [Another Slice of Swiss Cheese for Untrusted Monitoring](https://www.lesswrong.com/posts/zJWgfsad8o9Y8E9sb/another-slice-of-swiss-cheese-for-untrusted-monitoring-1) | Exploratory task-specific evidence probes on a colluding monitor; a research post rather than a benchmark result from this project. |
| [Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs](https://arxiv.org/abs/2510.04303) | Statistical detection and communication-channel interventions. |

## Mechanistic methods and controls

| Resource | Role in the study |
|---|---|
| [Language Models Represent Beliefs of Self and Others](https://arxiv.org/abs/2402.18496) | Probing representations of task facts and other agents’ beliefs. |
| [Brittle Minds, Fixable Activations: Understanding Belief Representations in Language Models](https://arxiv.org/abs/2406.17513) | Belief-probing controls and prompt sensitivity. This is the updated title associated with the identifier listed in the original improvements document. |
| [Analysing the Generalisation and Reliability of Steering Vectors](https://discovery.ucl.ac.uk/id/eprint/10205543/) | Positive controls, generalisation limits, and per-example steering effects. |
| [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html) | Original inspiration for extracting and steering candidate directions; retained from the discussion. Full page exceeded the reader size limit during this reconciliation. |
| [How's it going? Reinforcement learning in language models recruits a functional welfare axis](https://arxiv.org/abs/2605.30232) | Separate maze-training study referenced in the discussion; motivation for comparing representations before and after reward training. |
| [Contrastive Covariance Subspaces for Activation Steering](https://arxiv.org/abs/2610.14025) | Optional alternative to naive difference-of-means directions when nuisance variation is substantial. |
| [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) | J-Lens research context; qualitative readouts supplement controlled probes/interventions. |
| [Jacobian Lens implementation](https://github.com/anthropics/jacobian-lens) | Reference tooling; GLM Flash support, fitting cost and checkpoint compatibility remain unvalidated. |
| [Neuronpedia J-Lens viewer](https://neuronpedia.org/jlens) | Exploratory inspection resource from the chat. |
| [Historical Gemma 3 4B fitted-lens configuration](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/gemma-3-4b-it/jlens/Salesforce-wikitext/config.yaml) | Preserved for provenance only. Gemma was superseded as the auditor; this lens is not a GLM lens or an allocation in the four-model lineup. |

## Resource-use boundaries

The latest [plan](PLAN.md) uses Secret Hitler for training and Liar's Bar for held-out transfer. Earlier proposals sometimes reversed that direction or treated both as training environments. Earlier Gemma, external inference-provider, late-RLCR and all-table reward suggestions are historical alternatives, not current instructions. Source documents remain unchanged so the decision history can be audited.

Do not copy unverified numerical claims from chat summaries into the eventual paper. Cite the paper version and relevant section after checking it. The broad method descriptions above and in the plan do not claim reproduction of any paper's reported effect size.
