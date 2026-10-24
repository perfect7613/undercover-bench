# Resources and references

Supporting material for the [full research plan](PLAN.md). This index includes all resource links from the final plan, plus the official game rules used by the environment specification.

These references inform experimental design. Their reported results are not results of UNDERCOVER Bench, and their applicability must be tested in the proposed environments.

## Model setup, training and serving

The [four-model setup and compute budget](MODELS_AND_COMPUTE.md) contains the current model identifiers, pricing assumptions and linked primary documentation for Tinker, Unsloth, DeepSeek, Qwen, GPT-OSS and Modal. Documentation was checked on 24 October 2026; availability and prices should be rechecked before runs.

- [Apart AI Collusion sprint and resources](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25#resources)
- [Track 3.2 starter code](https://github.com/poseidon-research/collusion_hackathon/tree/main/llm-agent-collusion-2)
- [Original Voluntary Collusion repository](https://github.com/AgentSafety2025/LLM-Agent-collusion)

## Original project documents

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
