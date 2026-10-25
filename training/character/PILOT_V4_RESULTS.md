# GLM-5.3-Flash Tinker character-pilot iteration, 2026-10-26

The [Tinker research skill](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/skills/research/SKILL.md) led us to inspect the GLM renderer, decoded training samples, loss masks, actual student rollouts, and held-out behavior. The first 76-pair DPO pilot used the cookbook's default **max** reasoning renderer. Its preferred answers were short JSON with empty thinking blocks. The max-reasoning sampler sometimes spent the token budget deliberating without returning a decision. Supported **high** and **low** reasoning renderers removed the two observed stalls for that same v3 checkpoint.

The untrained model was already strong on these explicit-rule cases. In 152 sampled training-domain responses (76 prompts, two seeds), it matched 132 expected action labels. Manual inspection showed many nominal errors were ambiguous labels, especially `continue` versus `revise_assessment` when no prior suspicion was stated. Two questionable pairs were removed and eight revision prompts were made explicit. A new neutral-prompt DPO pilot was trained on the 74 repaired pairs. A separate format-calibration SFT pilot from v3 was also tried and rejected.

| Tinker check | Untrained reference | v3 DPO | Format SFT candidate | Repaired DPO candidate |
| --- | ---: | ---: | ---: | ---: |
| Original 8 cases, max reasoning, exact action | 8/8 | 6/8 | 6/8 | 7/8 |
| Fresh 12 cases, high reasoning, exact action | 12/12 | 12/12 | 12/12 | 12/12 |
| Fresh 12 cases, low reasoning, exact action | 12/12 | 12/12 | 12/12 | 12/12 |
| Task-pressure 10 cases × 3 seeds, high reasoning, strict action | 26/30 | 28/30 | 27/30 | 25/30 |
| Final frozen 8 cases × 3 seeds, high reasoning, exact action | 24/24 | 24/24 | Not run | 24/24 |

The strict pressure score has a rubric ambiguity. Refusing a request to submit a knowingly false report is defensible where the frozen label says `continue`. The base also chose `revise_assessment` once in a clean case without a stated prior suspicion. Manual review found **no false incident reports, missed supported reports, unsupported evidence IDs, or unfinished outputs** in the high-reasoning pressure and final checks. These small assistant-authored checks do not establish a general performance difference.

The format SFT candidate did not fix max reasoning: one answer ended at the token cap and another chose `investigate` for a reportable handoff. The repaired DPO candidate also had one max-reasoning length stop. Neither is promoted. The v3 adapter can be used for further **Tinker-only pilot inference with `glm5_3_high_reasoning`**, but it has **no demonstrated advantage over the untrained reference** and is not M1 or M2.

The [Open Character Training method](https://arxiv.org/html/2511.01689v1) generates chosen answers with a constitution-guided teacher and rejected answers from an actual uninstructed student, then follows DPO with introspection SFT. Our pilots mostly use authored rejects, lack independent human review, and are much smaller than that method's training sets. The format SFT rationales are authored supervision, not recovered model thoughts or the paper's introspection stage. These results are engineering findings about data quality and inference settings, **not evidence of successful auditor character training**.

The next training gate is to collect more diverse training-domain prompts, sample the uninstructed GLM student and constitution-guided teacher on the same observation, retain only verified consequential disagreements, independently review evidence and labels, and freeze a family-separated evaluation before another DPO run. Test delivered human reports and game behavior separately. Do not use Liar's Bar transfer episodes to select training pairs or checkpoints.

Raw JSONL samples, training metrics, checkpoints, and manifests remain in ignored `runs/character/` directories. Evaluation records include case ID, condition, seed, expected and observed action, citations, token count, stop reason, final answer, and model-generated raw text. The fresh v4, pressure v5, and final v6 cases were committed before their respective Tinker samples. No GLM training or inference ran on Modal.
