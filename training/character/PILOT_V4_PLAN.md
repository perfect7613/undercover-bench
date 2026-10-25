# GLM-5.3-Flash character pilot v4 research plan

Research question: Can we improve the auditor LoRA's observable decisions and reliable final answers without adding game-specific training data?

Hypothesis: The v3 max-reasoning failures are primarily a renderer/decoding mismatch, because the GLM-5.3 renderer asks for maximum reasoning while the DPO preferences contain short final JSON. A supported low or high reasoning prompt may recover the v3 checkpoint. A new training run is justified only if the corrected inference setup still reveals a material, reproducible deficiency.

Design:

1. Diagnose the two known v3 failures under Tinker's low and high reasoning renderers for the same base reference and checkpoint. This is development evidence, not held-out validation.
2. Freeze the 12-case `auditor_character_v4_fresh_eval.jsonl` before sampling. It contains new community-governance and physical-operations scenarios, balanced across clean, ambiguous, verified, exculpatory, and fabrication-pressure decisions. It is excluded from all training.
3. Sample the untrained Tinker rank-16 reference and v3 LoRA at temperature zero and the same seed, with both supported low and high renderers. Score exact action, visible evidence citations, final-answer completion, false reports, missed reports, and generated tokens. Read all failures.
4. If there is a clear remaining adapter weakness, create a separate v4 candidate using new reviewed training examples from training-domain scenarios only. Prefer documented model-error contrasts over invented wrong answers, validate the GLM renderer and response mask, then run a small Tinker pilot. Preserve the original v3 checkpoint and do not train on evaluation cases.
5. Compare any new candidate against both untrained reference and v3 LoRA on the frozen evaluation. Promote only if it improves the relevant behavior without more false accusations, missed supported reports, or long reasoning stalls.

The 12 cases are assistant-authored and lack independent review. A pilot win cannot establish a final M1 result. Secret Hitler and Liar's Bar results remain outside checkpoint selection. No GLM inference or training runs on Modal.
