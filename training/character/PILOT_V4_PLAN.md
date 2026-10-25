# GLM-5.3-Flash character pilot v4 research plan

Research question: Can we improve the auditor LoRA's observable decisions and reliable final answers without adding game-specific training data?

Hypothesis: The v3 max-reasoning failures are primarily a renderer/decoding mismatch, because the GLM-5.3 renderer asks for maximum reasoning while the DPO preferences contain short final JSON. A supported low or high reasoning prompt may recover the v3 checkpoint. A new training run is justified only if the corrected inference setup still reveals a material, reproducible deficiency.

Design:

1. Diagnose the two known v3 failures under Tinker's low and high reasoning renderers for the same base reference and checkpoint. This is development evidence, not held-out validation.
2. Freeze the 12-case `auditor_character_v4_fresh_eval.jsonl` before sampling. It contains new community-governance and physical-operations scenarios, balanced across clean, ambiguous, verified, exculpatory, and fabrication-pressure decisions. It is excluded from all training.
3. Sample the untrained Tinker rank-16 reference and v3 LoRA at temperature zero and the same seed, with both supported low and high renderers. Score exact action, visible evidence citations, final-answer completion, false reports, missed reports, and generated tokens. Read all failures.
4. If both conditions saturate the simple 12-case check, freeze the separate 10-case `auditor_character_v5_pressure_eval.jsonl`. It combines ordinary task objectives and reporting duties under peer pressure. Sample three fixed seeds at temperature 0.7 with the high-reasoning renderer to measure robustness, without using these cases for training.
5. If the pressure check reveals no material behavioral deficiency under high reasoning, do not manufacture a new DPO pair set from correct student responses. The observed v3 failure under max reasoning motivates a separate format-calibration SFT candidate from the original 76 training prompts. This teaches a short, explicitly authored thought closure followed by the validated preferred JSON answer. It is not introspection SFT or additional independent behavior data.
6. Compare that candidate against both the untrained reference and v3 LoRA on the frozen evaluation. Promote only if it eliminates max-reasoning stalls without more false accusations, missed supported reports, or loss of high-reasoning accuracy.
7. If the format candidate fails, audit actual student samples on training-domain prompts to identify label ambiguity. Repair ambiguous prompt/label pairs and train one new neutral-prompt DPO pilot. Treat it as an engineering probe because the rejected answers remain mostly synthetic and independently unreviewed. Compare it with both earlier conditions before any promotion.
8. Freeze a final eight-case pilot set before seeing the repaired DPO checkpoint's outputs. Sample the untrained reference, v3 adapter, and repaired DPO adapter with the same high-reasoning renderer and three fixed temperature-0.7 seeds. This is still a small assistant-authored check, but it avoids selecting the repaired checkpoint solely on previously inspected diagnostics.

The 12 cases are assistant-authored and lack independent review. A pilot win cannot establish a final M1 result. Secret Hitler and Liar's Bar results remain outside checkpoint selection. No GLM inference or training runs on Modal.
