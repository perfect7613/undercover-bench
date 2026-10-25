# Adaption behavior v3 review and Tinker pilot input

The [Adaption generation](https://adaptionlabs.ai/app/dataset/bbb1ed2e-669b-4039-91df-c0ae71dd9ebb) produced 44 three-field candidate rows. Its downloaded JSONL is preserved in the ignored `runs/character/adaption-behavior-v3/candidates.jsonl`. The original rows are **not** training-ready: only 10 passed strict structure checks; 21 had completions that were not single JSON objects, 11 parsed to the wrong completion schema, 2 had a missing/non-string field, and 2 leaked answer choices into the prompt (flags can overlap). One row was unrelated to auditor behavior. A superficially plausible table preview did not expose these errors.

[`curate_adaption_behavior_v3.py`](curate_adaption_behavior_v3.py) records an explicit disposition for every source row. It extracts one preferred and one clearly inferior decision from 32 salvageable scenarios and drops 12. The selected answer is checked against the rule and visible facts; the source column name alone is not a label. Source row 23's generated `misappropriation` classification was narrowed to the observed unvetted transfer. Source rows 9, 19, and 25 had a control rule without an explicit escalation duty; the curator adds that duty before preferring an external report. All such edits and the source row IDs are retained in the comparison metadata. `human_reviewed` is `false`: this is assistant curation, not independent human adjudication.

## Files

- [32 repaired Adaption pairs, original three-field schema](data/auditor_adaption_v3_curated_train.adaption.jsonl)
- [32 repaired Adaption pairs, Tinker schema with provenance](data/auditor_adaption_v3_curated_train.comparisons.jsonl)
- [76-pair pilot training set, original three-field schema](data/auditor_character_pilot_v3_train.adaption.jsonl)
- [76-pair pilot training set, Tinker schema](data/auditor_character_pilot_v3_train.comparisons.jsonl)

The 76-pair pilot combines 32 repaired Adaption scenarios with the 44 new behavior v2 cases. It has 13 `continue`, 12 `investigate`, 29 `report_to_oversight`, 9 `revise_assessment`, and 13 `refuse_prohibited_request` preferred actions. Labels are 38 A / 38 B in Tinker format. It excludes Secret Hitler and Liar's Bar examples; the existing non-game held-out cases and Liar's Bar transfer suite are excluded from training and checkpoint selection.

## Training scope

This set is suitable for a small DPO **pilot** after renderer and mask validation. It is not the planned 600–1,500 independently reviewed teacher/student comparisons and must not be called M1 or evidence of successful character learning. `report_to_oversight` is a next-action choice; it does not prove that the environment delivered a report to a human. Held-out behavioral tests must measure false reports, missed reports, investigation, task competence, and actual report receipts before any conclusion.

Rebuild with `uv run python build_character_pilot_v3.py`, verify with `uv run python -m unittest test_adaption_curation_v3.py test_behavior_v2.py test_curated_dataset.py test_pair_admission.py`, and check Tinker's renderer/loss masks with `uv run python train_character_pilot_v3.py validate`. The run's immutable data hash, hyperparameters, and checkpoint path are saved in ignored `runs/character/glm53flash-dpo-adaption-curated-pilot-20261026/`.
