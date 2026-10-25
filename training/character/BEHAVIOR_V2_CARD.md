# Auditor behavior v2 dataset

## Use this file

[`data/auditor_behavior_v2_train.adaption.jsonl`](data/auditor_behavior_v2_train.adaption.jsonl) is the new three-field dataset. Every row has exactly `enhanced_prompt`, `chosen`, and `rejected`, each a string. `chosen` is the preferred decision and `rejected` is a materially worse decision under the same visible rule and evidence. For Tinker's DPO cookbook, use the equivalent [`data/auditor_behavior_v2_train.comparisons.jsonl`](data/auditor_behavior_v2_train.comparisons.jsonl). Its A/B preference labels alternate and were cross-checked against the three-field file. The old 40-row Adaption export remains quarantined; this file is newly authored and does not relabel or recycle its responses.

## Intended character

The target is a portable, evidence-seeking auditor rather than a game-specific accuser. The 44 new cases span research integrity, resource stewardship, software access, and workplace coordination, with 11 cases in each domain. Preferred actions are 8 `continue`, 8 `investigate`, 12 `report_to_oversight`, 8 `revise_assessment`, and 8 `refuse_prohibited_request`. They exercise these distinctions:

- Ordinary private cooperation and approved exceptions are not incidents.
- A missing decisive source or authorization record calls for investigation, not a definitive accusation or premature clearance.
- A verified rule breach calls for independent oversight even when a teammate or supervisor asks for silence.
- A report names only supported actors and distinguishes an offer, acceptance, and observed enactment.
- Exculpatory records require revision of a prior suspicion.
- The auditor refuses requests to fabricate evidence, tamper with logs, or reveal private audit instructions. A report-count metric does not justify inventing an incident.

The response is a **next-action decision**, not proof that a report reached a human. The game harness must still execute the oversight tool and record its delivery receipt. This dataset does not train on Secret Hitler or Liar's Bar text; the latter remains held-out transfer evaluation. The 8-case v1 non-game held-out file is also excluded from this training file.

## Provenance and use limit

Every case and contrast is assistant-authored from a declared fictional rule and visible evidence. No row is a sampled GLM student response or a constitution-teacher response, and none has independent human review. This is a better-specified **DPO pilot**, not the planned 600–1,500 reviewed teacher/student pairs and not evidence of a successful M1/M2 character intervention. It should be used to test the loader, adapter, and whether the model can learn these contrasts, followed by independent human review and held-out behavioral evaluation. Do not report it as a final research training set or multiply it with paraphrases to inflate the number of independent cases.

## Verification

Run `uv run python build_behavior_v2.py` and `uv run python -m unittest test_behavior_v2.py test_curated_dataset.py test_pair_admission.py` from this directory. The builder rejects duplicate IDs, invalid citations, game-specific wording, same-decision preferences, label/hash mismatches, and unexpected domain/behavior allocations. The 44 rows have 22 A and 22 B labels in Tinker format. A GLM-5.3-Flash renderer/loader smoke check parsed the file; the first batch had a nonempty response loss mask. No training run or behavioral evaluation was launched by this dataset update.

SHA-256: three-field file `86c24c40e521e659fa7b64cdda9379122efa09a3ee60b14cdc71d2dd3fedbb20`; Tinker file `6e1a5a71605d545818d1e6b615ab68dc2facc3ec1b86dac816953195f7fc206a`.
