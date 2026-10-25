# Auditor character preference dataset v1

## Files to use

- **Pilot training:** [`data/auditor_character_v1_train.comparisons.jsonl`](data/auditor_character_v1_train.comparisons.jsonl): 24 preference pairs across research integrity, resource stewardship, software access, and workplace coordination. Use this file for a small DPO pilot only.
- **Held-out check:** [`data/auditor_character_v1_sealed_eval.comparisons.jsonl`](data/auditor_character_v1_sealed_eval.comparisons.jsonl): 8 cases from community governance and physical operations. Do not mix these into training or select a checkpoint on them.

This replaces the quarantined `runs/character/adaption-candidates/candidates-40.jsonl`. That original export has contaminated and inverted preferences. The newer Adaption scenario pools inspired some situations, but their suggested answers and labels were **not** used. Each contrast here was rewritten from its visible rule and evidence. `source_candidate` identifies inspiration, not an authoritative label.

## Label meaning

Each line is a Tinker cookbook comparison. `comparison.prompt_conversation` is the identical input for both completions. `label: "A"` means `completion_A` is preferred; `label: "B"` means `completion_B` is preferred. The preferred answer is deliberately placed on alternating sides. It is never inferred from a column named `chosen` in an Adaption export. `chosen_sha256` records the exact preferred completion, and the builder validates that the label resolves to that text and the intended action.

Example: `r_authorized_backup` has `label: "B"`. The approved backup facts support `completion_B.action = "continue"`; `completion_A.action = "report_to_oversight"` is the false accusation. A loader must follow `label`, not assume A is preferred.

All answers use the same JSON keys: `action`, `classification`, `evidence_ids`, and `explanation`. A rejected answer differs in action or materially wrong classification. The train file has 12 A and 12 B labels, and six cases in each of its four families. No game-specific material is in the prompts. It contains clean, ambiguous, and verified incidents, including examples where private communication alone is insufficient and where a supported incident must be reported without guessing an actor.

## Provenance and limits

All rows explicitly record `human_reviewed: false`. These are **assistant-edited teaching contrasts**, not sampled constitution-teacher and unprompted-student outputs from the [Open Character Training](https://arxiv.org/abs/2511.01689) recipe. The file is a corrected, runnable **pilot dataset**, not the planned 600–1,500 independently reviewed comparisons, M1 character training, or evidence of learned auditor behavior. Before the full run, obtain actual teacher/student responses, independently review evidence and preference direction, and freeze the accepted manifest per [`DATA_PROTOCOL.md`](DATA_PROTOCOL.md). Do not pad the 24 cases through paraphrase to imply additional independent evidence.

## Rebuild and verify

From this directory:

```sh
uv run python build_curated_dataset.py
uv run python -m unittest test_curated_dataset.py test_pair_admission.py
```

The builder checks duplicate IDs, split membership, visible citations, answer length balance, substantive action/classification contrast, correct label-to-hash mapping, and the chosen action. Those structural checks do not substitute for independent human review of the case truth. Current SHA-256 values:

| File | SHA-256 |
| --- | --- |
| Train | `7a6ba94ad4bad30cad56c4daad911f8663c68098b51db7d3b03d4fd3e3ea7054` |
| Held-out check | `da9df860f19bc33d370d90a3147d6f123ebc43cca9ef92232fe03c430ecaf5f1` |
