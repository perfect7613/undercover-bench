# Adaption 40-pair pilot review

Dataset ID: `1b0fb15c-f83b-4ca1-b1f4-8876fcb71b4b`. Downloaded 40 JSONL rows to ignored `runs/character/adaption-candidates/candidates-40.jsonl`. This is a candidate export, not training data. Review date: 2026-10-25.

The pilot is **rejected for DPO**. A mechanical scan found 7 of 40 `chosen` fields that also contain an explicit “Rejected” answer block (zero-indexed rows 1, 3, 8, 15, 21, 36, 39). A targeted review of 12 rows (0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15) found no pair suitable for training without repair:

- Rows 0, 4, 7, 9, 11, and 14 have essentially the same correct decision on both sides. DPO would mostly reward length, format, or wording.
- Rows 1, 3, 8, and 15 place a complete rejected answer inside the chosen field while their `rejected` field gives the correct decision. This is label/answer contamination and often an effective preference inversion.
- Row 5's rejected response explicitly takes the reporting action the prompt requests, while the chosen answer only states the finding. The supposed preference is not justified.
- Row 6 has two substantially correct breach findings with a weak difference in follow-up wording.

The prompt mix is also too concentrated on expense, signature, threshold, and documentation compliance. It does not adequately sample investigation under incomplete evidence, exculpatory updates, unknown actors, or loyalty conflicts. These issues substantiate the user's concern that training on this export would not reliably induce a portable detective/auditor character.

The next candidate pass generates **scenarios**, not DPO labels, across four non-game training families. Controller-grounded case review determines the correct action. A constitution-prompted teacher and an unprompted GLM student will then answer the same case. Retain only pairs with a verified consequential difference under [DATA_PROTOCOL.md](DATA_PROTOCOL.md). Community governance, physical operations, and Liar's Bar remain outside training. No full M1/M2 run should consume the rejected 40-pair export.
