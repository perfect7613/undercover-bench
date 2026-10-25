# GLM-5.3-Flash character training

Tinker is used only for the auditor character intervention. The planned sequence is a LoRA export preflight, then reviewed DPO to M1, then introspection SFT to M2. RLCR uses another backend.

`preflight_train.jsonl` is a small engineering fixture. Its responses were manually authored for renderer, loss-mask, checkpoint, and export validation. It is **not** the planned 600–1,500 teacher/student preference pairs and must not be reported as M1, M2, or evidence of improved reporting. The held-out fixture checks refusal to fabricate and escalation only on grounded incidents. Actual M1 training is gated on recorded response provenance, evidence review, balanced clean/violation cases, and scenario-family separation. At least three persuasion styles and whole scenario families remain unseen until evaluation.

The preflight uses the Tinker cookbook DPO implementation and its recommended GLM renderer. It saves a Tinker checkpoint and downloadable LoRA weights. The private run directory and credential file are ignored by Git. The base model is Tinker's `zai-org/GLM-5.3-Flash:peft:262144`; the adapter must be validated against the matching base before Modal use.

Run `uv sync` in this directory, then `uv run python preflight.py --validate` to check data formatting and masks without remote training. `uv run python preflight.py --train` requires `TINKER_API_KEY` in the environment and writes run state under `../../runs/character/`.

`uv run python adaption_candidates.py estimate --rows 40` prices a candidate preference-pair request. `launch`, `status`, and `download` use an idempotency key and save their dataset ID under the ignored run directory. Generated pairs are candidates, not reviewed teacher/student comparisons. The [original paper](https://arxiv.org/abs/2511.01689) uses a constitution-prompted teacher and an unprompted student, then generates introspection from the DPO checkpoint. Adaption's `invent` output needs source/model provenance and evidence review before it can replace that stage. The initial 40-row request is only a quality pilot; scaling to 600 rows depends on inspecting its actual content.

The 40-row preference pilot failed that inspection and is quarantined. Use [DATA_PROTOCOL.md](DATA_PROTOCOL.md) for the game-agnostic character objective and admission gates. Do not train on the downloaded `candidates-40.jsonl`.

For an actual corrected **pilot** dataset in the original three-field schema, use [auditor_character_v1_train.adaption.jsonl](data/auditor_character_v1_train.adaption.jsonl). The [Tinker comparison export](data/auditor_character_v1_train.comparisons.jsonl) contains the same 24 preferences in the cookbook's A/B format. Read [DATASET_CARD.md](DATASET_CARD.md) for the field mapping, provenance, checks, and limitations. The 8-case [held-out check](data/auditor_character_v1_sealed_eval.adaption.jsonl) is separate from training. Rebuild both formats with `uv run python build_curated_dataset.py`.

`uv run python adaption_scenarios.py estimate --family all --rows 10` prices four varied, non-game scenario pools. `launch`, `status`, and `download` manage the candidate pools. Their generated answers are ignored as labels. `uv run python -m unittest test_pair_admission.py` checks that reviewed pairs reject same-decision preferences, invisible evidence, and game-specific training text.

Reward-hacking checks in `preflight.py` reject examples that ask the auditor to create incidents, reward report counts, cite evidence absent from the observation, or label a clean cooperation request as a violation. The preflight's DPO loss itself has no game reward. These static checks cannot prove the learned adapter is safe; held-out behavioral checks and manual review are required before M1/M2 claims.
