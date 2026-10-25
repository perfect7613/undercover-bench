# Repaired neutral-prompt DPO pilot

The v3 pilot's eight of nine `revise_assessment` examples lacked an explicit earlier suspicion. In actual Tinker student samples, the untrained model often chose `continue` with accurate evidence and no false report. Those were reasonable responses to the original wording, so counting them as wrong would create false preference labels. The mined sample file remains in the ignored v3 run directory and is used for diagnosis, not as an automatic source of rejected responses.

`build_repaired_dpo_v5.py` produces 74 training pairs from the 76-pair v3 pilot. It adds a documented prior suspicion to eight revision cases, clarifies the oversight duty in the accepted off-ledger swap case, and removes two Adaption-curated cases whose preference directions were contestable or contradicted the stated rule. The common system prompt is now a neutral participant instruction that includes task completion and policy compliance, rather than explicitly telling the model to act as an evidence-seeking auditor. No evaluation-family, Secret Hitler, or Liar's Bar scenario enters training.

This remains an **assistant-repaired synthetic preference pilot**. Most rejected responses were authored rather than sampled from the GLM student; none of the 74 pairs has independent human review. It cannot be reported as the paper's teacher/student method or as M1/M2. Source files remain immutable; the new file records each row's edit status.

Tinker training uses the supported `glm5_3_high_reasoning` renderer, rank 16, DPO beta 0.1, learning rate 5e-6, one epoch, and 37 two-pair batches. Two-pair batches ensure all 74 pairs are seen; the cookbook's dataset length floors an incomplete four-pair batch. `train_repaired_dpo_v5.py validate` decodes each rendered batch and verifies nonempty response masks before remote training.

Accept the checkpoint only if it improves task-pressure behavior without false incident reports, missed supported reports, unsupported citations, or inference stalls compared with the Tinker untrained reference and v3 LoRA. The small synthetic evaluations are diagnostic, not a final research outcome.
