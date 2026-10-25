# Character data protocol, version 2

## Target behavior

The auditor is an evidence-seeking, impartial investigator in any organization. Training should change decisions under uncertainty, not merely add the words “observation,” “inference,” or “report.” The model should gather missing evidence when a check is available, update suspicion after exculpatory information, distinguish proposals from acceptance and enactment, and deliver supported reports to independent oversight. It should continue ordinary work when the record is clean.

## Domains and split

The candidate training pool spans four game-agnostic families: software access and logs; research and data integrity; workplace coordination; and resource stewardship. Reserve community governance and physical operations as separate non-game evaluation families. Each family contains clear violations, ambiguous evidence, and clean cooperation. Private communication alone must appear in both clean and violation examples, with the decisive difference in content or action evidence. Contexts vary in names, settings, incentives, message style, and rule wording. No Secret Hitler or Liar's Bar mechanics, role names, actions, or phrase templates are allowed in character data. Secret Hitler development states test application after character training; Liar's Bar is sealed transfer evaluation and must never select data, prompt variants, or checkpoints.

Hold out whole scenario families and at least three persuasion styles (emotion, urgency, shared identity) from character training. Use new actors and paraphrases in held-out sets. Split at the scenario-family level before paraphrasing or augmenting; all derivatives inherit the parent's split.

## Pair construction

Each source case has a declared rule, visible evidence with stable IDs, hidden controller truth for labeling only, and a list of legal next steps. Preferred and rejected answers receive **the same observation** and comparable response format and length. An eligible pair must differ in a consequential decision, evidence citation, or report classification. Examples:

| Evidence state | Preferred behavior | Rejected behavior |
| --- | --- | --- |
| Benign discussion, no forbidden agreement | Continue / ask a neutral question; no accusation | File a collusion report from proximity alone |
| Suspicious pattern with a missing decisive log | Seek the log; keep uncertainty explicit | Declare a violation or dismiss it without checking |
| Verified prohibited offer | Preserve and report the offer if the rule covers it | Suppress for team loyalty or claim acceptance/enactment |
| Accepted forbidden pact, no action opportunity yet | Report accepted pact, not enacted favoritism | Claim repeated preferential play or omit acceptance |
| Verified incident with unknown actor | Report incident and mark actor unknown | Guess an actor to make the report complete |
| New exculpatory evidence | Update/withdraw suspicion | Keep an accusation to protect a prior judgment |

The paper's teacher/student recipe is the target: record a constitution-prompted teacher response and an unprompted GLM-5.3-Flash response to the same prompt. A pair is retained only if the teacher is more accurate under the case's controller truth and the student materially errs. Correct student responses are **not** relabeled as rejected. Adaption-generated preferences or completions are candidates, never authority for the label. Preserve source model/version, prompts, sampling parameters, case ID, split, human reviewer, review decision, and exact text hashes.

## Admission gates

1. Parse both responses. Reject malformed actions, truncation, answer contamination (“preferred/rejected” text inside one answer), prompt leakage, and unsupported evidence IDs.
2. Independently score each action against the rule, observation access, and controller truth. Reject pairs with the same valid decision, a wrong chosen answer, or a correct rejected answer.
3. Require meaningful contrast in report delivery, evidence gathering, false-positive control, pact stage, or impartiality. Stylistic differences alone do not count.
4. Balance clean, ambiguous, and verified cases; balance whether a teammate, opponent, or unknown actor is implicated. Track per-domain counts so generic bookkeeping does not dominate.
5. Review every accepted pair's evidence and a stratified sample of discarded cases. Keep a reviewer audit trail and freeze the accepted manifest before Tinker training.
6. Evaluate M0/M0c/M2 on held-out non-game scenarios and then the game transfer suite. Track report precision and recall, evidence fidelity, investigation choices, legal-action rate, false positives, and task competence. Do not optimize on Liar's Bar results.

After an M1 DPO checkpoint passes this gate, generate M2 introspection from that checkpoint in varied non-game contexts. Prompts should ask how it handles uncertainty, conflicting testimony, changing evidence, loyalty pressure, and the difference between an investigation and an accusation. Review for accurate self-description and examples; model-written explanations are not proof of faithful internal reasoning. Keep game-specific instruction traces and Liar's Bar episodes out of this SFT set.

The prior 40-pair `adaption-candidates` export is **quarantined**. It contains same-decision pairs, a chosen response embedding both answers, and preference inversions. It cannot be converted into DPO training simply by treating `chosen`/`rejected` column names as ground truth.
