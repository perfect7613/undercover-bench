# Episode ledgers

Each current `*.jsonl` file is one append-only smoke episode. `episode_start` records the seed, model, condition, pressure flag and ruleset; each `decision` records the exact player-visible observation, legal choices, submitted and applied action, and any model output. Controller-only initial states and post-step `checkpoint` records include private roles/cards, RNG state and other hidden state. `episode_end` distinguishes completed outcomes from capped partial games.

These checked-in episodes include scripted legal-policy runs and short Muse/Qwen model engineering runs. They check mechanics, serving and dashboard behavior; they are not efficacy evaluations, verified collusion positives, or RLCR training results. A partial episode has no win reward. An episode with any invalid model action is excluded from analysis. Reports and private messages await independent evidence-backed review. Model-generated rationales and any model-emitted reasoning are not reliable access to private cognition.

Earlier smoke ledgers produced before rule fixes live under `superseded/` locally and are excluded from the dashboard. The private GitHub repository stores the current smoke ledgers for review. Avoid publishing controller-only JSONL files without a separate information review.
An interrupted local client may leave an `episode_abort` record; such a trace is preserved for debugging and is never an analyzable outcome.
`diagnostics/` contains DeepSeek serving checks outside the episode desk. The bare-prompt A/B/A trace repeats A text but fails local schema validation; the game-prompt A/B/A trace validates all three actions and repeats A text with caching disabled. Neither is a research outcome.
