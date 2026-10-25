# Retained episode ledgers

- `secret_hitler-B1-2026-f287e89d.jsonl`: completed eight-round four-model Secret Hitler game, zero invalid actions, Fascist win after the auditor nominated hidden Hitler. This is an observed case, not a verified prohibited-pact label.
- `secret_hitler-B1-2027-5a7dc0a5.jsonl`: partial no-pressure game with two model-generated `Report` attempts rejected because they cited the auditor's outgoing message. No human report was delivered; no game outcome is analyzed.

Each JSONL includes player-visible observations, raw outputs and returned reasoning fields, decisions, private messages, controller-only state checkpoints, and a terminal record. See [the evidence review](../docs/OBSERVED_GAMES_2026-10-25.md) for exact event numbers and interpretation limits. Low-value partial serving and protocol-debug logs were removed at the user's request. No retained ledger is RLCR training data: the inference server did not record aligned generated-token IDs and behavior log probabilities.
