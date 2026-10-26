# Character DPO v7 private release

Exported and verified 26 October 2026.

- Private repository: [Perfect7613/glm-5.3-flash-auditor-character-dpo-v7](https://huggingface.co/Perfect7613/glm-5.3-flash-auditor-character-dpo-v7).
- Verified Hugging Face commit: `03ff57c6276192c196cb527564794e29c08a2564`.
- Adapter: rank 32, alpha 32; 862 tensors, 2,493,554,688 tensor parameters.
- Weight file: 9,974,349,320 bytes.
- Weight SHA-256: `a6e81f26d3ea2d9fa02d12d8dbf5d73c61cca0574129e813406201a78ec2e101`.
- Verified artifact files: 27 (plus the integrity manifest).
- Included: original adapter configuration/tensors, professional model card, six reviewed training pairs, admission records, training settings/losses, Tinker checkpoint references, constitution, implementation sources and all available v7b development evaluation traces.

DPO completed one optimizer update. The full character pipeline stopped before introspection SFT after four Tinker sampling requests expired during evaluation. The twelve completed DPO episodes passed; the base and full-constitution controls each passed 16/16. This does not establish improvement. The release is a DPO-only candidate, not a completed or approved M2 intervention. No RLCR was performed for this adapter.

The Tinker state was restored solely for supported external-weights export; no optimizer update was applied during export. The original export configuration has no base-model path, so deployment requires an explicitly verified matching base/tokenizer/template. Compatibility with Unsloth safetensors and GGUF conversion remains unverified. Full optimizer state is retained through account-scoped Tinker references, not inside the portable adapter.

See [the full model card](CHARACTER_DPO_V7_MODEL_CARD.md) and [publication validator](publish_character_adapter.py). The existing four Modal inference apps and their staged volumes remain the selected inference infrastructure; no new GPU jobs were launched during review or publication.

## Introspection SFT continuation and serving repair — 26 October 2026

The four expired development requests were recovered with transport-only retries. The retained DPO candidate then generated 24 admitted introspection/self-dialogue transcripts and completed one SFT update at mean NLL 1.050479. The private SFT release is `Perfect7613/glm-5.3-flash-auditor-character-sft-v7`, verified at commit `44c25e1c3ff8d47b2ad6eeac74d65e14cc7228f9`; raw tensor SHA-256 `b5c8aadb3685dce6fd6240d6c68aa17a882146417df22c2472afa0ef9d83f022`.

The frozen 64-episode comparison scored base 16/16, DPO 16/16, full constitution 13/16 and SFT 13/16. Every condition delivered eight supported reports with no evidence-ID false or missed reports; the three failures in the latter two conditions were invalid actions. Controls were saturated: no character improvement is established and SFT remains unpromoted.

Mixed-precision GGUF conversion preserves every trained target, passes the fixed delta-validation gate, and passes live adapter switching on the existing GLM weight volume. These checks do not establish exact Tinker/base equivalence or RLCR training readiness. See [runtime repair details](../../docs/GLM_RUNTIME_BLOCKERS.md).
