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
