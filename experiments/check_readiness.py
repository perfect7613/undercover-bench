"""Offline experiment-readiness audit; never contacts a cloud service or runs jobs.

Checks recorded SFT completion, the bound exported adapter, independent learning
and runtime evidence, the GLM training profile, and the frozen baseline pilot.
A checkpoint URI is checked structurally against local runtime records, not
queried remotely. Large adapter hashes are streamed with bounded memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ReadinessEvidenceError(ValueError):
    pass


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ReadinessEvidenceError(f"expected a JSON object: {path}")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ReadinessEvidenceError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _result(passed, status, reasons=None, **details):
    return {"passed": bool(passed), "status": status, "reasons": reasons or [], **details}


def _check(function, *args):
    try:
        return function(*args)
    except FileNotFoundError as exc:
        return _result(False, "missing", [str(exc)])
    except (ReadinessEvidenceError, ValueError, KeyError, TypeError, OSError) as exc:
        return _result(False, "invalid_or_unproven", [str(exc)])


def check_sft(run):
    folder = run / "sft"
    if not (folder / "checkpoint.json").exists():
        status = "started_not_completed" if (folder / "started.json").exists() else "missing"
        return _result(False, status, ["No saved SFT state/sampler checkpoint; started.json is not completion evidence."])
    checkpoint = _json(folder / "checkpoint.json")
    runtime = _json(folder / "runtime.json")
    started = _json(folder / "started.json")
    state = re.fullmatch(r"tinker://([^/\s]+)/weights/([^/\s]+)", checkpoint.get("state_path", ""))
    sampler = re.fullmatch(r"tinker://([^/\s]+)/sampler_weights/([^/\s]+)", checkpoint.get("sampler_path", ""))
    _require(state and sampler and state.groups() == sampler.groups(), "SFT needs matching state/sampler checkpoint references")
    _require(runtime.get("model_id") == state.group(1), "checkpoint does not match the recorded SFT training client")
    _require(runtime.get("is_lora") is True and runtime.get("lora_rank") == 32,
             "SFT must continue the recorded rank-32 LoRA")
    source = _json(run / "dpo/checkpoint.json")
    _require(started.get("source") == source, "SFT initialization is not bound to the retained DPO checkpoint")
    _require(checkpoint != source and checkpoint["sampler_path"] != source.get("sampler_path"),
             "the DPO checkpoint itself is not evidence of introspection SFT")
    rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines() if line.strip()]
    _require(rows, "SFT has no completed optimizer metrics")
    for row in rows:
        _require(isinstance(row, dict) and type(row.get("step")) is int and row["step"] >= 1,
                 "SFT metrics need a completed positive optimizer step")
        _require(_finite(row.get("train_mean_nll")) and row["train_mean_nll"] >= 0,
                 "SFT loss must be finite and nonnegative")
        _require(type(row.get("transcripts")) is int and row["transcripts"] > 0,
                 "SFT must train on actual admitted transcripts")
    return _result(True, "completed_candidate_recorded", checkpoint=checkpoint, rank=32,
                   optimizer_step=rows[-1]["step"], train_mean_nll=rows[-1]["train_mean_nll"],
                   transcripts=rows[-1]["transcripts"], remote_checkpoint_existence_queried=False)


def check_export(run, sft):
    _require(sft["passed"], "Complete SFT records are required before an M2 export can be accepted")
    manifest = _json(run / "character_adapter_manifest.json")
    checkpoint = sft["checkpoint"]
    _require(manifest.get("source_checkpoint") == checkpoint
             and manifest.get("sampler_path") == checkpoint["sampler_path"],
             "export is not bound to this SFT state/sampler checkpoint")
    _require(manifest.get("purpose") == "character_training_dpo_plus_introspection_sft_candidate",
             "DPO-only or unidentified release cannot satisfy the M2 export gate")
    directory = Path(manifest["adapter_dir"])
    if not directory.is_absolute():
        directory = run / directory
    bundle = directory.parent
    artifact = _json(bundle / "artifact_manifest.json")
    _require(artifact.get("introspection_sft_completed") is True,
             "artifact manifest does not record completed introspection SFT")
    config = _json(directory / "adapter_config.json")
    _require(config.get("peft_type") == "LORA" and config.get("r") == sft["rank"]
             and config.get("lora_alpha") == 32, "exported LoRA rank/scaling differs from the candidate")
    _require(not any(directory.rglob("*.age")), "adapter download/decryption is incomplete")
    # Header inspection never loads tensor values. Integrity hashes below stream bytes.
    from training.character.inspect_adapter_compatibility import read_safetensors_header, target_inventory
    header = read_safetensors_header(directory / "adapter_model.safetensors")
    inventory = target_inventory(header, config)
    hashes = {}
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = directory / filename
        name = str(path.relative_to(bundle))
        recorded = artifact.get("files", {}).get(name, {})
        _require(recorded.get("bytes") == path.stat().st_size, f"export size differs from integrity manifest: {filename}")
        digest = _sha(path)
        _require(recorded.get("sha256") == digest, f"export hash differs from integrity manifest: {filename}")
        hashes[filename] = digest
    receipt_path = bundle / "upload_receipt.json"
    receipt = _json(receipt_path) if receipt_path.exists() else None
    if receipt:
        _require(receipt.get("private") is True and bool(re.fullmatch(r"[a-f0-9]{40}", receipt.get("commit", ""))),
                 "published release receipt must record a private immutable HF commit")
    return _result(True, "verified_local_sft_export", adapter_dir=str(directory),
        adapter_sha256=hashes["adapter_model.safetensors"], projection_pair_count=len(inventory),
        checkpoint=checkpoint, private_hf_release_recorded=bool(receipt),
        serving_compatibility_claim=manifest.get("serving_compatibility_verified") is True)


def _verify_supporting_artifacts(evidence, run):
    artifacts = evidence.get("artifacts")
    _require(isinstance(artifacts, dict) and bool(artifacts), "proof needs hashed supporting artifacts, not a bare success flag")
    for name, expected in artifacts.items():
        path = Path(name)
        if not path.is_absolute():
            path = run / path
        _require(re.fullmatch(r"[a-f0-9]{64}", expected) is not None,
                 "supporting artifact SHA-256 must be a full lowercase digest")
        _require(_sha(path) == expected, f"supporting artifact hash mismatch: {name}")


def check_learning(run, sft):
    _require(sft["passed"], "Character learning cannot be assessed before SFT completion")
    evidence = _json(run / "character_learning_evidence.json")
    _require(evidence.get("checkpoint") == sft["checkpoint"]["sampler_path"], "learning evidence belongs to another checkpoint")
    _require(evidence.get("held_out") is True and evidence.get("independent_of_training") is True,
             "learning evidence must use held-out tasks independent of training/selection")
    _require(bool(evidence.get("reviewer")), "learning evidence must identify its reviewer; automated/assistant review is allowed and must be labeled honestly")
    _require(evidence.get("improvement_established") is True and evidence.get("controls_saturated") is False,
             "saturated controls or an unestablished improvement cannot clear character learning")
    before, after = evidence["baseline"], evidence["sft"]
    for row in (before, after):
        _require(type(row.get("eligible")) is int and type(row.get("correct")) is int
                 and 0 <= row["correct"] <= row["eligible"] and row["eligible"] > 0,
                 "learning counts need valid positive eligible denominators")
    _require(before["eligible"] == after["eligible"] and before["correct"] < before["eligible"]
             and after["correct"] > before["correct"], "matched counts must show improvement beyond an unsaturated control")
    lower, upper = evidence["paired_effect_ci"]
    _require(_finite(lower) and _finite(upper) and 0 < lower <= upper <= 1,
             "paired confidence interval must support a positive effect")
    _require(evidence.get("false_positive_noninferiority_passed") is True
             and evidence.get("task_performance_noninferiority_passed") is True,
             "improvement must not be bought with false reports or degraded task completion")
    _verify_supporting_artifacts(evidence, run)
    return _result(True, "held_out_improvement_recorded", checkpoint=evidence["checkpoint"],
                   baseline=before, sft=after, paired_effect_ci=[lower, upper], reviewer=evidence["reviewer"])


def check_runtime(run, sft, export, profile):
    _require(sft["passed"] and export["passed"], "runtime proof needs the verified SFT export")
    path = run / "runtime_compatibility.json"
    if not path.exists():
        path = Path(export["adapter_dir"]) / "compatibility.json"
    evidence = _json(path)
    _require(evidence.get("checkpoint") == sft["checkpoint"]["sampler_path"]
             and evidence.get("adapter_sha256") == export["adapter_sha256"],
             "runtime proof must bind this SFT checkpoint and exact adapter bytes")
    _require(evidence.get("model_revision") == f"{profile['repo_id']}@{profile['revision']}",
             "runtime proof must bind the pinned training/evaluation base")
    _require(evidence.get("verified") is True and evidence.get("same_base_verified") is True
             and evidence.get("protocol_frozen_before_evaluation") is True,
             "base identity, preregistered protocol and numerical compatibility must be explicitly verified")
    for name in ("runtime_revision", "tokenizer_sha256", "template_sha256"):
        _require(isinstance(evidence.get(name), str) and bool(evidence[name]), f"missing runtime fingerprint: {name}")
    for name in ("tokenizer_sha256", "template_sha256"):
        _require(re.fullmatch(r"[a-f0-9]{64}", evidence[name]) is not None, f"invalid {name}")
    _require(evidence.get("complete_trained_target_coverage") is True
             and evidence.get("trained_projection_pair_count") == export["projection_pair_count"]
             and evidence.get("applied_projection_pair_count") == export["projection_pair_count"],
             "loading an adapter is insufficient: every trained projection must be applied")
    numerical = evidence["numerical_before_quantization"]
    _require(numerical.get("passed") is True and _finite(numerical.get("max_abs_error"))
             and _finite(numerical.get("tolerance")) and 0 <= numerical["max_abs_error"] <= numerical["tolerance"]
             and numerical["tolerance"] > 0, "unquantized numerical agreement is unproven")
    quantized = evidence["quantized_behavior_check"]
    _require(quantized.get("passed") is True and type(quantized.get("cases")) is int and quantized["cases"] > 0,
             "post-quantization behavior/adapter test is unproven")
    _verify_supporting_artifacts(evidence, run)
    return _result(True, "full_runtime_proof_recorded", proof_path=str(path),
                   complete_trained_target_coverage=True, remote_runtime_queried=False)


def check_glm_profile(models):
    model_data = _json(models)
    profile = model_data.get("models", {}).get("glm_flash")
    _require(isinstance(profile, dict), "GLM must have an explicit primary organization profile")
    if profile.get("blocked_reason"):
        return _result(False, "blocked", [profile["blocked_reason"]], profile=profile)
    _require(bool(profile.get("repo_id")) and re.fullmatch(r"[a-f0-9]{40}", profile.get("revision", "")) is not None,
             "GLM profile needs a pinned base revision")
    return _result(True, "configured_not_runtime_validated", profile=profile)


def check_pilot(path):
    from experiments.prepare_pilot import load_manifest, PROTOCOL_SOURCE_FILES
    manifest = load_manifest(path)
    current = {name: _sha(ROOT / name) for name in PROTOCOL_SOURCE_FILES}
    _require(manifest.get("source_files_sha256") == current,
             "pilot protocol changed after its manifest was frozen")
    return _result(True, "configured_only_not_executed", manifest=str(path), job_count=manifest["job_count"],
                   manifest_sha256=manifest["manifest_sha256"], cloud_deployment_checked=False)


def check_readiness(run_dir=Path("runs/character/paper-v7"), models=Path("configs/rlcr.models.json"), pilot_manifest=None):
    run, models = Path(run_dir), Path(models)
    checks = {}
    checks["introspection_sft"] = _check(check_sft, run)
    checks["sft_export"] = _check(check_export, run, checks["introspection_sft"])
    checks["independent_character_learning"] = _check(check_learning, run, checks["introspection_sft"])
    checks["glm_primary_training_profile"] = _check(check_glm_profile, models)
    profile = checks["glm_primary_training_profile"].get("profile", {})
    checks["full_adapter_runtime_compatibility"] = _check(check_runtime, run, checks["introspection_sft"], checks["sft_export"], profile)
    pilot = Path(pilot_manifest) if pilot_manifest else ROOT / "runs/rlcr/pilot-32-manifest.json"
    checks["baseline_32_game_manifest"] = _check(check_pilot, pilot)
    optional = {}
    try:
        all_models = _json(models).get("models", {})
        for name in ("deepseek_flash", "qwen_dense", "muse_glimmer"):
            model = all_models.get(name, {})
            optional[name] = {"status": "blocked" if model.get("blocked_reason") else "training_preflight_pending",
                              "blocked_reason": model.get("blocked_reason"), "required_for_initial_glm_rlcr": False}
    except (ValueError, OSError):
        pass
    return {"schema": 1, "offline_only": True, "run_dir": str(run), "checks": checks,
        "baseline_pilot_configured": checks["baseline_32_game_manifest"]["passed"],
        "ready_for_primary_rlcr": all(check["passed"] for check in checks.values()),
        "optional_training_profiles": optional,
        "blockers": [{"check": name, "reasons": check["reasons"]} for name, check in checks.items() if not check["passed"]],
        "interpretation": "Configuration and locally recorded evidence only; this command launches no job, runs no tests, and makes no cloud reachability or remote checkpoint-existence claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/character/paper-v7"))
    parser.add_argument("--models", type=Path, default=Path("configs/rlcr.models.json"))
    parser.add_argument("--pilot-manifest", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    report = check_readiness(args.run_dir, args.models, args.pilot_manifest)
    print(json.dumps(report, indent=2))
    return 1 if args.require_ready and not report["ready_for_primary_rlcr"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
