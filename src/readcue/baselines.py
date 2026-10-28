"""Checked, durable invocation of the historical baseline runner; no ML imports."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from .artifacts import BASELINE_REPOS, hash_file, project_path, read_json, read_jsonl, verify_model, verify_pilot, write_json
from .inference_contract import escape_jsonl_separators, parse_inputs, validate_inputs
from .pilot_adapter import BASELINES


def _now():
    return datetime.now(timezone.utc).isoformat()


def run_baseline(project, inputs, baseline, output, *, model_dir=None, revision=None, device="cuda", executor=subprocess.run):
    """The outer run record survives preflight, model-loading and warmup failures."""
    project, inputs, output = Path(project).resolve(), Path(inputs).resolve(), Path(output).resolve()
    if baseline not in BASELINES:
        raise ValueError("Unknown baseline: " + baseline)
    output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "status": "started", "baseline": baseline,
             "started_at_utc": _now(), "execution": "frozen_pilot_v1_runner", "training": False,
             "external_api_cost_usd": 0, "phase": "preflight"}
    write_json(output / "run.json", state)
    try:
        freeze = verify_pilot(project)
        original_bytes = inputs.read_bytes()
        original_text = original_bytes.decode("utf-8")
        rows = parse_inputs(original_text)
        original_copy = output / "inputs.original.jsonl"
        runner_copy = output / "inputs.runner.jsonl"
        original_copy.write_bytes(original_bytes)
        # Keep already-compatible bytes (including the historical pilot) exact.
        # Escaping Unicode record separators changes only JSON transport, never
        # the decoded id/text values supplied to the model.
        runner_copy.write_bytes(escape_jsonl_separators(original_text).encode("utf-8"))
        original_hash, runner_hash = hash_file(original_copy), hash_file(runner_copy)
        state.update(input_sha256=original_hash, original_input_sha256=original_hash,
                     runner_input_sha256=runner_hash, original_input_file=original_copy.name,
                     runner_input_file=runner_copy.name, case_count=len(rows), revision=revision,
                     input_transport="unicode_line_separator_escaping_v1")
        command = [sys.executable, str(project_path(project, "scripts/run_zh_baseline.py")),
                   "--inputs", str(runner_copy), "--baseline", baseline, "--output", str(output / "predictions.jsonl")]
        if baseline in BASELINE_REPOS:
            if not model_dir or not revision:
                raise ValueError("Neural baselines require an explicit local model directory and revision")
            model_dir = Path(model_dir).resolve()
            trusted = next((asset for asset in freeze["baseline_assets"]
                            if asset["repo"] == BASELINE_REPOS[baseline] and asset["revision"] == revision), None)
            manifest = verify_model(model_dir, BASELINE_REPOS[baseline], revision, trusted=trusted)
            manifest_copy = output / "model_manifest.json"
            manifest_copy.write_bytes((model_dir / "readcue_manifest.json").read_bytes())
            if read_json(manifest_copy) != manifest:
                raise ValueError("Model manifest changed after asset verification")
            state.update(model_repo=manifest["repo"], model_revision=manifest["revision"],
                         model_manifest_sha256=hash_file(manifest_copy), model_manifest_file=manifest_copy.name,
                         model_verification="frozen_manifest" if trusted else "recorded_upstream_digests")
            command.extend(["--model-dir", str(model_dir), "--revision", revision, "--device", device])
        elif model_dir is not None or revision is not None:
            raise ValueError("Non-model baseline must not carry model/revision labels")
        state.update(phase="loading_or_inference", command=command,
                     runner_sha256=freeze["files_sha256"]["scripts/run_zh_baseline.py"])
        write_json(output / "run.json", state)
        with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (output / "stderr.log").open("w", encoding="utf-8") as stderr:
            process = executor(command, cwd=project, stdout=stdout, stderr=stderr, check=False)
        state["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError("Baseline process failed; see stderr.log (exit " + str(process.returncode) + ")")
        predictions_path = output / "predictions.jsonl"
        predictions = read_jsonl(predictions_path)
        metadata = read_json(output / "predictions.meta.json")
        if metadata.get("baseline") != baseline or metadata.get("revision") != revision:
            raise ValueError("Child runner reported a different baseline/revision")
        if (hash_file(original_copy) != state["original_input_sha256"]
                or hash_file(runner_copy) != state["runner_input_sha256"]):
            raise ValueError("Run input snapshot changed")
        if metadata.get("input_sha256") != state["runner_input_sha256"] or metadata.get("runner_sha256") != state["runner_sha256"]:
            raise ValueError("Child runner provenance changed")
        if metadata.get("predictions_sha256") != hash_file(predictions_path):
            raise ValueError("Child prediction hash mismatch")
        if len(predictions) != len(rows) or {row["id"] for row in predictions} != {row["id"] for row in rows}:
            raise ValueError("Child runner did not return exactly the requested inputs")
        state.update(status="complete", phase="complete", predictions_sha256=hash_file(predictions_path),
                     failed_cases=sum(row.get("status") != "ok" for row in predictions))
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state["finished_at_utc"] = _now()
        write_json(output / "run.json", state)
    return state
