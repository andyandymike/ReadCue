"""Checked, durable invocation of the historical baseline runner; no ML imports."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from .artifacts import BASELINE_REPOS, hash_file, project_path, read_json, read_jsonl, verify_model, verify_pilot, write_json
from .evaluation import BASELINES


def validate_inputs(path):
    rows, seen = read_jsonl(path), set()
    if not rows:
        raise ValueError("No inference inputs")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "text"}:
            raise ValueError("Inference input must contain only id and text")
        if not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in seen:
            raise ValueError("Missing or duplicate inference id")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Inference text must be a nonempty string")
        seen.add(row["id"])
    return rows


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
        rows = validate_inputs(inputs)
        state.update(input_sha256=hash_file(inputs), case_count=len(rows), revision=revision)
        command = [sys.executable, str(project_path(project, "scripts/run_zh_baseline.py")),
                   "--inputs", str(inputs), "--baseline", baseline, "--output", str(output / "predictions.jsonl")]
        if baseline in BASELINE_REPOS:
            if not model_dir or not revision:
                raise ValueError("Neural baselines require an explicit local model directory and revision")
            model_dir = Path(model_dir).resolve()
            trusted = next((asset for asset in freeze["baseline_assets"]
                            if asset["repo"] == BASELINE_REPOS[baseline] and asset["revision"] == revision), None)
            manifest = verify_model(model_dir, BASELINE_REPOS[baseline], revision, trusted=trusted)
            state.update(model_repo=manifest["repo"], model_revision=manifest["revision"],
                         model_manifest_sha256=hash_file(model_dir / "readcue_manifest.json"))
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
        if metadata.get("input_sha256") != state["input_sha256"] or metadata.get("runner_sha256") != state["runner_sha256"]:
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
