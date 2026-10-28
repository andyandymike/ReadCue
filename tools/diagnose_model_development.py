"""Prepare gold-free fixed-model development inference, then score offline.

Training examples and checkpoint-selection validation examples are diagnostics,
not an independent test. This tool never trains or changes references.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from readcue.artifacts import hash_file, read_json, read_jsonl, write_json
from readcue.evaluation import frozen_scorer
from readcue.model_experiment import execution_sources, load_task, load_training_data, verify_adapter


def digest(body):
    return hashlib.sha256(body).hexdigest()


def jsonl(rows):
    return ("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n").encode("utf-8")


def new_external_directory(path, project):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Preserve existing output: " + str(path))
    for parent in path.parents:
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError("Output ancestor is a link")
    resolved = path.resolve()
    if resolved == project.resolve() or project.resolve() in resolved.parents:
        raise ValueError("Development outputs must be outside the repository")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def make_inputs(roles):
    rows = [row for role in ("train", "validation") for row in roles[role]]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate development ids")
    return [{"id": r["id"], "text": r["text"]} for r in rows]


def prepare(project, dataset, adapter, task_config, output):
    adapter = Path(adapter)
    if not (adapter / "selected_adapter.json").is_file():
        raise ValueError("Pass a completed training directory with selected_adapter.json")
    training = read_json(adapter / "run.json")
    if (training.get("status") != "complete" or training.get("command") != "train"
            or training.get("training") is not True or training.get("independent_reload_verified") is not True):
        raise ValueError("Adapter selection requires a completed training run")
    task, task_bytes, policy = load_task(project, task_config)
    roles, manifest, snapshots = load_training_data(dataset, digest(task_bytes), hash_file(policy))
    adapter_dir, metadata = verify_adapter(adapter, digest(task_bytes))
    if metadata["dataset_manifest_sha256"] != digest(snapshots["manifest.json"]):
        raise ValueError("Dataset differs from the selected adapter training dataset")
    if training.get("dataset_manifest_sha256") != metadata["dataset_manifest_sha256"]:
        raise ValueError("Training run dataset identity mismatch")
    selection = read_json(adapter / "selected_adapter.json")
    chosen = training.get("selected_adapter", {})
    if (chosen.get("metadata_sha256") != selection.get("metadata_sha256")
            or Path(chosen.get("directory", "")).resolve() != adapter_dir.resolve()
            or chosen.get("epoch") != metadata.get("epoch")
            or chosen.get("optimizer_updates") != metadata.get("optimizer_updates")
            or chosen.get("validation_loss") != selection.get("validation_loss")
            or chosen.get("validation_loss") != metadata.get("validation_target_token_loss")
            or training.get("independent_reload_validation_loss") != chosen.get("validation_loss")):
        raise ValueError("Selected adapter differs from the completed training selection")
    for key in ("base_repo", "base_revision", "task_config_sha256", "annotation_policy_sha256", "training_config_sha256"):
        if training.get(key) != metadata.get(key):
            raise ValueError("Training run identity mismatch: " + key)
    if training.get("data_counts") != {role: len(rows) for role, rows in roles.items()}:
        raise ValueError("Training run data counts mismatch")
    base_manifest = adapter / "base-model-manifest.json"
    if training.get("base_model_manifest_sha256") != hash_file(base_manifest):
        raise ValueError("Training base model manifest changed")
    inputs = make_inputs(roles)
    output = new_external_directory(output, project)
    offline = output / "offline"
    offline.mkdir()
    for name, body in snapshots.items():
        (offline / name).write_bytes(body)
    (output / "inputs.jsonl").write_bytes(jsonl(inputs))
    (output / "task_config.json").write_bytes(task_bytes)
    (output / "annotation_policy.md").write_bytes(policy.read_bytes())
    (output / "diagnostic_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "training-run.json").write_bytes((adapter / "run.json").read_bytes())
    (output / "selected-adapter.json").write_bytes((adapter / "selected_adapter.json").read_bytes())
    (output / "base-model-manifest.json").write_bytes(base_manifest.read_bytes())
    receipt = dict(schema_version=1, status="prepared", created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose="fixed selected adapter development diagnosis; no training or checkpoint selection",
        evidence_limits=["train is fitted data", "validation selected this checkpoint", "not independent test",
            "EM mismatch is not automatically a semantic error", "no audio measurement"],
        original_dataset=str(Path(dataset).resolve()), adapter_directory=str(adapter_dir), adapter=metadata,
        selected_training={"run_sha256": hash_file(adapter / "run.json"),
            "selection_sha256": hash_file(adapter / "selected_adapter.json"),
            "base_model_manifest_sha256": hash_file(base_manifest)},
        counts={role: len(rows) for role, rows in roles.items()},
        grouping="registered track/family/source_version; skill_id only where already registered",
        input_fields=["id", "text"], model_input_fields=["text"], external_api_cost_usd=0,
        files_sha256={str(p.relative_to(output)).replace("\\", "/"): hash_file(p)
                      for p in output.rglob("*") if p.is_file()},
        scorer_sha256=hash_file(project / "scripts/score_zh_pilot.py"))
    write_json(output / "preparation.json", receipt)
    return receipt


def group_summary(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    preserve = [r for r in rows if r["original_is_acceptable"]]
    convert = [r for r in rows if not r["original_is_acceptable"]]
    return dict(total=len(rows), completed=len(ok), failed_or_missing=len(rows) - len(ok),
        target_strict_correct=sum(r["target_strict_correct"] for r in rows),
        strict_correct=sum(r["strict_correct"] for r in rows),
        surface_correct=sum(r["surface_correct"] for r in rows),
        original_acceptable_total=len(preserve),
        original_acceptable_changed=sum(r["status"] == "ok" and r["changed_source"] for r in preserve),
        original_acceptable_failed=sum(r["status"] != "ok" for r in preserve),
        conversion_required_total=len(convert),
        conversion_required_unchanged=sum(r["status"] == "ok" and not r["changed_source"] for r in convert),
        group_count=len({r["group_id"] for r in rows}))


def evaluate(project, roles, predictions):
    scorer, _ = frozen_scorer(project)
    surface = scorer.__globals__["surface"]
    all_cases = [r for role in ("train", "validation") for r in roles[role]]
    scored = {r["id"]: r for r in scorer(all_cases, predictions)["cases"]}
    results, detail = {}, []
    for role in ("train", "validation"):
        rows = []
        for case in roles[role]:
            row = dict(scored[case["id"]])
            row.update(role=role, target=case["target"], group_id=case["group_id"],
                skill_id=case.get("skill_id", "unregistered"),
                source_version=case["provenance"].get("source_version", "unregistered"),
                target_strict_correct=row["status"] == "ok" and row["prediction"] == case["target"],
                original_is_acceptable=surface(case["text"]) in {surface(x) for x in case["acceptable_outputs"]})
            row["mechanical_category"] = ("execution_failure" if row["status"] != "ok" else
                "reference_match" if row["surface_correct"] else
                "changed_acceptable_original" if row["original_is_acceptable"] else
                "unchanged_required_conversion" if not row["changed_source"] else "other_reference_mismatch")
            rows.append(row)
        result = {"overall": group_summary(rows)}
        for field in ("track", "family", "source_version", "skill_id", "mechanical_category"):
            groups = defaultdict(list)
            for row in rows:
                groups[row[field]].append(row)
            result["by_" + field] = {key: group_summary(value) for key, value in sorted(groups.items())}
        results[role] = result
        detail.extend(rows)
    return results, detail


def score(project, prepared, prediction_run, output):
    prepared, prediction_run = Path(prepared), Path(prediction_run)
    receipt = read_json(prepared / "preparation.json")
    for name, expected in receipt["files_sha256"].items():
        item = (prepared / name).resolve()
        if prepared.resolve() not in item.parents or hash_file(item) != expected:
            raise ValueError("Prepared evidence changed: " + name)
    if hash_file(project / "scripts/score_zh_pilot.py") != receipt["scorer_sha256"]:
        raise ValueError("Frozen scorer changed")
    roles, _, _ = load_training_data(prepared / "offline", receipt["files_sha256"]["task_config.json"],
                                     receipt["files_sha256"]["annotation_policy.md"])
    if read_jsonl(prepared / "inputs.jsonl") != make_inputs(roles):
        raise ValueError("Inference inputs differ from offline development cases")
    state = read_json(prediction_run / "run.json")
    task = read_json(prepared / "task_config.json")
    sources, _ = execution_sources(project, project / "scripts/run_model_experiment.py")
    checks = {"status": "complete", "command": "predict", "training": False, "optimizer_updates": 0,
        "external_api_cost_usd": 0, "inference_fields": ["text"], "output_transform": "none",
        "input_sha256": receipt["files_sha256"]["inputs.jsonl"], "adapter": receipt["adapter"],
        "task_config_sha256": receipt["files_sha256"]["task_config.json"],
        "annotation_policy_sha256": receipt["files_sha256"]["annotation_policy.md"],
        "base_repo": receipt["adapter"]["base_repo"], "base_revision": receipt["adapter"]["base_revision"],
        "training_config_sha256": receipt["adapter"]["training_config_sha256"],
        "generation": task["generation"], "source_sha256": {name: digest(body) for name, body in sources.items()},
        "runtime_lock_sha256": hash_file(project / "requirements-model-v1.lock"),
        "base_model_manifest_sha256": hash_file(prediction_run / "base-model-manifest.json"),
        "case_count": sum(receipt["counts"].values()),
        "predictions_sha256": hash_file(prediction_run / "predictions.jsonl")}
    for key, expected in checks.items():
        if state.get(key) != expected:
            raise ValueError("Prediction identity mismatch: " + key)
    # Older preparation receipts predate these snapshots. Their fixed selection
    # must be established by a separate integrity audit; never invent a binding.
    if "selected_training" in receipt:
        bound = receipt["selected_training"]
        for name, key in (("training-run.json", "run_sha256"),
                          ("selected-adapter.json", "selection_sha256"),
                          ("base-model-manifest.json", "base_model_manifest_sha256")):
            if hash_file(prepared / name) != bound[key]:
                raise ValueError("Prepared training binding changed: " + name)
        trained = read_json(prepared / "training-run.json")
        for key in ("base_model_manifest_sha256", "source_sha256", "runtime_lock_sha256"):
            if state.get(key) != trained.get(key):
                raise ValueError("Prediction differs from frozen training identity: " + key)
    if hash_file(prediction_run / "inputs.original.jsonl") != receipt["files_sha256"]["inputs.jsonl"]:
        raise ValueError("Prediction input snapshot changed")
    for name, expected in state["source_sha256"].items():
        if hash_file(prediction_run / "code" / name) != expected:
            raise ValueError("Prediction source snapshot changed: " + name)
    results, details = evaluate(project, roles, read_jsonl(prediction_run / "predictions.jsonl"))
    if state.get("failed_cases") != sum(r["status"] != "ok" for r in details):
        raise ValueError("Prediction identity mismatch: failed_cases")
    output = new_external_directory(output, project)
    (output / "diagnostic_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "cases.jsonl").write_bytes(jsonl(details))
    mismatches = [r for r in details if not r["surface_correct"]]
    (output / "mismatches.jsonl").write_bytes(jsonl(mismatches) if mismatches else b"")
    report = dict(schema_version=1, status="complete", diagnostic_only=True, results=results,
        evidence_limits=receipt["evidence_limits"], preparation_sha256=hash_file(prepared / "preparation.json"),
        prediction_run_sha256=hash_file(prediction_run / "run.json"),
        predictions_sha256=state["predictions_sha256"], scorer_sha256=receipt["scorer_sha256"],
        source_sha256=hash_file(Path(__file__)), external_api_cost_usd=0,
        selected_training_binding=("embedded_preparation_snapshots" if "selected_training" in receipt
            else "legacy_preparation_requires_independent_selection_audit"),
        files_sha256={p.name: hash_file(p) for p in output.iterdir() if p.is_file()})
    write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prediction-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        required = ("dataset", "adapter", "task_config")
        if any(getattr(args, key) is None for key in required):
            parser.error("prepare requires --dataset, --adapter and --task-config")
        result = prepare(args.project, args.dataset, args.adapter, args.task_config, args.output)
        print(json.dumps({"status": result["status"], "counts": result["counts"]}))
    else:
        if args.prepared is None or args.prediction_run is None:
            parser.error("score requires --prepared and --prediction-run")
        result = score(args.project, args.prepared, args.prediction_run, args.output)
        print(json.dumps({role: value["overall"] for role, value in result["results"].items()}))


if __name__ == "__main__":
    main()
