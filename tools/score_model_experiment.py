"""Offline scoring after training selection; no model execution or label writes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from readcue.artifacts import hash_file, read_json, read_jsonl, write_json
from readcue.evaluation import frozen_scorer
from readcue.inference_contract import write_jsonl
from readcue.model_scoring import evaluate, paired_changes
from readcue.model_experiment import verify_adapter, BASE_REPO, BASE_REVISION


def validate_run_evidence(args):
    """Bind evaluation to the proof selected before this actual training run."""
    directory = Path(args.training_run)
    train = read_json(directory / "run.json")
    if (train.get("status") != "complete" or train.get("command") != "train"
            or train.get("training") is not True or train.get("independent_reload_verified") is not True
            or not isinstance(train.get("optimizer_updates"), int) or train["optimizer_updates"] < 1
            or train.get("base_repo") != BASE_REPO or train.get("base_revision") != BASE_REVISION):
        raise ValueError("Require completed training and independently reloaded selected adapter")
    proof_sha = hash_file(args.heldout_proof)
    if train.get("heldout_proof_sha256") != proof_sha or hash_file(directory / "heldout-proof.json") != proof_sha:
        raise ValueError("Heldout proof differs from the exact proof bound before training")
    proof = read_json(args.heldout_proof)
    if proof.get("task_config_sha256") != train.get("task_config_sha256"):
        raise ValueError("Heldout and training use different task policies")
    if (train.get("dataset_version") == "model-data-v2"
            and proof.get("training_dataset_manifest_sha256") != train.get("dataset_manifest_sha256")):
        raise ValueError("The v2 heldout proof belongs to a different training dataset")
    selected = read_json(directory / "selected_adapter.json")
    if selected.get("metadata_sha256") != train.get("selected_adapter", {}).get("metadata_sha256"):
        raise ValueError("Selected adapter differs from the recorded validation choice")
    _, adapter = verify_adapter(directory, train["task_config_sha256"])
    if (adapter.get("dataset_manifest_sha256") != train.get("dataset_manifest_sha256")
            or adapter.get("dataset_version", "model-data-v1") != train.get("dataset_version", "model-data-v1")):
        raise ValueError("Selected adapter comes from a different training dataset")
    previous = getattr(args, "previous", None)
    previous_training_run = getattr(args, "previous_training_run", None)
    if bool(previous) != bool(previous_training_run):
        raise ValueError("Previous predictions require their original --previous-training-run, and conversely")
    previous_adapter = None
    if previous:
        previous_directory = Path(previous_training_run)
        previous_train = read_json(previous_directory / "run.json")
        if (previous_train.get("status") != "complete" or previous_train.get("command") != "train"
                or previous_train.get("training") is not True
                or previous_train.get("independent_reload_verified") is not True
                or type(previous_train.get("optimizer_updates")) is not int
                or previous_train["optimizer_updates"] < 1
                or previous_train.get("dataset_version", "model-data-v1") != "model-data-v1"):
            raise ValueError("Previous comparison requires a completed independently reloaded v1 training run")
        for key in ("base_repo", "base_revision", "task_config_sha256", "annotation_policy_sha256",
                    "training_config_sha256", "base_model_manifest_sha256"):
            if previous_train.get(key) != train.get(key):
                raise ValueError("Previous training uses a different fixed configuration: " + key)
        previous_selection = read_json(previous_directory / "selected_adapter.json")
        if previous_selection.get("metadata_sha256") != previous_train.get("selected_adapter", {}).get("metadata_sha256"):
            raise ValueError("Previous adapter differs from its original validation choice")
        _, previous_adapter = verify_adapter(previous_directory, train["task_config_sha256"])
        if (previous_adapter.get("dataset_manifest_sha256") != previous_train.get("dataset_manifest_sha256")
                or previous_adapter.get("dataset_version", "model-data-v1") != "model-data-v1"
                or previous_adapter == adapter
                or previous_train.get("dataset_manifest_sha256") == train.get("dataset_manifest_sha256")):
            raise ValueError("Previous adapter must be the distinct original v1 dataset and selected artifact")
    inputs_sha = proof.get("files_sha256", {}).get("inputs.jsonl")
    sources = {}
    for name in (("base", "adapted", "previous") if previous else ("base", "adapted")):
        prediction_path = Path(getattr(args, name))
        record_path = prediction_path.parent / "run.json"
        record = read_json(record_path)
        if (record.get("status") != "complete" or record.get("command") != "predict"
                or record.get("training") is not False or record.get("inference_fields") != ["text"]
                or record.get("output_transform") != "none" or record.get("input_sha256") != inputs_sha
                or hash_file(prediction_path.parent / "inputs.original.jsonl") != inputs_sha
                or record.get("predictions_sha256") != hash_file(prediction_path)):
            raise ValueError("Invalid or differently sourced prediction run: " + name)
        for key in ("base_repo", "base_revision", "task_config_sha256", "annotation_policy_sha256",
                    "training_config_sha256", "base_model_manifest_sha256", "source_sha256"):
            if record.get(key) != train.get(key):
                raise ValueError("Prediction and training execution identity differ: " + name + ":" + key)
        if record.get("generation") != {"do_sample": False, "max_new_tokens": 1024}:
            raise ValueError("Prediction decoding differs from the fixed comparison")
        if (name == "base" and "adapter" in record) or (name == "adapted" and record.get("adapter") != adapter):
            raise ValueError("Base/adapted system identity or selected adapter mismatch")
        if name == "previous" and record.get("adapter") != previous_adapter:
            raise ValueError("Previous prediction did not use the original v1 selected adapter")
        sources[name] = {"path": str(record_path.resolve()), "sha256": hash_file(record_path)}
    if previous:
        sources["previous_training"] = {"path": str((Path(previous_training_run) / "run.json").resolve()),
                                        "sha256": hash_file(Path(previous_training_run) / "run.json"),
                                        "selected_adapter_metadata_sha256": previous_selection["metadata_sha256"]}
    rules_path = Path(args.rules)
    rules = read_json(rules_path.parent / "run.json")
    if (rules.get("status") != "complete" or rules.get("policy_version") != "audit-corrected-v1"
            or rules.get("model_executed") is not False or rules.get("candidate_audit_only") is not False
            or rules.get("inputs_sha256", {}).get("inputs") != inputs_sha
            or rules.get("predictions_sha256") != hash_file(rules_path)):
        raise ValueError("Rule comparison identity or frozen inputs differ")
    sources["rules"] = {"path": str((rules_path.parent / "run.json").resolve()),
                         "sha256": hash_file(rules_path.parent / "run.json")}
    return train, proof, sources


def run(args):
    train, proof, run_sources = validate_run_evidence(args)
    heldout = Path(args.heldout)
    if proof.get("status") != "frozen":
        raise ValueError("Require pretraining heldout freeze")
    for name, expected in proof["files_sha256"].items():
        path = (heldout / name).resolve()
        if not path.is_relative_to(heldout.resolve()) or hash_file(path) != expected:
            raise ValueError("Heldout artifact differs from pretraining freeze: " + name)
    cases = read_jsonl(heldout / "cases.jsonl")
    inputs = read_jsonl(heldout / "inputs.jsonl")
    if [{"id": c["id"], "text": c["text"]} for c in cases] != inputs:
        raise ValueError("Heldout inputs do not match reference roster")
    scorer, _ = frozen_scorer(args.project)
    surface = scorer.__globals__["surface"]
    systems = {"identity": [{"id": row["id"], "prediction": row["text"], "status": "ok"} for row in inputs]}
    sources = {}
    names = ("base", "adapted", "rules", "previous") if getattr(args, "previous", None) else ("base", "adapted", "rules")
    for name in names:
        path = Path(getattr(args, name))
        rows = read_jsonl(path)
        sources[name] = {"file": str(path.resolve()), "sha256": hash_file(path)}
        if name == "rules":
            rows = [dict(row, original_status=row.get("status"), status="ok" if row.get("status") == "rules_only" else row.get("status")) for row in rows]
        systems[name] = rows
    scores = {name: evaluate(cases, rows, scorer, surface) for name, rows in systems.items()}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name, result in scores.items():
        write_json(output / (name + ".score.json"), result)
    pairing = paired_changes(scores["base"], scores["adapted"])
    summary = {"schema_version": 1, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "experiment": "zh-reading-model-v2" if train.get("dataset_version") == "model-data-v2" else "zh-reading-model-v1",
               "dataset_version": train.get("dataset_version", "model-data-v1"),
               "heldout_proof_sha256": hash_file(args.heldout_proof),
               "training_run_sha256": hash_file(Path(args.training_run) / "run.json"), "prediction_sources": sources,
               "run_sources": run_sources,
               "results": {name: {key: result[key] for key in ("overall", "by_track", "families", "total_cases", "reference_coverage", "execution_statuses", "local_targets", "protected_spans")} for name, result in scores.items()},
               "base_to_adapter": pairing, "semantic_review_status": "pending_blinded_review",
               "data_scope": "Assistant-authored synthetic evaluation, isolated from training and selection; not a random real-comment benchmark.",
               "metric_scope": "Text and mechanical span metrics; no measured pronunciation or TTS benefit."}
    summary["rules_to_adapter"] = paired_changes(scores["rules"], scores["adapted"])
    if "previous" in scores:
        summary["previous_to_adapter"] = paired_changes(scores["previous"], scores["adapted"])
    write_json(output / "summary.json", summary)
    blind_rows, key = [], {}
    indexes = {name: {row["id"]: row for row in rows} for name, rows in systems.items()}
    for case in cases:
        order = sorted(systems, key=lambda name: hashlib.sha256(("readcue-blind-v1|" + case["id"] + "|" + name).encode()).hexdigest())
        candidates = []
        key[case["id"]] = {}
        for index, name in enumerate(order):
            code = chr(ord("A") + index)
            row = indexes[name].get(case["id"], {"prediction": "", "status": "missing"})
            candidates.append({"code": code, "output": row["prediction"], "status": row["status"]})
            key[case["id"]][code] = name
        blind_rows.append({"id": case["id"], "text": case["text"], "acceptable_outputs": case["acceptable_outputs"],
                           "track": case["track"], "targets": case.get("targets", []), "candidates": candidates})
    write_jsonl(output / "blinded_review.jsonl", blind_rows)
    write_json(output / "blinding_key.json", key)
    return {name: result["overall"] for name, result in scores.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(PROJECT))
    for name in ("training-run", "heldout", "heldout-proof", "base", "adapted", "rules", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--previous", help="Saved predictions from the original v1 selected adapter on these same new inputs")
    parser.add_argument("--previous-training-run", help="Original completed v1 training directory, paired with --previous")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))
