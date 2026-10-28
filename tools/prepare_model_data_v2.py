"""Bind reviewed v2 seed data to unchanged v1 roles; never read new test labels.

Only explicitly reviewed project-original targets are consumed. This prepares a
dataset, not a training run. Existing output directories are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_model_v2_seed as seed_check

VERSION = "model-data-v2"
USAGE = "second_bounded_model_research_training_validation"
OLD_USAGE = "first_bounded_model_research_training_validation"
SEED_NAMES = ("numeric.draft.jsonl", "context.draft.jsonl")
SKILLS = {"quantity_identifier", "decimal_reading", "percentage_reading",
          "measurement_reading", "date_time_reading", "quantity_negation",
          "initialism_reading", "english_word_boundary", "unknown_identifier",
          "reply_signature", "emoji_context", "protected_literal"}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def text_hash(text):
    return sha(text.encode("utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def jsonl_bytes(rows):
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)).encode("utf-8")


def parse_json(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(body, object_pairs_hook=unique)


def parse_rows(body):
    return [parse_json(line) for line in body.decode("utf-8-sig").splitlines() if line.strip()]


def validate_review(rows, review, sources, policy_hash):
    """Bind every accepted decision to original input and exact target bytes."""
    if (review.get("schema_version") != 1 or review.get("reviewer_role") != "assistant"
            or review.get("human_reviewed") is not False
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
            or review.get("policy", {}).get("sha256") != policy_hash):
        raise ValueError("Seed review policy or reviewer binding is invalid")
    declared = review.get("source_files", [])
    if len(declared) != len(sources) or {Path(item["path"]).name: item["sha256"] for item in declared} != sources:
        raise ValueError("Seed review source hashes changed")
    decisions = review.get("rows", [])
    if (len(decisions) != len(rows) or len({r["id"] for r in decisions}) != len(decisions)
            or {r["id"] for r in decisions} != {r["id"] for r in rows}):
        raise ValueError("Seed review must decide every candidate exactly once")
    indexed = {r["id"]: r for r in decisions}
    for row in rows:
        decision = indexed[row["id"]]
        if (decision.get("verdict") != "accept" or decision.get("reviewer_role") != "assistant"
                or decision.get("human_reviewed") is not False
                or decision.get("text_sha256") != text_hash(row["text"])
                or decision.get("target_sha256") != text_hash(row["target"])
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
            raise ValueError("Unaccepted or changed seed target: " + row["id"])
    return indexed


def connected_groups(rows):
    """Keep declared and text-similar components whole without changing roles."""
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ID across source versions")
    parents = {ident: ident for ident in ids}
    joins = []

    def find(ident):
        while parents[ident] != ident:
            parents[ident] = parents[parents[ident]]
            ident = parents[ident]
        return ident

    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            match = seed_check.match_text(left["text"], right["text"])
            declared = left["group_id"] == right["group_id"]
            if declared or match:
                a, b = find(left["id"]), find(right["id"])
                parents[max(a, b)] = min(a, b)
                joins.append({"left": left["id"], "right": right["id"],
                              "reason": "declared_group" if declared else match["match_kind"],
                              "similarity": match["similarity"] if match else None})
    components = defaultdict(list)
    for row in rows:
        components[find(row["id"])].append(row)
    report, mapping = [], {}
    for members in components.values():
        roles = {r["provenance"]["assigned_role"] for r in members}
        if len(roles) != 1:
            raise ValueError("Cross-role component; quarantine/review entire component: " + ",".join(sorted(r["id"] for r in members)))
        role = next(iter(roles))
        member_ids = sorted(r["id"] for r in members)
        group = "model-v2-group-" + text_hash("\n".join(member_ids))[:16]
        report.append({"group_id": group, "assigned_role": role, "ids": member_ids,
                       "source_groups": sorted({r["group_id"] for r in members})})
        mapping.update({ident: group for ident in member_ids})
    return mapping, {"method": "input_only_connected_components_fixed_source_roles_v2",
                     "threshold": seed_check.THRESHOLD, "groups": sorted(report, key=lambda r: r["group_id"]),
                     "joins": joins, "cross_role_overlap": {"id": [], "group_id": [], "normalized": [], "numeric_template": []},
                     "role_changes": [], "semantic_independence_proven": False}


def validate_component_coverage(seed_rows, mapping):
    """Recheck coverage after transitive joins; declared groups may collapse."""
    coverage = {}
    for skill in sorted({row["skill_id"] for row in seed_rows}):
        coverage[skill] = {}
        for role, minimum in (("train", 2), ("validation", 1)):
            rows = [row for row in seed_rows if row["skill_id"] == skill and row["proposed_role"] == role]
            components = sorted({mapping[row["id"]] for row in rows})
            transforms = sum(row["operation"] == "transform" for row in rows)
            if len(components) < minimum or transforms == 0:
                raise ValueError(f"Insufficient merged-component coverage: {skill}/{role}; "
                                 f"components={len(components)}, required={minimum}, transforms={transforms}")
            coverage[skill][role] = {"components": components, "component_count": len(components),
                                     "rows": len(rows), "transform_rows": transforms}
    return coverage


def validate_old_rows(rows, role, task_hash, policy_hash):
    for row in rows:
        required = {"id", "text", "target", "acceptable_outputs", "track", "family", "group_id", "provenance"}
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("Malformed v1 row")
        p = row["provenance"]
        if (p.get("origin") != "assistant_original" or p.get("license") != "Apache-2.0"
                or p.get("usage") != OLD_USAGE or p.get("assigned_role") != role
                or p.get("training_ready") is not True or p.get("human_reviewed") is not False
                or p.get("reference_status") != "assistant_policy_reviewed"
                or p.get("task_config_sha256") != task_hash or p.get("annotation_policy_sha256") != policy_hash
                or row["target"] not in row["acceptable_outputs"]):
            raise ValueError("Unregistered v1 row: " + row["id"])


def build_dataset(project, v1_dataset, seed_dir, policy_config, old_heldout_inputs, output):
    project, v1_dataset, seed_dir, policy_config, old_heldout_inputs, output = [Path(p).resolve() for p in
        (project, v1_dataset, seed_dir, policy_config, old_heldout_inputs, output)]
    if output.exists():
        raise FileExistsError("Choose a new versioned dataset output directory")
    staging = output.is_relative_to(project)
    if staging:
        if (not output.is_relative_to(project / "runs")
                or subprocess.run(["git", "-C", str(project), "check-ignore", "--quiet", "--no-index",
                                   str(output / "manifest.json")], check=False).returncode != 0):
            raise ValueError("Within a checkout, only a Git-ignored runs staging directory is allowed")
    elif any((parent / ".git").exists() for parent in output.parents):
        raise ValueError("Dataset cannot be written inside another Git checkout")
    consumed, snapshots = {}, {}

    def read(name, path):
        body = path.read_bytes()
        consumed[name] = {"path": str(path), "sha256": sha(body), "bytes": len(body)}
        snapshots[name] = body
        return body

    config_body = read("task_config.json", policy_config)
    config = parse_json(config_body)
    if config.get("task_version") != "zh-reading-model-v1":
        raise ValueError("v2 data retains the unchanged v1 task contract")
    policy_path = (project / config["annotation_policy"]).resolve()
    if not policy_path.is_relative_to(project):
        raise ValueError("Policy must belong to the project")
    policy_body = read("annotation_policy.md", policy_path)
    task_hash, policy_hash = sha(config_body), sha(policy_body)
    manifest_body = read("v1/manifest.json", v1_dataset / "manifest.json")
    original_manifest = parse_json(manifest_body)
    if (original_manifest.get("schema_version") != 1 or original_manifest.get("dataset_version") != "model-data-v1"
            or original_manifest.get("roles") != {"train": "train.jsonl", "validation": "validation.jsonl"}
            or original_manifest.get("task_config_sha256") != task_hash
            or original_manifest.get("annotation_policy_sha256") != policy_hash):
        raise ValueError("v1 dataset task/policy binding mismatch")
    original, prepared = {}, []
    for name in ("train.jsonl", "validation.jsonl", "pending.jsonl", "rejected.jsonl", "ORIGINAL_DATA_LICENSE.md"):
        body = read("v1/" + name, v1_dataset / name)
        if sha(body) != original_manifest.get("files_sha256", {}).get(name):
            raise ValueError("v1 source hash changed: " + name)
        if name not in ("train.jsonl", "validation.jsonl"):
            continue
        role = name[:-6]
        rows = parse_rows(body)
        validate_old_rows(rows, role, task_hash, policy_hash)
        if len(rows) != original_manifest.get("counts", {}).get(role):
            raise ValueError("v1 role count mismatch")
        original[role] = rows
        for row in rows:
            new = deepcopy(row)
            new["provenance"] = {**new["provenance"], "previous_version": deepcopy(row["provenance"]),
                "usage": USAGE, "source_version": "model-data-v1", "source_row_sha256": sha(json_bytes(row)),
                "source_manifest_sha256": sha(manifest_body), "source_file_sha256": sha(body),
                "targets_unchanged_from_source": True, "role_unchanged_from_source": True}
            prepared.append(new)
    seed_rows, source_hashes = [], {}
    for name in SEED_NAMES:
        body = read("seed/" + name, seed_dir / name)
        source_hashes[name] = sha(body)
        seed_rows.extend(parse_rows(body))
    if len(seed_rows) != 72 or {r.get("skill_id") for r in seed_rows} != SKILLS:
        raise ValueError("This v2 seed release must account for 72 rows and all 12 skills")
    review_body = read("seed/assistant_review.json", seed_dir / "assistant_review.json")
    review = parse_json(review_body)
    decisions = validate_review(seed_rows, review, source_hashes, policy_hash)
    old_inputs_body = read("old_heldout.inputs.jsonl", old_heldout_inputs)
    heldout, _ = seed_check.load_heldout_inputs(old_heldout_inputs)
    if sha(old_heldout_inputs.read_bytes()) != sha(old_inputs_body):
        raise ValueError("Historical inputs changed during read")
    history = [{"id": r["id"], "text": r["text"], "proposed_role": role}
               for role, rows in original.items() for r in rows]
    fresh_check = seed_check.audit_rows(seed_rows, historical=history, heldout=heldout)
    if fresh_check["blockers"]:
        raise ValueError("Seed structure, coverage or input isolation blocked: " + json.dumps(fresh_check["blockers"]))
    receipt = parse_json(read("seed/check.reviewed.json", seed_dir / "check.reviewed.json"))
    if (receipt.get("status") != "draft_checks_passed" or receipt.get("blockers")
            or receipt.get("checker_sha256") != sha(Path(seed_check.__file__).read_bytes())):
        raise ValueError("Seed check receipt is not bound to this successful checker version")
    expected = {consumed[name]["sha256"] for name in ("v1/manifest.json", "v1/train.jsonl", "v1/validation.jsonl",
                 "seed/numeric.draft.jsonl", "seed/context.draft.jsonl", "old_heldout.inputs.jsonl")}
    if {item["sha256"] for item in receipt.get("input_evidence", [])} != expected:
        raise ValueError("Seed check receipt source identities changed")
    for row in seed_rows:
        p = {**row["provenance"], "usage": USAGE, "training_ready": True,
             "reference_status": "assistant_policy_reviewed", "assigned_role": row["proposed_role"],
             "previously_seen": True, "task_config_sha256": task_hash, "annotation_policy_sha256": policy_hash,
             "reviewer": review["reviewer"], "reviewer_role": "assistant", "review_sha256": sha(review_body),
             "review_decision": decisions[row["id"]], "original_draft_training_ready": row["training_ready"],
             "source_version": "v2-development-seed-2026-10-27", "source_row_sha256": sha(json_bytes(row)),
             "reference_changed": False, "role_unchanged_from_source": True}
        prepared.append({"id": row["id"], "text": row["text"], "target": row["target"],
                         "acceptable_outputs": row["acceptable_outputs"], "track": row["track"],
                         "family": row["skill_id"], "skill_id": row["skill_id"],
                         "group_id": row["group_id"], "provenance": p})
    mapping, split_report = connected_groups(prepared)
    component_coverage = validate_component_coverage(seed_rows, mapping)
    split_report["new_seed_component_coverage"] = component_coverage
    for row in prepared:
        row["provenance"]["source_group_id"] = row["group_id"]
        row["group_id"] = mapping[row["id"]]
    splits = {role: [r for r in prepared if r["provenance"]["assigned_role"] == role] for role in ("train", "validation")}
    for role, cap in (("train", 256), ("validation", 64)):
        if not 0 < len(splits[role]) <= cap:
            raise ValueError("Role count outside fixed limit: " + role)
        keys = [seed_check.normalized_key(r["text"]) for r in splits[role]]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate normalized source in " + role)
    # Validate all inputs and decisions before creating even an empty output.
    payloads = {role + ".jsonl": jsonl_bytes(rows) for role, rows in splits.items()}
    payloads.update({"task_config.json": config_body, "annotation_policy.md": policy_body,
        "split_inputs.jsonl": jsonl_bytes([{"id": r["id"], "text": r["text"], "group_id": r["group_id"], "role": role}
                                          for role, rows in splits.items() for r in rows]),
        "inputs.jsonl": jsonl_bytes([{"id": r["id"], "text": r["text"]} for r in prepared]),
        "split_report.json": json_bytes(split_report), "seed_check.json": json_bytes(fresh_check),
        "coverage.json": json_bytes({"new_seed": fresh_check["coverage"],
            "new_seed_after_component_merge": component_coverage, "combined_by_role_track": {
            role: dict(sorted(Counter(r["track"] for r in rows).items())) for role, rows in splits.items()},
            "combined_by_role_family": {role: dict(sorted(Counter(r["family"] for r in rows).items())) for role, rows in splits.items()}}),
        "consumption.json": json_bytes({"files": consumed, "seed_decisions_consumed": len(decisions),
            "old_targets_and_roles_unchanged": True, "new_heldout_files_read": [],
            "old_heldout_use": "input-only anti-reuse screening; no labels or predictions",
            "labels_source": "explicit v1 registered targets and bound assistant-reviewed new seed targets only"}),
        "ORIGINAL_DATA_LICENSE.md": b"# Original derivative data\n\nApache-2.0 covers these project-original synthetic sentences and assistant-reviewed targets only. No third-party comments or model predictions are used as supervision. Labels are not individual human approvals. This dataset serves the second bounded research training/validation experiment, not independent testing. Original snapshots, roles and targets are retained.\n"})
    for name, body in snapshots.items():
        payloads["sources/" + name] = body
    for name, path in (("prepare_model_data_v2.py", Path(__file__)), ("check_model_v2_seed.py", Path(seed_check.__file__))):
        payloads["code/" + name] = path.read_bytes()
    manifest = {"schema_version": 1, "dataset_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_config_sha256": task_hash, "annotation_policy_sha256": policy_hash,
        "source_v1_manifest_sha256": sha(manifest_body), "usage": USAGE,
        "roles": {"train": "train.jsonl", "validation": "validation.jsonl"},
        "files_sha256": {name: sha(body) for name, body in sorted(payloads.items())},
        "origin": "assistant_original", "license": "Apache-2.0", "human_reviewed": False,
        "previously_seen": True, "independent_test": False, "training_ready": True,
        "counts": {"train": len(splits["train"]), "validation": len(splits["validation"]),
                   "v1_train": len(original["train"]), "v1_validation": len(original["validation"]),
                   "new_seed": len(seed_rows), "pending_v1_excluded": len(parse_rows(snapshots["v1/pending.jsonl"])),
                   "targets_changed": 0, "roles_changed": 0,
                   "by_role_track": {role: dict(Counter(r["track"] for r in rows)) for role, rows in splits.items()}},
        "split": {"report": "split_report.json", "groups": len(split_report["groups"]),
                  "cross_role_overlap": split_report["cross_role_overlap"]},
        "consumed_evidence": consumed, "external_api_cost_usd": 0,
        "storage": {"git_ignored_preparation_staging": staging, "training_budget_root": False},
        "limitations": "Reviewed synthetic development data; heuristic isolation does not establish semantic independence. No new test labels were read, no model was trained."}
    payloads["manifest.json"] = json_bytes(manifest)
    output.mkdir(parents=True, exist_ok=False)
    for name, body in payloads.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(body)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--v1-dataset", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, default=PROJECT / "eval/zh_reading_model_v2_seed")
    parser.add_argument("--policy-config", type=Path, default=PROJECT / "configs/model_task_v1.json")
    parser.add_argument("--old-heldout-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_dataset(args.project, args.v1_dataset, args.seed_dir, args.policy_config, args.old_heldout_inputs, args.output)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "prepared", "output": str(args.output), "counts": result["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
