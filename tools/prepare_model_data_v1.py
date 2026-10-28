"""Freeze assistant-reviewed original training data without producing gold labels.

The explicit, versioned review recipe supplies every target. This builder checks
the original audit eligibility, provenance and source hashes before consuming any
recipe label. It does not load a model, apply a TN policy, or open held-out data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from readcue.inference_contract import write_jsonl
from audit_research_data import check_candidates, target_hash, text_hash


VERSION = "model-data-v1"
ORIGINAL_DATASETS = {"original_pilot", "comment_regression", "reading_regression", "controls", "semantic_controls"}
TRACKS = {"core_tn", "context_reading", "preservation"}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def rows_from_bytes(body):
    return [json.loads(line) for line in body.decode("utf-8-sig").split("\n") if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def normalized_key(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def numeric_template(text):
    return re.sub(r"[0-9]+(?:\.[0-9]+)*", "<number>", normalized_key(text))


def grouped_split(rows, seed=20260928, validation_fraction=0.2):
    """Join declared templates, exact/normalized duplicates and close phrasing.

    Similarity here is a conservative split safeguard, never a target generator
    or evidence of semantic equivalence. The comparison uses source text only.
    """
    if not rows:
        raise ValueError("No reviewed original rows for training")
    parents = {row["id"]: row["id"] for row in rows}

    def find(ident):
        while parents[ident] != ident:
            parents[ident] = parents[parents[ident]]
            ident = parents[ident]
        return ident

    joins = []

    def join(left, right, reason, similarity=None):
        a, b = find(left), find(right)
        parents[max(a, b)] = min(a, b)
        record = {"left": left, "right": right, "reason": reason}
        if similarity is not None:
            record["similarity"] = similarity
        joins.append(record)

    keys = {name: {} for name in ("declared_template", "exact", "normalized", "numeric_template")}
    for row in rows:
        values = {"declared_template": row["group_id"], "exact": row["text"],
                  "normalized": normalized_key(row["text"]), "numeric_template": numeric_template(row["text"])}
        for kind, value in values.items():
            previous = keys[kind].setdefault(value, row["id"])
            if previous != row["id"]:
                join(previous, row["id"], kind)
    for index, left in enumerate(rows):
        left_key = numeric_template(left["text"])
        for right in rows[index + 1:]:
            right_key = numeric_template(right["text"])
            ratio = SequenceMatcher(None, left_key, right_key, autojunk=False).ratio()
            if ratio >= 0.82 and left_key != right_key:
                join(left["id"], right["id"], "close_phrasing_joined_conservatively", round(ratio, 6))
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["id"])].append(row)
    if len(groups) < 2:
        raise ValueError("At least two independent template groups are needed for train/validation")
    group_items = []
    for _, members in sorted(groups.items()):
        ids = sorted(row["id"] for row in members)
        group_id = "model-v1-group-" + sha("\n".join(ids).encode("utf-8"))[:16]
        group_items.append({"group_id": group_id, "ids": ids, "rows": len(ids),
                            "tracks": dict(Counter(row["track"] for row in members)),
                            "declared_templates": sorted({row["group_id"] for row in members})})
        for row in members:
            row["provenance"]["declared_template_group"] = row["group_id"]
            row["group_id"] = group_id
    total_tracks = Counter(row["track"] for row in rows)
    validation_tracks, validation_groups, count = Counter(), set(), 0
    target = round(len(rows) * validation_fraction)
    ranked = sorted(group_items, key=lambda group: sha(f"{seed}\0{group['group_id']}".encode("utf-8")))

    def fits(group):
        return (count + group["rows"] < len(rows) and count + group["rows"] <= 64
                and all(validation_tracks[track] + n < total_tracks[track]
                        for track, n in group["tracks"].items()))

    for group in ranked:
        if fits(group) and abs(count + group["rows"] - target) < abs(count - target):
            validation_groups.add(group["group_id"])
            validation_tracks.update(group["tracks"])
            count += group["rows"]
    for track in total_tracks:
        if not validation_tracks[track]:
            choices = sorted((group for group in ranked if group["group_id"] not in validation_groups
                              and group["tracks"].get(track) and fits(group)),
                             key=lambda group: (group["rows"], group["group_id"]))
            if choices:
                group = choices[0]
                validation_groups.add(group["group_id"])
                validation_tracks.update(group["tracks"])
                count += group["rows"]
    for group in group_items:
        group["assigned_role"] = "validation" if group["group_id"] in validation_groups else "train"
    splits = {"train": [], "validation": []}
    for row in rows:
        role = "validation" if row["group_id"] in validation_groups else "train"
        row["provenance"]["assigned_role"] = role
        splits[role].append(row)
    if not splits["train"] or not splits["validation"] or len(splits["train"]) > 256 or len(splits["validation"]) > 64:
        raise ValueError("Invalid or over-budget train/validation split")
    overlap = {}
    for name, key in (("id", lambda r: r["id"]), ("exact", lambda r: r["text"]),
                      ("normalized", lambda r: normalized_key(r["text"])),
                      ("numeric_template", lambda r: numeric_template(r["text"])),
                      ("group_id", lambda r: r["group_id"])):
        overlap[name] = sorted({key(row) for row in splits["train"]} & {key(row) for row in splits["validation"]})
    if any(overlap.values()):
        raise ValueError("Split contains overlapping input or template groups")
    return splits, {"method": "source_only_template_components_then_seeded_group_allocation_v1",
                    "seed": seed, "desired_validation_fraction": validation_fraction,
                    "actual_validation_fraction": len(splits["validation"]) / len(rows),
                    "allocation_reason": "Keep template/near-phrase components intact; approach 80/20 with deterministic seed, retain training coverage and add validation track coverage when feasible; never pad data.",
                    "near_duplicate_threshold": 0.82, "groups": group_items, "joins": joins,
                    "cross_role_overlap": overlap, "independent_test": False,
                    "previously_seen": True,
                    "limitation": "Both roles derive from already inspected assistant-original development data. Validation selects checkpoints and is not an independent test."}


def load_originals(audit, audit_bytes, dispositions_bytes, recipe):
    if set(recipe["source_datasets"]) != ORIGINAL_DATASETS:
        raise ValueError("This experiment must explicitly account for all five original candidate groups")
    selected, sources, rejected = [], {}, []
    indexed = {dataset["name"]: dataset for dataset in audit["datasets"]}
    dispositions = {row["id"]: row for row in rows_from_bytes(dispositions_bytes)}
    for name in recipe["source_datasets"]:
        dataset = indexed[name]
        if dataset["origin"] != "assistant_original":
            raise ValueError("Only assistant-original sources are permitted")
        case_path = Path(dataset["path"]) / "cases.jsonl"
        body = case_path.read_bytes()
        cases_hash = sha(body)
        if cases_hash != dataset["hashes"]["cases.jsonl"]:
            raise ValueError("Original cases changed: " + name)
        rows = rows_from_bytes(body)
        if len(rows) != dataset["count"]:
            raise ValueError("Original count changed: " + name)
        input_body = (case_path.parent / "inputs.jsonl").read_bytes()
        if (sha(input_body) != dataset["hashes"]["inputs.jsonl"]
                or rows_from_bytes(input_body) != [{"id": row["id"], "text": row["text"]} for row in rows]):
            raise ValueError("Original input snapshot does not match cases: " + name)
        sources[name] = {"path": str(case_path), "sha256": cases_hash, "cases_bytes": body,
                         "inputs_bytes": input_body}
        for row in rows:
            disposition = dispositions.get(row["id"])
            if disposition is None or disposition.get("cases_file_sha256") != cases_hash:
                raise ValueError("Disposition does not bind original case file: " + row["id"])
            selected.append({"row": row, "dataset": name, "cases_path": str(case_path),
                             "cases_file_sha256": cases_hash, "disposition": disposition})
    failures = check_candidates([item["row"] for item in selected], dispositions_bytes, audit)
    if any(item["id"] is None for item in failures):
        raise ValueError("Original candidate audit failed: " + json.dumps(failures))
    rejection_index = {item["id"]: item["reason"] for item in failures}
    for item in selected:
        if item["row"]["id"] in rejection_index:
            rejected.append({"id": item["row"]["id"], "dataset": item["dataset"],
                             "stage": "original_reference_eligibility",
                             "reason": rejection_index[item["row"]["id"]],
                             "text_sha256": text_hash(item["row"]["text"]),
                             "previous_refs_sha256": target_hash(item["row"]["acceptable_outputs"])})
    return selected, sources, rejected


def erratum_regression(errata, dispositions_bytes, audit):
    checks = []
    if not errata.get("items"):
        raise ValueError("Reference errata must include the withdrawn-reference regression")
    for erratum in errata["items"]:
        body = Path(erratum["cases_path"]).read_bytes()
        if sha(body) != erratum["cases_sha256"]:
            raise ValueError("Erratum source snapshot changed")
        row = next(row for row in rows_from_bytes(body) if row["id"] == erratum["id"])
        if (row["acceptable_outputs"] != erratum["old_acceptable_outputs"]
                or target_hash(row["acceptable_outputs"]) != erratum["old_acceptable_outputs_sha256"]
                or target_hash(erratum["candidate_acceptable_outputs"]) != erratum["candidate_acceptable_outputs_sha256"]
                or erratum["candidate_eligible_for_future_supervision"] is not False):
            raise ValueError("Erratum old/pending reference binding is invalid")
        for version, refs in (("withdrawn_original", row["acceptable_outputs"]),
                              ("pending_erratum_candidate", erratum["candidate_acceptable_outputs"])):
            failures = check_candidates([{**row, "acceptable_outputs": refs}], dispositions_bytes, audit)
            if not failures or failures[0]["reason"] != "withdrawn_for_future_use":
                raise ValueError("Withdrawn or pending erratum reference was incorrectly authorized")
            checks.append({"id": row["id"], "reference_version": version, "refs_sha256": target_hash(refs),
                           "expected": "reject", "actual": "reject", "candidate_check_failures": failures,
                           "used_for_training": False, "source_sha256": sha(body)})
    return checks


def build_dataset(project, audit_dir, policy_config, review_recipe, output):
    project, audit_dir, policy_config, review_recipe, output = [Path(p).resolve() for p in
                                                              (project, audit_dir, policy_config, review_recipe, output)]
    if output == project or output.is_relative_to(project) or any((parent / ".git").exists() for parent in output.parents):
        raise ValueError("Model experiment data must be outside Git checkouts")
    if output.exists():
        raise FileExistsError("Choose a new versioned output directory")
    consumed = {}

    def read_bound(name, path):
        body = path.read_bytes()
        consumed[name] = {"path": str(path), "sha256": sha(body), "bytes": len(body)}
        return body

    audit_bytes = read_bound("data_audit", audit_dir / "data-audit.json")
    audit = json.loads(audit_bytes)
    dispositions_bytes = read_bound("reference_dispositions", audit_dir / "reference-dispositions.jsonl")
    errata_bytes = read_bound("reference_errata", audit_dir / "reference-errata.json")
    errata = json.loads(errata_bytes)
    recipe_bytes = read_bound("review_recipe", review_recipe)
    recipe = json.loads(recipe_bytes)
    config_bytes = read_bound("task_config", policy_config)
    config = json.loads(config_bytes)
    policy_path = (project / config["annotation_policy"]).resolve()
    if not policy_path.is_relative_to(project):
        raise ValueError("Annotation policy must belong to this project")
    policy_bytes = read_bound("annotation_policy", policy_path)
    if (recipe.get("schema_version") != 1 or recipe.get("task_config_sha256") != sha(config_bytes)
            or recipe.get("annotation_policy_sha256") != sha(policy_bytes)
            or recipe.get("origin") != "assistant_original" or recipe.get("license") != "Apache-2.0"
            or recipe.get("reviewer_role") != "assistant" or recipe.get("human_reviewed") is not False
            or recipe.get("usage") != "first_bounded_model_research_training_validation"):
        raise ValueError("Review recipe policy, provenance or use declarations are invalid")
    originals, sources, rejected = load_originals(audit, audit_bytes, dispositions_bytes, recipe)
    checks = erratum_regression(errata, dispositions_bytes, audit)
    eligible_failures = {row["id"]: row["reason"] for row in rejected}
    decisions = recipe["decisions"]
    if (len(decisions) != len(originals) or len({row["id"] for row in decisions}) != len(decisions)
            or {row["id"] for row in decisions} != {item["row"]["id"] for item in originals}):
        raise ValueError("Recipe must explicitly review every original candidate exactly once")
    decision_index = {row["id"]: row for row in decisions}
    prepared, pending = [], []
    for original in originals:
        row = original["row"]
        decision = decision_index[row["id"]]
        if (decision.get("source_text_sha256") != text_hash(row["text"])
                or decision.get("previous_refs_sha256") != target_hash(row["acceptable_outputs"])
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
                or not isinstance(decision.get("group_id"), str) or not decision["group_id"].strip()):
            raise ValueError("Recipe is unbound or missing a review rationale: " + row["id"])
        if row["id"] in eligible_failures or decision.get("status") == "pending":
            if row["id"] in eligible_failures and decision.get("status") != "pending":
                raise ValueError("Recipe cannot restore an ineligible old reference: " + row["id"])
            pending.append({"id": row["id"], "text": row["text"], "previous_refs": row["acceptable_outputs"],
                            "group_id": decision["group_id"], "source_dataset": original["dataset"],
                            "reason": decision["reason"], "audit_rejection": eligible_failures.get(row["id"]),
                            "status": "pending", "used_for_supervision": False})
            continue
        refs = decision.get("acceptable_outputs")
        if (decision.get("status") != "assistant_policy_reviewed" or decision.get("track") not in TRACKS
                or not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
                or decision.get("target") not in refs or not isinstance(decision.get("family"), str)
                or not decision["family"].strip()):
            raise ValueError("Recipe lacks an explicit reviewed target: " + row["id"])
        if decision["track"] == "preservation" and any(ref != row["text"] for ref in refs):
            raise ValueError("Preservation targets must retain original text")
        prepared.append({"id": row["id"], "text": row["text"], "target": decision["target"],
                         "acceptable_outputs": refs, "track": decision["track"], "family": decision["family"],
                         "group_id": decision["group_id"], "provenance": {
                             "origin": "assistant_original", "license": "Apache-2.0",
                             "reference_status": "assistant_policy_reviewed", "reviewer_role": "assistant",
                             "reviewer": recipe["reviewer"], "human_reviewed": False,
                             "usage": recipe["usage"], "previously_seen": True,
                             "independent_test_eligible": False, "training_ready": True,
                             "original_audit_training_ready": original["disposition"]["training_ready"],
                             "original_audit_disposition": original["disposition"]["disposition"],
                             "source": {"dataset": original["dataset"], "cases_path": original["cases_path"],
                                        "cases_file_sha256": original["cases_file_sha256"], "original_id": row["id"],
                                        "text_sha256": text_hash(row["text"]), "previous_refs": row["acceptable_outputs"],
                                        "previous_refs_sha256": target_hash(row["acceptable_outputs"])},
                             "task_config_sha256": sha(config_bytes), "annotation_policy_sha256": sha(policy_bytes),
                             "review_recipe_sha256": sha(recipe_bytes), "modified_reason": decision["reason"],
                             "reference_changed": refs != row["acceptable_outputs"],
                             "new_refs_sha256": target_hash(refs)}})
    splits, split_report = grouped_split(prepared, seed=config["seed"])
    # All validity/eligibility checks run before creating any output.
    output.mkdir(parents=True, exist_ok=False)
    for role, rows in splits.items():
        write_jsonl(output / (role + ".jsonl"), rows, sort_keys=True)
    write_jsonl(output / "pending.jsonl", pending, sort_keys=True)
    write_jsonl(output / "rejected.jsonl", rejected, sort_keys=True)
    write_jsonl(output / "source_candidates.jsonl", [{"dataset": item["dataset"], "case": item["row"]} for item in originals], sort_keys=True)
    write_jsonl(output / "split_inputs.jsonl", [{"id": row["id"], "text": row["text"], "group_id": row["group_id"], "role": role}
                                               for role, rows in splits.items() for row in rows], sort_keys=True)
    (output / "review_recipe.json").write_bytes(recipe_bytes)
    (output / "task_config.json").write_bytes(config_bytes)
    (output / "annotation_policy.md").write_bytes(policy_bytes)
    (output / "sources").mkdir()
    for name, source in sources.items():
        for kind in ("cases", "inputs"):
            (output / "sources" / (name + "." + kind + ".jsonl")).write_bytes(source[kind + "_bytes"])
    write_json(output / "rejection_regression.json", {"checks": checks, "all_rejected": True,
                                                      "audit_pending_originals_rejected": rejected})
    write_json(output / "split_report.json", split_report)
    write_json(output / "consumption.json", {"files": consumed, "candidate_checker": "tools/audit_research_data.py:check_candidates",
                                             "original_candidates_checked": len(originals), "audit_rejections": rejected,
                                             "original_versions_checked_before_new_labels": True,
                                             "new_targets_from_explicit_recipe_only": True, "heldout_files_read": [],
                                             "new_role_assignment": "Explicit new derivative version; old audit training_ready=false remains unchanged."})
    (output / "ORIGINAL_DATA_LICENSE.md").write_text(
        "# Original derivative training and validation data\n\n"
        "ReadCue declares the assistant-original sentences and newly reviewed derivative targets in train.jsonl, validation.jsonl and the original candidate snapshots under Apache-2.0. They were authored for this project, are synthetic, and are not third-party Bilibili comments.\n\n"
        "This declaration covers only these project-original materials. It does not relicense third-party audit sources, base weights or any upstream dataset. The pending/rejected records confer no supervision approval. All old sources and scores remain unchanged. The new role assignment is for the first bounded local model research experiment, not independent testing. Labels were reviewed by an assistant, not individually approved by the user.\n",
        encoding="utf-8", newline="\n")
    (output / "code").mkdir()
    for name, path in (("prepare_model_data_v1.py", Path(__file__)),
                       ("audit_research_data.py", Path(__file__).with_name("audit_research_data.py")),
                       ("inference_contract.py", PROJECT / "src/readcue/inference_contract.py")):
        (output / "code" / name).write_bytes(path.read_bytes())
    all_files = {path.relative_to(output).as_posix(): sha(path.read_bytes())
                 for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"schema_version": 1, "dataset_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "task_config_sha256": sha(config_bytes), "annotation_policy_sha256": sha(policy_bytes),
                "roles": {"train": "train.jsonl", "validation": "validation.jsonl"}, "files_sha256": all_files,
                "origin": "assistant_original", "license": "Apache-2.0", "human_reviewed": False,
                "previously_seen": True, "independent_test": False,
                "counts": {"original_candidates": len(originals), "train": len(splits["train"]),
                           "validation": len(splits["validation"]), "pending": len(pending),
                           "audit_rejected": len(rejected),
                           "changed_reference_sets": sum(row["provenance"]["reference_changed"] for row in prepared),
                           "by_role_track": {role: dict(Counter(row["track"] for row in rows)) for role, rows in splits.items()}},
                "split": {"report": "split_report.json", "groups": len(split_report["groups"]),
                          "cross_role_overlap": split_report["cross_role_overlap"]},
                "consumed_evidence": consumed, "external_api_cost_usd": 0}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--review-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_dataset(args.project, args.audit_dir, args.policy_config, args.review_recipe, args.output)
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"], "groups": manifest["split"]["groups"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
