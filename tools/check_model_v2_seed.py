"""Audit original v2 development drafts without exporting a training dataset.

Similarity uses input text only. Passing this audit does not approve targets,
establish semantic independence, or turn development samples into a blind test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

PROJECT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.82
FIELDS = {"id", "skill_id", "group_id", "proposed_role", "text", "target",
          "acceptable_outputs", "track", "operation", "rationale", "provenance",
          "training_ready"}
PROVENANCE = {"origin": "assistant_original", "author_role": "assistant",
              "human_reviewed": False, "license": "Apache-2.0",
              "intended_use": "v2_development_seed", "independent_test_eligible": False}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def read_rows(path):
    body = Path(path).read_bytes()
    rows = [json.loads(line) for line in body.decode("utf-8-sig").splitlines() if line.strip()]
    return rows, {"path": str(path), "sha256": sha(body), "rows": len(rows)}


def normalized_key(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def numeric_template(text):
    return re.sub(r"[0-9]+(?:\.[0-9]+)*", "<number>", normalized_key(text))


def match_text(left, right):
    if left == right:
        return {"match_kind": "exact", "similarity": 1.0}
    if normalized_key(left) == normalized_key(right):
        return {"match_kind": "normalized", "similarity": 1.0}
    a, b = numeric_template(left), numeric_template(right)
    if a == b:
        return {"match_kind": "numeric_template", "similarity": 1.0}
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    if ratio >= THRESHOLD:
        return {"match_kind": "near_phrase", "similarity": round(ratio, 6)}
    return None


def validate_row(row, index):
    if not isinstance(row, dict):
        return [f"row {index}: expected object"]
    errors = []
    ident = row.get("id", f"row {index}")
    if set(row) != FIELDS:
        errors.append(f"{ident}: unexpected or missing fields")
    for field in ("id", "skill_id", "group_id", "text", "target", "rationale"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"{ident}: {field} must be a nonempty string")
    for field, allowed in (("proposed_role", {"train", "validation"}),
                           ("track", {"core_tn", "context_reading", "preservation"}),
                           ("operation", {"transform", "preserve"})):
        if not isinstance(row.get(field), str) or row[field] not in allowed:
            errors.append(f"{ident}: invalid {field}")
    outputs = row.get("acceptable_outputs")
    if (not isinstance(outputs, list) or not outputs
            or any(not isinstance(value, str) or not value.strip() for value in outputs)
            or row.get("target") not in outputs):
        errors.append(f"{ident}: acceptable_outputs must contain the nonempty target")
    if row.get("operation") == "preserve" and row.get("target") != row.get("text"):
        errors.append(f"{ident}: preserve must leave text unchanged")
    if row.get("operation") == "transform" and row.get("target") == row.get("text"):
        errors.append(f"{ident}: transform must change text")
    if row.get("track") == "preservation" and row.get("operation") != "preserve":
        errors.append(f"{ident}: preservation track requires preserve operation")
    if row.get("training_ready") is not False:
        errors.append(f"{ident}: draft training_ready must be false")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or any(
            provenance.get(key) != value or type(provenance.get(key)) is not type(value)
            for key, value in PROVENANCE.items()):
        errors.append(f"{ident}: invalid development-draft provenance")
    return errors


def load_v1_dataset(directory):
    root = Path(directory)
    manifest_body = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_body)
    if manifest.get("dataset_version") != "model-data-v1":
        raise ValueError("Expected model-data-v1 manifest")
    records, evidence = [], [{"path": str(root / "manifest.json"), "sha256": sha(manifest_body)}]
    for role in ("train", "validation"):
        filename = role + ".jsonl"
        rows, source = read_rows(root / filename)
        if source["sha256"] != manifest.get("files_sha256", {}).get(filename):
            raise ValueError(f"v1 manifest hash mismatch: {filename}")
        evidence.append(source)
        for row in rows:
            if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in ("id", "text")):
                raise ValueError(f"Invalid v1 source input in {filename}")
            records.append({"id": row["id"], "text": row["text"], "proposed_role": role})
    return records, evidence


def load_heldout_inputs(path):
    rows, evidence = read_rows(path)
    ids = set()
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"id", "text"}
                or not all(isinstance(row[k], str) and row[k].strip() for k in ("id", "text"))):
            raise ValueError("Historical heldout file must contain only nonempty id/text; gold fields are forbidden")
        if row["id"] in ids:
            raise ValueError("Duplicate historical heldout input ID")
        ids.add(row["id"])
    if not rows:
        raise ValueError("Historical heldout inputs cannot be empty")
    return rows, evidence


def audit_rows(rows, *, historical=None, heldout=None, input_evidence=None):
    blockers, warnings, valid = [], [], []
    for index, row in enumerate(rows, 1):
        errors = validate_row(row, index)
        blockers.extend({"kind": "schema", "message": error} for error in errors)
        if not errors:
            valid.append(row)
    if not rows:
        blockers.append({"kind": "empty_samples"})
    ids = Counter(row["id"] for row in valid)
    blockers.extend({"kind": "duplicate_id", "id": key} for key, count in ids.items() if count > 1)
    groups = defaultdict(set)
    skills = defaultdict(list)
    for row in valid:
        groups[row["group_id"]].add(row["proposed_role"])
        skills[row["skill_id"]].append(row)
    for group, roles in sorted(groups.items()):
        if len(roles) > 1:
            blockers.append({"kind": "group_cross_role", "group_id": group, "roles": sorted(roles)})
    coverage = {}
    for skill, members in sorted(skills.items()):
        coverage[skill] = {}
        operations = {row["operation"] for row in members}
        if "transform" not in operations or (skill != "quantity_identifier" and "preserve" not in operations):
            blockers.append({"kind": "missing_operation_coverage", "skill_id": skill})
        for role, minimum in (("train", 2), ("validation", 1)):
            selected = [row for row in members if row["proposed_role"] == role]
            role_groups = sorted({row["group_id"] for row in selected})
            coverage[skill][role] = {"rows": len(selected), "groups": len(role_groups),
                                     "group_ids": role_groups,
                                     "operations": dict(sorted(Counter(row["operation"] for row in selected).items()))}
            if len(role_groups) < minimum:
                blockers.append({"kind": "insufficient_groups", "skill_id": skill,
                                 "role": role, "required": minimum, "actual": len(role_groups)})
            if not any(row["operation"] == "transform" for row in selected):
                blockers.append({"kind": "missing_role_transform_coverage", "skill_id": skill,
                                 "role": role})
    for index, left in enumerate(valid):
        for right in valid[index + 1:]:
            match = match_text(left["text"], right["text"])
            if not match:
                continue
            item = {"left": left["id"], "right": right["id"], **match}
            if match["match_kind"] == "exact":
                blockers.append({"kind": "duplicate_text", **item})
            elif left["proposed_role"] != right["proposed_role"]:
                blockers.append({"kind": "seed_cross_role_similarity", **item})
            elif left["group_id"] != right["group_id"]:
                warnings.append({"kind": "seed_same_role_similarity", **item})
    for left in valid:
        for right in historical or []:
            match = match_text(left["text"], right["text"])
            if match:
                cross_role = left["proposed_role"] != right["proposed_role"]
                item = {"kind": "v1_cross_role_similarity" if cross_role else "v1_same_role_similarity",
                        "left": left["id"], "right": right["id"], **match}
                (blockers if cross_role else warnings).append(item)
        for right in heldout or []:
            match = match_text(left["text"], right["text"])
            if match:
                blockers.append({"kind": "old_heldout_similarity", "left": left["id"],
                                 "right": right["id"], **match})
    return {"schema_version": 1, "status": "blocked" if blockers else "draft_checks_passed",
            "checker_sha256": sha(Path(__file__).read_bytes()),
            "training_ready": False, "independent_test_eligible": False,
            "semantic_review_complete": False,
            "similarity_method": "input_only_nfkc_casefold_no_whitespace_numeric_template_sequence_matcher",
            "similarity_threshold": THRESHOLD,
            "limitations": "Heuristic duplicate screening does not establish semantic independence or target correctness. No training export or test freeze is produced.",
            "counts": {"total": len(rows), "valid_schema": len(valid), "skills": len(skills),
                       "groups": len(groups), "by_role": dict(sorted(Counter(row["proposed_role"] for row in valid).items())),
                       "by_track": dict(sorted(Counter(row["track"] for row in valid).items()))},
            "coverage": coverage, "history_checks": {
                "v1_train_validation": {"performed": historical is not None, "rows": len(historical or [])},
                "old_heldout_inputs_only": {"performed": heldout is not None, "rows": len(heldout or [])}},
            "input_evidence": input_evidence or [], "blockers": blockers, "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, action="append")
    parser.add_argument("--v1-dataset", type=Path)
    parser.add_argument("--v1-heldout-inputs", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON report file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        paths = args.samples or [PROJECT / "eval/zh_reading_model_v2_seed" / name
                                for name in ("numeric.draft.jsonl", "context.draft.jsonl")]
        rows, evidence = [], []
        for path in paths:
            batch, source = read_rows(path)
            rows.extend(batch)
            evidence.append(source)
        historical = heldout = None
        if args.v1_dataset:
            historical, sources = load_v1_dataset(args.v1_dataset)
            evidence.extend(sources)
        if args.v1_heldout_inputs:
            heldout, source = load_heldout_inputs(args.v1_heldout_inputs)
            evidence.append(source)
        result = audit_rows(rows, historical=historical, heldout=heldout, input_evidence=evidence)
        body = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(body)
            print(json.dumps({"status": result["status"], "output": str(args.output),
                              "blockers": len(result["blockers"]), "warnings": len(result["warnings"])}, ensure_ascii=False))
        else:
            print(body, end="")
        return 2 if result["blockers"] else 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"status": "error", "training_ready": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
