"""Bounded audit of the already completed ReadCue exploration, without rescoring.

The default audit writes a NEW directory. --check-cases rejects known withdrawn,
pending, changed, or untracked references; a pass is not training/license approval.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import wave


DEFECT_ID = "bilibili-job-change-ai:BV1iofBYsEyc:252677141809"
V4 = "semantic-context-v4-2026-10-01"
FEEDBACK_RUNS = (
    "reading-policy-v2-listening-2026-10-01",
    "context-reading-v3-listening-2026-10-01",
    "semantic-context-v4-listening-2026-10-01",
)
REAL_GROUPS = {"development", "seen_real_24", "seen_supplement_6", "new_real_8"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").split("\n") if line.strip()]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_hash(refs):
    encoded = json.dumps(refs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_candidates(rows, dispositions):
    """Fail closed on untracked/changed references, not only the known bad string."""
    if not isinstance(rows, list) or not rows:
        return [{"id": None, "reason": "empty_or_invalid_candidate_set"}]
    index = {row["id"]: row for row in dispositions}
    failures = []
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip()
                or not isinstance(row.get("text"), str) or not row["text"].strip()
                or not isinstance(row.get("acceptable_outputs"), list)
                or any(not isinstance(ref, str) or not ref.strip() for ref in row["acceptable_outputs"])):
            failures.append({"id": row.get("id") if isinstance(row, dict) else None, "reason": "invalid_candidate_row"})
            continue
        item = index.get(row["id"])
        if row["id"] in seen:
            reason = "duplicate_candidate_id"
        elif item is None:
            reason = "untracked_reference_requires_review"
        elif text_hash(row["text"]) != item["text_sha256"]:
            reason = "source_text_changed_since_audit"
        elif not item["eligible_as_reference_candidate"]:
            reason = item["disposition"]
        elif target_hash(row.get("acceptable_outputs", [])) != item["acceptable_outputs_sha256"]:
            reason = "reference_changed_since_audit"
        else:
            seen.add(row["id"])
            continue
        seen.add(row["id"])
        failures.append({"id": row.get("id"), "reason": reason})
    return failures


def inspect_manifest(path):
    obj = read_json(path)
    failures = []
    for name, expected in obj["files"].items():
        item = path.parent / name
        if not item.is_file() or file_hash(item) != expected["sha256"] or item.stat().st_size != expected["bytes"]:
            failures.append(name)
    return {"path": str(path), "sha256": file_hash(path), "checked_files": len(obj["files"]), "failures": failures}


def audit(runtime, project):
    run = runtime / "runs" / V4
    freeze = read_json(run / "freeze.json")
    baseline_manifests = runtime / "runs/work-review-2026-10-01/archive-manifests-before.json"
    archives = []
    for path_string, expected in read_json(baseline_manifests).items():
        item = inspect_manifest(Path(path_string))
        item["matches_prechange_manifest_hash"] = item["sha256"] == expected["sha256"]
        archives.append(item)
    export = Path(freeze["datasets"]["development"]["path"])
    export_integrity = inspect_manifest(export / "manifest.json")
    datasets, dispositions, all_rows = [], [], []
    frozen_dataset_hashes_ok = True
    groups = dict(freeze["datasets"])
    groups["original_pilot"] = {"path": str(project / "eval/zh_reading_pilot_v1"), "count": 72, "hashes": {}}
    for name, info in groups.items():
        folder = Path(info["path"])
        rows = read_rows(folder / "cases.jsonl")
        inputs = read_rows(folder / "inputs.jsonl")
        assert len(rows) == info["count"] == len(inputs)
        assert [{"id": r["id"], "text": r["text"]} for r in rows] == inputs
        hashes = {filename: file_hash(folder / filename) for filename in ("inputs.jsonl", "cases.jsonl")}
        hashes_ok = all(hashes[n] == h for n, h in info["hashes"].items())
        frozen_dataset_hashes_ok &= hashes_ok
        dataset = {"name": name, "path": str(folder), "count": len(rows), "nonempty_reference_count": sum(bool(r["acceptable_outputs"]) for r in rows),
                   "pending_reference_count": sum(not r["acceptable_outputs"] for r in rows), "hashes": hashes,
                   "matches_run_freeze": hashes_ok if info["hashes"] else "pilot_separate_archive", "origin": "real_comment" if name in REAL_GROUPS else "assistant_original",
                   "user_approved_text_count": len(rows) if name == "development" else 0,
                   "independent_test_eligible_count": 0, "assigned_training_count": 0}
        datasets.append(dataset)
        for row in rows:
            pending = not row["acceptable_outputs"] or row.get("eligible_for_provisional_em") is False
            status = "withdrawn_for_future_use" if row["id"] == DEFECT_ID else "needs_review" if pending else "retained_exploratory_candidate"
            dispositions.append({"id": row["id"], "dataset": name, "cases_file_sha256": hashes["cases.jsonl"],
                                 "text_sha256": text_hash(row["text"]),
                                 "acceptable_outputs_sha256": target_hash(row["acceptable_outputs"]), "disposition": status,
                                 "eligible_as_reference_candidate": status == "retained_exploratory_candidate",
                                 "independent_test_eligible": False, "training_ready": False,
                                 "blockers": (["source_rights_unresolved"] if name in REAL_GROUPS else []) + ["no_training_assignment_or_split", "task_contract_review_needed"],
                                 "human_text_review": name == "development"})
        all_rows.extend(rows)
    assert len({r["id"] for r in all_rows}) == len(all_rows)
    defect = next(r for r in all_rows if r["id"] == DEFECT_ID)
    assert len(defect["acceptable_outputs"]) == 1 and defect["acceptable_outputs"][0].count("[吃瓜]") == 1
    candidate = [defect["acceptable_outputs"][0].replace("[吃瓜]", "")]
    errata = {"schema_version": 1, "version": "reference-errata-v1", "scope": "proposed overlay only; old packages and first-run scores unchanged",
              "items": [{"id": DEFECT_ID, "dataset": "new_real_8", "dataset_index": 5,
                         "cases_path": str(Path(groups["new_real_8"]["path"]) / "cases.jsonl"), "cases_sha256": file_hash(Path(groups["new_real_8"]["path"]) / "cases.jsonl"),
                         "old_acceptable_outputs": defect["acceptable_outputs"], "old_acceptable_outputs_sha256": target_hash(defect["acceptable_outputs"]),
                         "old_reference_disposition": "withdrawn_for_future_use", "old_eligible_for_future_supervision": False,
                         "candidate_acceptable_outputs": candidate, "candidate_acceptable_outputs_sha256": target_hash(candidate),
                         "candidate_status": "assistant_corrected_pending_review", "candidate_eligible_for_future_supervision": False,
                         "human_reviewed": False, "reason": "原参考误留平台表情 [吃瓜]，与既定省略政策及该条原有标注说明冲突。候选只去掉这一标记，其余正文不变。",
                         "policy_evidence": "docs/AI_PREANNOTATION.md:31; frozen annotation notes for this item",
                         "not_derived_from_prediction": "Candidate is the single-token policy correction; both frozen model outputs independently agree, but agreement is not human approval.",
                         "first_run_scores_unchanged": True, "rescored": False}]}
    inputs_checked = []
    for relative in ("combined.inputs.jsonl", "new.inputs.jsonl", "pipeline_v4/transformed_inputs.jsonl", "v3_1_new/transformed_inputs.jsonl"):
        path = run / relative
        rows = read_rows(path)
        inputs_checked.append({"path": str(path), "sha256": file_hash(path), "count": len(rows), "extra_field_rows": sum(set(r) != {"id", "text"} for r in rows)})
    feedback, feedback_failures = [], []
    for name in FEEDBACK_RUNS:
        path = runtime / "runs" / name / "feedback-001.json"
        obj = read_json(path)
        clips = obj.get("clips", [{"wav_path": obj.get("original_audio_path"), "wav_sha256": obj.get("original_audio_sha256"), "clip_id": "jdk-original", "text": obj.get("original_text")}])
        checked = []
        for clip in clips:
            actual = file_hash(clip["wav_path"])
            ok = actual == clip["wav_sha256"]
            if "text_sha256" in clip:
                ok &= hashlib.sha256(clip["text"].encode("utf-8")).hexdigest() == clip["text_sha256"]
            if "pcm_sha256" in clip:
                with wave.open(clip["wav_path"], "rb") as wav:
                    ok &= hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest() == clip["pcm_sha256"]
            if not ok:
                feedback_failures.append(clip["clip_id"])
            checked.append({"clip_id": clip["clip_id"], "wav_path": clip["wav_path"], "wav_sha256": actual, "hashes_match": ok})
        feedback.append({"path": str(path), "sha256": file_hash(path), "verbatim_feedback": obj["verbatim_feedback"], "scope": obj["scope_resolution"], "clips": checked})
    before_path = runtime / "data/bilibili-pipeline-2026-09-29/finalize-review-2026-10-01/annotations.jsonl.before"
    final_path = runtime / "data/bilibili-pipeline-2026-09-29/prepare-02/annotations.jsonl"
    before, after = read_rows(before_path), read_rows(final_path)
    assert [r["id"] for r in before] == [r["id"] for r in after]
    changes = [{"row": n, "id": a["id"], "changed_review_fields": sorted(k for k in a["review"] if a["review"][k] != b["review"][k])}
               for n, (a, b) in enumerate(zip(before, after), 1) if a != b]
    readings_unchanged = all(a["review"]["acceptable_outputs"] == b["review"]["acceptable_outputs"] for a, b in zip(before, after))
    verification = read_json(run / "scoring/verification.json")
    time_order_ok = all(datetime.fromisoformat(t) > datetime.fromisoformat(verification["batch_finished_at_utc"]) for t in verification["new_reference_content_read_by_root_at_utc"].values())
    failures = [a["path"] for a in archives if a["failures"] or not a["matches_prechange_manifest_hash"]]
    export_matches_current = file_hash(final_path) == file_hash(export / "submitted_annotations.jsonl")
    integrity_checks = {"completed_archives_unchanged": not failures, "export_archive_unchanged": not export_integrity["failures"],
                        "feedback_hashes_match": not feedback_failures, "frozen_dataset_hashes_match": frozen_dataset_hashes_ok,
                        "inference_schema_gold_free": not any(i["extra_field_rows"] for i in inputs_checked),
                        "references_read_after_batch_complete": time_order_ok, "export_matches_current_annotations": export_matches_current,
                        "finalization_readings_unchanged": readings_unchanged}
    integrity_ok = all(integrity_checks.values())
    result = {"schema_version": 1, "status": "verified_with_reference_erratum" if integrity_ok else "failed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Recent completed exploration, original 72-item pilot, frozen datasets, inference interfaces, and five accepted listening clips. No training, rescoring, or new audio judgment.",
              "critical_integrity_checks": integrity_checks, "archives": archives, "export_integrity": export_integrity, "datasets": datasets,
              "inventory": {"total_case_rows_excluding_demos": len(all_rows), "real_comment_rows": sum(d["count"] for d in datasets if d["origin"] == "real_comment"),
                            "assistant_original_rows": sum(d["count"] for d in datasets if d["origin"] == "assistant_original"), "nonempty_reference_rows": sum(d["nonempty_reference_count"] for d in datasets),
                            "pending_reference_rows": sum(d["pending_reference_count"] for d in datasets), "withdrawn_nonempty_references": 1,
                            "retained_nonempty_candidates": sum(d["eligible_as_reference_candidate"] for d in dispositions),
                            "human_approved_text_rows": 138, "excluded_original_review_rows": 2, "independent_test_ready_rows": 0, "assigned_training_ready_rows": 0,
                            "interpretation": "0 assigned training rows means no completed task-contract/rights/split assignment, not that project-original data cannot be used for training. 72 pilot rows have explicit Apache-2.0 data terms; 88 later original controls are provenance-recorded candidates. Do not combine legacy preservation targets with newer expansion targets without a versioned contract review."},
              "annotation_finalization": {"before_sha256": file_hash(before_path), "after_sha256": file_hash(final_path), "export_matches_current": export_matches_current,
                                          "changes": changes, "readings_unchanged": readings_unchanged, "decision_counts": dict(Counter(r["review"]["status"] for r in after)),
                                          "authorization_scope": "Rows 10 and 18 approved as explicitly requested; row 74 only synchronized obsolete notes on already approved alternatives. No output text changed.",
                                          "reviewer_field_limit": "Stored user attestations, not independent proof of human attention or speech accuracy."},
              "inference_isolation": {"inputs_checked": inputs_checked, "new_reference_read_after_batch_complete": time_order_ok, "scoring_verification_sha256": file_hash(run / "scoring/verification.json"),
                                      "static_evidence": ["frozen source/src/readcue/baselines.py:11 validates exact id/text keys", "frozen source/scripts/run_semantic_context_pipeline.py:114,202,218 builds only text inputs for TN", "runs/score_semantic_v4.py:43-46 loads new references after completed batch"],
                                      "conclusion": "No evidence of evaluation label fields entering these inference paths. Hashing case files for freeze integrity is not label use. Prior seen examples informing development policy prevents independent-test claims."},
              "listening_feedback": {"records": feedback, "confirmed_exact_clips": 5, "scope_expansion_found": False,
                                     "limits": "1 original JDK demo; 2 normalized v3 excerpts; 2 normalized v4 full outputs. No relative preference, general pronunciation accuracy, all-row approval, reference approval or blanket emoji wording preference follows."},
              "findings": [{"id": "reference_real8_05", "severity": "reference_defect", "action": "Withdraw original [吃瓜] reference from future supervision; correction remains assistant proposal pending review. Preserve frozen first-run scores."},
                           {"id": "training_and_test_roles", "severity": "scope_limit", "action": "Use existing seen data as development/regression candidates. Review target contract, establish source rights and a separately assigned training split; acquire independent evaluation material without using it to tune."},
                           {"id": "legacy_contracts", "severity": "target_contract_review_needed", "action": "The original pilot preserves yyds/English/model strings while later reading references may spell or expand them. Do not concatenate these targets under one contract without versioned review."}],
              "mutations": "Only new audit/errata/dispositions files; no frozen artifacts or scores changed", "training": False, "external_api_cost_usd": 0}
    return result, errata, dispositions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path("E:/CodexRuntime/ReadCue"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--check-cases", type=Path)
    parser.add_argument("--dispositions", type=Path)
    args = parser.parse_args()
    if args.check_cases:
        if not args.dispositions:
            parser.error("--check-cases requires --dispositions")
        try:
            failures = check_candidates(read_rows(args.check_cases), read_rows(args.dispositions))
        except (ValueError, TypeError, KeyError, OSError) as error:
            failures = [{"id": None, "reason": "invalid_audit_or_candidate_file", "error_type": type(error).__name__}]
        print(json.dumps({"status": "rejected" if failures else "no_known_withdrawn_reference", "failures": failures,
                          "training_authorized": False, "note": "This only checks recorded reference dispositions, not license, policy compatibility or train/test isolation."}))
        return 2 if failures else 0
    if not args.output_directory:
        parser.error("--output-directory is required and must not exist")
    if args.output_directory.exists():
        parser.error("Refusing to overwrite an existing audit directory")
    result, errata, dispositions = audit(args.runtime_root, args.project)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    for name, obj in (("data-audit.json", result), ("reference-errata.json", errata)):
        with (args.output_directory / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    with (args.output_directory / "reference-dispositions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for item in dispositions:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "inventory": result["inventory"], "output_directory": str(args.output_directory)}, ensure_ascii=True))
    return 0 if result["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
