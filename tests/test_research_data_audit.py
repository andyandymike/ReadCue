import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

spec = importlib.util.spec_from_file_location("audit_research_data", Path(__file__).resolve().parents[1] / "tools/audit_research_data.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def bound_check(rows, dispositions):
    data = audit.dispositions_bytes(dispositions)
    report = {"schema_version": 2, "audit_version": "reference-integrity-v2",
              "status": "verified_with_reference_erratum",
              "critical_integrity_checks": {key: True for key in audit.INTEGRITY_CHECKS},
              "reference_dispositions": {"file": "reference-dispositions.jsonl", "count": len(dispositions),
                                         "sha256": hashlib.sha256(data).hexdigest()}}
    return audit.check_candidates(rows, data, report)


class ReferenceDispositionTests(unittest.TestCase):
    def setUp(self):
        self.dispositions = [
            {"id": "ok", "text_sha256": audit.text_hash("原文"), "eligible_as_reference_candidate": True, "disposition": "retained_exploratory_candidate", "acceptable_outputs_sha256": audit.target_hash(["原文"])},
            {"id": "withdrawn", "text_sha256": audit.text_hash("正文[吃瓜]"), "eligible_as_reference_candidate": False, "disposition": "withdrawn_for_future_use", "acceptable_outputs_sha256": audit.target_hash(["正文[吃瓜]"])},
            {"id": "pending", "text_sha256": audit.text_hash("歧义原文"), "eligible_as_reference_candidate": False, "disposition": "needs_review", "acceptable_outputs_sha256": audit.target_hash([])},
        ]

    def test_retained_reference_passes_only_reference_check(self):
        self.assertEqual(bound_check([{"id": "ok", "text": "原文", "acceptable_outputs": ["原文"]}], self.dispositions), [])

    def test_withdrawn_cannot_be_reenabled_by_submitted_eligible_flag(self):
        row = {"id": "withdrawn", "text": "正文[吃瓜]", "acceptable_outputs": ["正文[吃瓜]"], "eligible_for_provisional_em": True}
        self.assertEqual(bound_check([row], self.dispositions)[0]["reason"], "withdrawn_for_future_use")

    def test_unreviewed_correction_does_not_bypass_withdrawal(self):
        row = {"id": "withdrawn", "text": "正文[吃瓜]", "acceptable_outputs": ["正文"]}
        self.assertEqual(bound_check([row], self.dispositions)[0]["reason"], "withdrawn_for_future_use")

    def test_pending_and_unknown_rejected(self):
        rows = [{"id": "pending", "text": "歧义原文", "acceptable_outputs": []}, {"id": "new", "text": "新原文", "acceptable_outputs": ["答案"]}]
        self.assertEqual([r["reason"] for r in bound_check(rows, self.dispositions)], ["needs_review", "untracked_reference_requires_review"])

    def test_changed_reference_rejected(self):
        row = {"id": "ok", "text": "原文", "acceptable_outputs": ["新答案"]}
        self.assertEqual(bound_check([row], self.dispositions)[0]["reason"], "reference_changed_since_audit")

    def test_same_id_and_reference_for_changed_source_rejected(self):
        row = {"id": "ok", "text": "另一句话", "acceptable_outputs": ["原文"]}
        self.assertEqual(bound_check([row], self.dispositions)[0]["reason"], "source_text_changed_since_audit")

    def test_empty_set_rejected(self):
        self.assertEqual(bound_check([], self.dispositions)[0]["reason"], "empty_or_invalid_candidate_set")

    def test_duplicate_id_rejected(self):
        row = {"id": "ok", "text": "原文", "acceptable_outputs": ["原文"]}
        self.assertEqual(bound_check([row, row], self.dispositions)[0]["reason"], "duplicate_candidate_id")

    def test_malformed_rows_rejected(self):
        rows = [None, "bad", {}, {"id": "ok", "text": 123, "acceptable_outputs": ["原文"]},
                {"id": "ok", "text": "原文", "acceptable_outputs": "原文"}]
        failures = bound_check(rows, self.dispositions)
        self.assertEqual(len(failures), len(rows))
        self.assertTrue(all(f["reason"] == "invalid_candidate_row" for f in failures))

    def test_jsonl_preserves_non_lf_unicode_separators_and_text_binding(self):
        text = "正文\u2028继续\u0085结尾"
        row = {"id": "unicode", "text": text, "acceptable_outputs": [text]}
        dispositions = [{"id": "unicode", "text_sha256": audit.text_hash(text),
                         "acceptable_outputs_sha256": audit.target_hash([text]),
                         "eligible_as_reference_candidate": True, "disposition": "retained_exploratory_candidate"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            loaded = audit.read_rows(path)
        self.assertEqual(loaded, [row])
        self.assertEqual(bound_check(loaded, dispositions), [])

    def test_unbound_tables_and_malformed_dispositions_cannot_pass(self):
        row = {"id": "ok", "text": "原文", "acceptable_outputs": ["原文"]}
        self.assertEqual(audit.check_candidates([row], self.dispositions)[0]["reason"], "current_bound_audit_required")
        for dispositions in ([*self.dispositions, self.dispositions[0]],
                             [{**self.dispositions[0], "eligible_as_reference_candidate": "false"}],
                             [{**self.dispositions[0], "disposition": "withdrawn_for_future_use"}]):
            with self.subTest(dispositions=dispositions):
                self.assertEqual(bound_check([row], dispositions)[0]["reason"], "invalid_dispositions")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.dispositions_bytes(rows))


def write_manifest(folder, names):
    path = folder / "manifest.json"
    write_json(path, {"files": {name: {"sha256": audit.file_hash(folder / name), "bytes": (folder / name).stat().st_size}
                                for name in names}})
    return path


def synthetic_runtime(root):
    """Build the bounded historical layout from original synthetic strings only."""
    runtime, project = root / "runtime", root / "project"
    pilot = project / "eval/zh_reading_pilot_v1"
    pilot_rows = [{"id": "pilot-" + str(index), "text": "合成句子" + str(index),
                   "acceptable_outputs": ["合成读法" + str(index)]} for index in range(72)]
    write_rows(pilot / "cases.jsonl", pilot_rows)
    write_rows(pilot / "inputs.jsonl", [{"id": row["id"], "text": row["text"]} for row in pilot_rows])
    write_json(project / "protocol.json", {"synthetic_protocol": True})
    frozen = ("eval/zh_reading_pilot_v1/cases.jsonl", "eval/zh_reading_pilot_v1/inputs.jsonl", "protocol.json")
    write_json(pilot / "freeze.json", {"files_sha256": {name: audit.file_hash(project / name) for name in frozen}})

    final = runtime / "data/bilibili-pipeline-2026-09-29/prepare-02/annotations.jsonl"
    annotations = [{"id": "development", "review": {"status": "approved", "acceptable_outputs": ["合成开发句"]}}]
    write_rows(final, annotations)
    write_rows(runtime / "data/bilibili-pipeline-2026-09-29/finalize-review-2026-10-01/annotations.jsonl.before", annotations)
    groups = {
        "development": [{"id": "development", "text": "合成开发句", "acceptable_outputs": ["合成开发句"]}],
        "new_real_8": [{"id": audit.DEFECT_ID, "text": "合成待撤回句[吃瓜]", "acceptable_outputs": ["合成待撤回句[吃瓜]"]},
                       {"id": "candidate", "text": "合成候选", "acceptable_outputs": ["合成候选"]}],
    }
    datasets = {}
    for name, rows in groups.items():
        folder = runtime / "datasets" / name
        write_rows(folder / "cases.jsonl", rows)
        write_rows(folder / "inputs.jsonl", [{"id": row["id"], "text": row["text"]} for row in rows])
        datasets[name] = {"path": str(folder), "count": len(rows),
                          "hashes": {filename: audit.file_hash(folder / filename) for filename in ("cases.jsonl", "inputs.jsonl")}}
    export = Path(datasets["development"]["path"])
    write_rows(export / "submitted_annotations.jsonl", annotations)
    write_manifest(export, ("cases.jsonl", "inputs.jsonl", "submitted_annotations.jsonl"))
    run = runtime / "runs" / audit.V4
    write_json(run / "freeze.json", {"datasets": datasets})
    for relative in ("combined.inputs.jsonl", "new.inputs.jsonl", "pipeline_v4/transformed_inputs.jsonl", "v3_1_new/transformed_inputs.jsonl"):
        write_rows(run / relative, [{"id": "candidate", "text": "合成候选"}])
    write_json(run / "scoring/verification.json", {"batch_finished_at_utc": "2026-10-01T01:00:00+00:00",
               "new_reference_content_read_by_root_at_utc": {"synthetic": "2026-10-01T02:00:00+00:00"}})
    archive = runtime / "original-archive"
    write_json(archive / "original.json", {"synthetic_original": True})
    manifest = write_manifest(archive, ("original.json",))
    write_json(runtime / "runs/work-review-2026-10-01/archive-manifests-before.json",
               {str(manifest): {"sha256": audit.file_hash(manifest)}})
    wav_path = runtime / "synthetic.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 4)
    for name in audit.FEEDBACK_RUNS:
        write_json(runtime / "runs" / name / "feedback-001.json", {
            "clips": [{"wav_path": str(wav_path), "wav_sha256": audit.file_hash(wav_path), "clip_id": "synthetic"}],
            "verbatim_feedback": "Synthetic test feedback", "scope_resolution": "This fixture only"})
    return runtime, project, pilot_rows[0], groups["new_real_8"][1]


class AuditIntegrityCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime, self.project, self.pilot_candidate, self.candidate = synthetic_runtime(self.root)
        self.output = self.root / "audit"

    def cli(self, *args, optimized=False):
        return subprocess.run([sys.executable, *(["-O"] if optimized else []), str(Path(audit.__file__)), *map(str, args)],
                              capture_output=True, text=True, encoding="utf-8")

    def create_audit(self, expected=0, optimized=False):
        result = self.cli("--runtime-root", self.runtime, "--project", self.project,
                          "--output-directory", self.output, optimized=optimized)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return audit.read_json(self.output / "data-audit.json")

    def check(self, candidate=None, expected=0, report=True):
        path = self.root / "check-cases.jsonl"
        write_rows(path, [candidate or self.candidate])
        args = ["--check-cases", path, "--dispositions", self.output / "reference-dispositions.jsonl"]
        if report:
            args += ["--audit-report", self.output / "data-audit.json"]
        result = self.cli(*args)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return json.loads(result.stdout) if result.stdout else result

    def test_successful_cli_binds_dispositions_and_preserves_original_files(self):
        before = {path: audit.file_hash(path) for base in (self.project, self.runtime) for path in base.rglob("*") if path.is_file()}
        report = self.create_audit()
        self.assertEqual(report["schema_version"], 2)
        self.assertTrue(report["critical_integrity_checks"]["original_pilot_freeze_matches"])
        self.assertEqual(report["reference_dispositions"]["sha256"], audit.file_hash(self.output / "reference-dispositions.jsonl"))
        checked = self.check(self.pilot_candidate)
        self.assertEqual(checked["failures"], [])
        self.assertFalse(checked["training_authorized"])
        self.assertEqual(before, {path: audit.file_hash(path) for path in before})

    def test_changed_pilot_reference_fails_and_cannot_be_reauthorized(self):
        path = self.project / "eval/zh_reading_pilot_v1/cases.jsonl"
        rows = audit.read_rows(path)
        rows[0]["acceptable_outputs"] = ["未经审核的新读法"]
        write_rows(path, rows)
        report = self.create_audit(expected=1)
        self.assertFalse(report["critical_integrity_checks"]["original_pilot_freeze_matches"])
        self.assertEqual(report["inventory"]["retained_nonempty_candidates"], 0)
        self.assertTrue(all(not row["eligible_as_reference_candidate"] for row in audit.read_rows(self.output / "reference-dispositions.jsonl")))
        self.assertEqual(self.check(rows[0], expected=2)["failures"][0]["reason"], "audit_integrity_not_verified")

    def test_failed_nonpilot_audit_emits_diagnostics_without_eligible_candidates(self):
        path = self.runtime / "datasets/new_real_8/cases.jsonl"
        rows = audit.read_rows(path)
        rows[1]["acceptable_outputs"] = ["未经审核的新读法"]
        write_rows(path, rows)
        report = self.create_audit(expected=1)
        self.assertFalse(report["critical_integrity_checks"]["frozen_dataset_hashes_match"])
        dispositions = audit.read_rows(self.output / "reference-dispositions.jsonl")
        self.assertTrue(all(row["eligible_as_reference_candidate"] is False and "audit_integrity_failed" in row["blockers"] for row in dispositions))
        self.assertEqual(self.check(rows[1], expected=2)["failures"][0]["reason"], "audit_integrity_not_verified")

    def test_every_declared_pilot_file_is_verified(self):
        write_json(self.project / "protocol.json", {"changed": True})
        report = self.create_audit(expected=1)
        self.assertIn("protocol.json", report["original_pilot_integrity"]["failures"])

    def test_dispositions_file_tampering_is_rejected(self):
        self.create_audit()
        path = self.output / "reference-dispositions.jsonl"
        rows = audit.read_rows(path)
        next(row for row in rows if row["id"] == "candidate")["acceptable_outputs_sha256"] = audit.target_hash(["伪造读法"])
        write_rows(path, rows)
        self.assertEqual(self.check({**self.candidate, "acceptable_outputs": ["伪造读法"]}, expected=2)["failures"][0]["reason"], "dispositions_hash_mismatch")

    def test_historical_report_and_missing_report_are_rejected(self):
        report = self.create_audit()
        missing = self.check(expected=2, report=False)
        self.assertIn("--audit-report", missing.stderr)
        report["schema_version"] = 1
        report.pop("audit_version")
        report.pop("reference_dispositions")
        write_json(self.output / "data-audit.json", report)
        self.assertEqual(self.check(expected=2)["failures"][0]["reason"], "current_bound_audit_required")

    def test_success_label_cannot_override_missing_or_false_integrity_checks(self):
        report = self.create_audit()
        for change in (False, "true", None):
            with self.subTest(change=change):
                altered = copy.deepcopy(report)
                if change is None:
                    altered["critical_integrity_checks"].pop("original_pilot_freeze_matches")
                else:
                    altered["critical_integrity_checks"]["original_pilot_freeze_matches"] = change
                write_json(self.output / "data-audit.json", altered)
                self.assertEqual(self.check(expected=2)["failures"][0]["reason"], "audit_integrity_not_verified")

    def test_empty_pilot_freeze_and_optimized_count_mismatch_never_publish_candidates(self):
        freeze_path = self.project / "eval/zh_reading_pilot_v1/freeze.json"
        write_json(freeze_path, {"files_sha256": {}})
        result = self.cli("--runtime-root", self.runtime, "--project", self.project, "--output-directory", self.output)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.output / "reference-dispositions.jsonl").exists())
        # Restore the freeze, then verify -O cannot disable the count guard.
        hashes = {name: audit.file_hash(self.project / name) for name in (
            "eval/zh_reading_pilot_v1/cases.jsonl", "eval/zh_reading_pilot_v1/inputs.jsonl")}
        write_json(freeze_path, {"files_sha256": hashes})
        path = self.project / "eval/zh_reading_pilot_v1/cases.jsonl"
        write_rows(path, audit.read_rows(path)[:-1])
        result = self.cli("--runtime-root", self.runtime, "--project", self.project, "--output-directory", self.output, optimized=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("count mismatch", result.stderr)
        self.assertFalse((self.output / "reference-dispositions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
