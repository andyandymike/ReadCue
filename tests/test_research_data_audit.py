import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("audit_research_data", Path(__file__).resolve().parents[1] / "tools/audit_research_data.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ReferenceDispositionTests(unittest.TestCase):
    def setUp(self):
        self.dispositions = [
            {"id": "ok", "text_sha256": audit.text_hash("原文"), "eligible_as_reference_candidate": True, "disposition": "retained_exploratory_candidate", "acceptable_outputs_sha256": audit.target_hash(["原文"])},
            {"id": "withdrawn", "text_sha256": audit.text_hash("正文[吃瓜]"), "eligible_as_reference_candidate": False, "disposition": "withdrawn_for_future_use", "acceptable_outputs_sha256": audit.target_hash(["正文[吃瓜]"])},
            {"id": "pending", "text_sha256": audit.text_hash("歧义原文"), "eligible_as_reference_candidate": False, "disposition": "needs_review", "acceptable_outputs_sha256": audit.target_hash([])},
        ]

    def test_retained_reference_passes_only_reference_check(self):
        self.assertEqual(audit.check_candidates([{"id": "ok", "text": "原文", "acceptable_outputs": ["原文"]}], self.dispositions), [])

    def test_withdrawn_cannot_be_reenabled_by_submitted_eligible_flag(self):
        row = {"id": "withdrawn", "text": "正文[吃瓜]", "acceptable_outputs": ["正文[吃瓜]"], "eligible_for_provisional_em": True}
        self.assertEqual(audit.check_candidates([row], self.dispositions)[0]["reason"], "withdrawn_for_future_use")

    def test_unreviewed_correction_does_not_bypass_withdrawal(self):
        row = {"id": "withdrawn", "text": "正文[吃瓜]", "acceptable_outputs": ["正文"]}
        self.assertEqual(audit.check_candidates([row], self.dispositions)[0]["reason"], "withdrawn_for_future_use")

    def test_pending_and_unknown_rejected(self):
        rows = [{"id": "pending", "text": "歧义原文", "acceptable_outputs": []}, {"id": "new", "text": "新原文", "acceptable_outputs": ["答案"]}]
        self.assertEqual([r["reason"] for r in audit.check_candidates(rows, self.dispositions)], ["needs_review", "untracked_reference_requires_review"])

    def test_changed_reference_rejected(self):
        row = {"id": "ok", "text": "原文", "acceptable_outputs": ["新答案"]}
        self.assertEqual(audit.check_candidates([row], self.dispositions)[0]["reason"], "reference_changed_since_audit")

    def test_same_id_and_reference_for_changed_source_rejected(self):
        row = {"id": "ok", "text": "另一句话", "acceptable_outputs": ["原文"]}
        self.assertEqual(audit.check_candidates([row], self.dispositions)[0]["reason"], "source_text_changed_since_audit")

    def test_empty_set_rejected(self):
        self.assertEqual(audit.check_candidates([], self.dispositions)[0]["reason"], "empty_or_invalid_candidate_set")

    def test_duplicate_id_rejected(self):
        row = {"id": "ok", "text": "原文", "acceptable_outputs": ["原文"]}
        self.assertEqual(audit.check_candidates([row, row], self.dispositions)[0]["reason"], "duplicate_candidate_id")

    def test_malformed_rows_rejected(self):
        rows = [None, "bad", {}, {"id": "ok", "text": 123, "acceptable_outputs": ["原文"]},
                {"id": "ok", "text": "原文", "acceptable_outputs": "原文"}]
        failures = audit.check_candidates(rows, self.dispositions)
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
        self.assertEqual(audit.check_candidates(loaded, dispositions), [])


if __name__ == "__main__":
    unittest.main()
