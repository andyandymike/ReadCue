import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("check_model_v2_seed", Path(__file__).resolve().parents[1] / "tools/check_model_v2_seed.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def sample(ident, text, role="train", group=None, preserve=False):
    target = text if preserve else text + "改"
    return {"id": ident, "skill_id": "fixture", "group_id": group or ident,
            "proposed_role": role, "text": text, "target": target,
            "acceptable_outputs": [target], "track": "preservation" if preserve else "core_tn",
            "operation": "preserve" if preserve else "transform", "rationale": "Test fixture only",
            "provenance": dict(CHECK.PROVENANCE), "training_ready": False}


class ModelV2SeedTests(unittest.TestCase):
    def test_shared_group_and_numeric_substitution_cannot_cross_roles(self):
        rows = [sample("a", "费用是18元。", group="same"),
                sample("b", "费用是42元。", "validation", group="same")]
        result = CHECK.audit_rows(rows)
        kinds = {item["kind"] for item in result["blockers"]}
        self.assertIn("group_cross_role", kinds)
        self.assertIn("seed_cross_role_similarity", kinds)
        self.assertTrue(any(item.get("match_kind") == "numeric_template" for item in result["blockers"]))

    def test_passing_draft_never_becomes_training_or_test_ready(self):
        rows = [sample("a", "晨间登记物资。"), sample("b", "门口的纸条请勿擦除。", preserve=True),
                sample("c", "服务器已经停止响应！", "validation")]
        result = CHECK.audit_rows(rows)
        self.assertEqual(result["status"], "draft_checks_passed")
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["independent_test_eligible"])
        self.assertFalse(result["history_checks"]["v1_train_validation"]["performed"])
        self.assertFalse(result["history_checks"]["old_heldout_inputs_only"]["performed"])
        self.assertEqual(result["checker_sha256"], hashlib.sha256(Path(CHECK.__file__).read_bytes()).hexdigest())

    def test_validation_transform_cannot_hide_missing_training_transform(self):
        rows = [sample("a", "晨间登记物资。", preserve=True),
                sample("b", "门口的纸条请勿擦除。", preserve=True),
                sample("c", "服务器已经停止响应！", "validation")]
        result = CHECK.audit_rows(rows)
        self.assertIn({"kind": "missing_role_transform_coverage", "skill_id": "fixture", "role": "train"},
                      result["blockers"])

    def test_preservation_track_cannot_contain_transform(self):
        row = sample("a", "晨间登记物资。")
        row["track"] = "preservation"
        result = CHECK.audit_rows([row])
        self.assertTrue(any("preservation track requires preserve operation" in item.get("message", "")
                            for item in result["blockers"]))

    def test_heldout_loader_rejects_gold_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.jsonl"
            path.write_text(json.dumps({"id": "old", "text": "原文", "target": "答案"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gold fields"):
                CHECK.load_heldout_inputs(path)

    def test_history_collisions_use_only_input_and_preserve_roles(self):
        rows = [sample("a", "编号123正在排队。")]
        result = CHECK.audit_rows(rows, historical=[{"id": "old-v", "text": "编号456正在排队。", "proposed_role": "validation"}],
                                  heldout=[{"id": "old-t", "text": "编号789正在排队。"}])
        kinds = {item["kind"] for item in result["blockers"]}
        self.assertIn("v1_cross_role_similarity", kinds)
        self.assertIn("old_heldout_similarity", kinds)
        self.assertNotIn("编号", json.dumps(result, ensure_ascii=False))

    def test_v1_dataset_requires_matching_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            body = (json.dumps({"id": "old", "text": "原文"}) + "\n").encode()
            for name in ("train.jsonl", "validation.jsonl"):
                (root / name).write_bytes(body)
            manifest = {"dataset_version": "model-data-v1", "files_sha256": {
                name: hashlib.sha256(body).hexdigest() for name in ("train.jsonl", "validation.jsonl")}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            records, _ = CHECK.load_v1_dataset(root)
            self.assertEqual(len(records), 2)
            (root / "train.jsonl").write_bytes(body + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                CHECK.load_v1_dataset(root)

    def test_report_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, output = root / "samples.jsonl", root / "report.json"
            path.write_text(json.dumps(sample("a", "完整正文。")), encoding="utf-8")
            output.write_text("keep", encoding="utf-8")
            self.assertEqual(CHECK.main(["--samples", str(path), "--output", str(output)]), 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
