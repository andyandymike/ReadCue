"""Synthetic checks for versioned v1 reuse and explicit v2 target review."""
from copy import deepcopy
import json
from pathlib import Path
import random
import string
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
import prepare_model_data_v2 as builder
import check_model_v2_seed as checker


def old_row(ident, text, role, task_hash, policy_hash):
    return {"id": ident, "text": text, "target": text, "acceptable_outputs": [text],
            "track": "preservation", "family": "fixture", "group_id": ident,
            "provenance": {"origin": "assistant_original", "license": "Apache-2.0",
                "assigned_role": role, "reference_status": "assistant_policy_reviewed",
                "usage": builder.OLD_USAGE, "training_ready": True, "human_reviewed": False,
                "task_config_sha256": task_hash, "annotation_policy_sha256": policy_hash}}


class ModelDataV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project, self.old, self.seed = [self.root / name for name in ("project", "old", "seed")]
        for path in (self.project, self.old, self.seed):
            path.mkdir()
        self.config = self.project / "task.json"
        self.config.write_bytes(builder.json_bytes({"task_version": "zh-reading-model-v1", "annotation_policy": "policy.md"}))
        (self.project / "policy.md").write_text("Synthetic fixed policy.", encoding="utf-8")
        self.th = builder.sha(self.config.read_bytes())
        self.ph = builder.sha((self.project / "policy.md").read_bytes())
        self.original = {"train": [old_row("old-train", "向南走到河边。", "train", self.th, self.ph)],
                         "validation": [old_row("old-val", "清理工作已经结束！", "validation", self.th, self.ph)]}
        for role, rows in self.original.items():
            (self.old / (role + ".jsonl")).write_bytes(builder.jsonl_bytes(rows))
        (self.old / "pending.jsonl").write_bytes(builder.jsonl_bytes([{"id": "pending", "training_ready": False}]))
        (self.old / "rejected.jsonl").write_bytes(b"")
        (self.old / "ORIGINAL_DATA_LICENSE.md").write_bytes(b"Original synthetic fixture only.")
        manifest = {"schema_version": 1, "dataset_version": "model-data-v1",
                    "roles": {role: role + ".jsonl" for role in self.original},
                    "task_config_sha256": self.th, "annotation_policy_sha256": self.ph,
                    "counts": {role: len(rows) for role, rows in self.original.items()},
                    "files_sha256": {p.name: builder.sha(p.read_bytes()) for p in self.old.iterdir()}}
        (self.old / "manifest.json").write_bytes(builder.json_bytes(manifest))
        rng = random.Random(771)
        self.rows = []
        for skill in sorted(builder.SKILLS):
            for group in range(3):
                role = "validation" if group == 2 else "train"
                for pair in range(2):
                    text = "".join(rng.choices(string.ascii_lowercase, k=32))
                    preserve = pair == 1
                    target = text if preserve else text + "改"
                    self.rows.append({"id": f"{skill}-{group}-{pair}", "skill_id": skill,
                        "group_id": f"{skill}-{group}", "proposed_role": role, "text": text, "target": target,
                        "acceptable_outputs": [target], "track": "preservation" if preserve else "core_tn",
                        "operation": "preserve" if preserve else "transform", "rationale": "Synthetic test only",
                        "provenance": dict(checker.PROVENANCE), "training_ready": False})
        for name, rows in zip(builder.SEED_NAMES, (self.rows[:36], self.rows[36:])):
            (self.seed / name).write_bytes(builder.jsonl_bytes(rows))
        self.review = {"schema_version": 1, "reviewer": "synthetic assistant", "reviewer_role": "assistant",
            "human_reviewed": False, "policy": {"sha256": self.ph},
            "source_files": [{"path": name, "sha256": builder.sha((self.seed / name).read_bytes())} for name in builder.SEED_NAMES],
            "rows": [{"id": r["id"], "verdict": "accept", "reviewer_role": "assistant", "human_reviewed": False,
                      "text_sha256": builder.text_hash(r["text"]), "target_sha256": builder.text_hash(r["target"]),
                      "reason": "Explicit synthetic review"} for r in self.rows]}
        (self.seed / "assistant_review.json").write_bytes(builder.json_bytes(self.review))
        self.heldout = self.root / "old-inputs.jsonl"
        self.heldout.write_bytes(builder.jsonl_bytes([{"id": "old-test", "text": "外海灯塔暂未启用。"}]))
        evidence = [{"path": str(p), "sha256": builder.sha(p.read_bytes())} for p in
                    [self.old / n for n in ("manifest.json", "train.jsonl", "validation.jsonl")]
                    + [self.seed / n for n in builder.SEED_NAMES] + [self.heldout]]
        receipt = checker.audit_rows(self.rows, historical=[{"id": r["id"], "text": r["text"], "proposed_role": role}
                                     for role, rows in self.original.items() for r in rows],
                                    heldout=[{"id": "old-test", "text": "外海灯塔暂未启用。"}], input_evidence=evidence)
        self.assertEqual(receipt["blockers"], [])
        (self.seed / "check.reviewed.json").write_bytes(builder.json_bytes(receipt))

    def build(self):
        return builder.build_dataset(self.project, self.old, self.seed, self.config, self.heldout, self.root / "result")

    def test_build_preserves_old_targets_roles_and_bound_review_sources(self):
        (self.root / "cases.jsonl").write_text("New test labels must never be opened", encoding="utf-8")
        source_hashes = {str(p): builder.sha(p.read_bytes()) for folder in (self.old, self.seed)
                         for p in folder.iterdir() if p.is_file()}
        manifest = self.build()
        self.assertEqual(manifest["dataset_version"], "model-data-v2")
        self.assertEqual((manifest["counts"]["train"], manifest["counts"]["validation"]), (49, 25))
        self.assertEqual(manifest["counts"]["pending_v1_excluded"], 1)
        result = self.root / "result"
        for role, originals in self.original.items():
            rows = builder.parse_rows((result / (role + ".jsonl")).read_bytes())
            old = next(r for r in rows if r["id"] == originals[0]["id"])
            self.assertEqual(old["target"], originals[0]["target"])
            self.assertEqual(old["provenance"]["previous_version"], originals[0]["provenance"])
            self.assertEqual(old["provenance"]["usage"], builder.USAGE)
            self.assertTrue(all(r["provenance"]["assigned_role"] == role for r in rows))
        for name, expected in manifest["files_sha256"].items():
            self.assertEqual(builder.sha((result / name).read_bytes()), expected)
        for name, expected in source_hashes.items():
            self.assertEqual(builder.sha(Path(name).read_bytes()), expected)
        inputs = builder.parse_rows((result / "split_inputs.jsonl").read_bytes())
        self.assertTrue(all(set(r) == {"id", "text", "group_id", "role"} for r in inputs))
        self.assertEqual(builder.parse_json((result / "consumption.json").read_bytes())["new_heldout_files_read"], [])

    def test_changed_reviewed_target_is_rejected_before_output(self):
        self.review["rows"][0]["target_sha256"] = "0" * 64
        (self.seed / "assistant_review.json").write_bytes(builder.json_bytes(self.review))
        with self.assertRaisesRegex(ValueError, "changed seed target"):
            self.build()
        self.assertFalse((self.root / "result").exists())

    def test_missing_or_pending_review_decision_is_not_training_data(self):
        for change in ("missing", "pending"):
            with self.subTest(change=change):
                review = deepcopy(self.review)
                if change == "missing":
                    review["rows"].pop()
                else:
                    review["rows"][0]["verdict"] = "pending"
                (self.seed / "assistant_review.json").write_bytes(builder.json_bytes(review))
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse((self.root / "result").exists())

    def test_changed_parent_target_cannot_hide_behind_registered_provenance(self):
        path = self.old / "train.jsonl"
        rows = builder.parse_rows(path.read_bytes())
        rows[0]["target"] = "Changed old target"
        path.write_bytes(builder.jsonl_bytes(rows))
        with self.assertRaisesRegex(ValueError, "v1 source hash changed"):
            self.build()

    def test_gold_fields_in_old_test_inputs_are_rejected(self):
        self.heldout.write_bytes(builder.jsonl_bytes([{"id": "x", "text": "原文", "target": "答案"}]))
        with self.assertRaisesRegex(ValueError, "gold fields"):
            self.build()

    def test_transitive_component_cannot_bridge_existing_roles(self):
        a = old_row("a", "柜子里有18件工具。", "train", self.th, self.ph)
        b = old_row("b", "柜子里有99件工具。", "train", self.th, self.ph)
        c = old_row("c", "完全不同的另一句。", "validation", self.th, self.ph)
        b["group_id"] = c["group_id"] = "declared-pair"
        with self.assertRaisesRegex(ValueError, "Cross-role component"):
            builder.connected_groups([a, b, c])
        self.assertEqual(c["provenance"]["assigned_role"], "validation")

    def test_existing_output_is_not_overwritten(self):
        result = self.root / "result"
        result.mkdir()
        (result / "keep").write_text("original", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual((result / "keep").read_text(), "original")

    def test_joined_training_groups_cannot_count_as_two_independent_components(self):
        prepared = [old_row("a", "柜子里有18件工具。", "train", self.th, self.ph),
                    old_row("b", "柜子里有99件工具。", "train", self.th, self.ph),
                    old_row("c", "没有其他问题需要说明。", "validation", self.th, self.ph)]
        seed = [{"id": row["id"], "skill_id": "fixture", "proposed_role": row["provenance"]["assigned_role"],
                 "operation": "transform"} for row in prepared]
        mapping, _ = builder.connected_groups(prepared)
        self.assertEqual(mapping["a"], mapping["b"])
        with self.assertRaisesRegex(ValueError, "merged-component coverage: fixture/train"):
            builder.validate_component_coverage(seed, mapping)

    def test_component_coverage_still_requires_transform_in_each_role(self):
        seed = [{"id": "a", "skill_id": "fixture", "proposed_role": "train", "operation": "preserve"},
                {"id": "b", "skill_id": "fixture", "proposed_role": "train", "operation": "preserve"},
                {"id": "c", "skill_id": "fixture", "proposed_role": "validation", "operation": "transform"}]
        with self.assertRaisesRegex(ValueError, "transforms=0"):
            builder.validate_component_coverage(seed, {row["id"]: row["id"] for row in seed})

    def test_stale_checker_receipt_is_rejected(self):
        path = self.seed / "check.reviewed.json"
        receipt = builder.parse_json(path.read_bytes())
        receipt["checker_sha256"] = "f" * 64
        path.write_bytes(builder.json_bytes(receipt))
        with self.assertRaisesRegex(ValueError, "checker version"):
            self.build()


if __name__ == "__main__":
    unittest.main()
