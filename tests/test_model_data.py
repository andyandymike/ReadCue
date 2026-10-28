"""Synthetic tests for audited original-data reuse; never load real gold/model data."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
from audit_research_data import INTEGRITY_CHECKS, target_hash, text_hash
from prepare_model_data_v1 import ORIGINAL_DATASETS, build_dataset, grouped_split, sha, write_json


def jsonl_bytes(rows):
    return ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")


class ModelDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.policy = self.project / "policy.md"
        self.policy.write_text("Synthetic immutable annotation policy.", encoding="utf-8")
        self.config = self.project / "task.json"
        write_json(self.config, {"annotation_policy": "policy.md", "seed": 20260928})
        self.audit_dir = self.root / "audit"
        self.audit_dir.mkdir()
        self.datasets, dispositions, decisions = [], [], []
        examples = ["小桥旁边有3只鸟。", "最后只剩7本杂志。", "桌上放着两朵花。", "窗帘需要轻轻合上。", "松鼠绕到院子后方。"]
        for index, name in enumerate(sorted(ORIGINAL_DATASETS)):
            folder = self.root / name
            folder.mkdir()
            text = examples[index]
            row = {"id": f"fixture-{index}", "text": text, "acceptable_outputs": [text],
                   "track": "preservation", "family": "synthetic"}
            rows = [row]
            if name == "semantic_controls":
                rows.append({"id": "pending-fixture", "text": "这里的图形是什么意思？", "acceptable_outputs": [],
                             "track": "context_reading", "family": "pending"})
            cases = jsonl_bytes(rows)
            inputs = jsonl_bytes([{"id": r["id"], "text": r["text"]} for r in rows])
            (folder / "cases.jsonl").write_bytes(cases)
            (folder / "inputs.jsonl").write_bytes(inputs)
            self.datasets.append({"name": name, "path": str(folder), "origin": "assistant_original", "count": len(rows),
                                  "hashes": {"cases.jsonl": sha(cases), "inputs.jsonl": sha(inputs)}})
            for current in rows:
                pending = not current["acceptable_outputs"]
                dispositions.append({"id": current["id"], "dataset": name, "cases_file_sha256": sha(cases),
                                     "text_sha256": text_hash(current["text"]),
                                     "acceptable_outputs_sha256": target_hash(current["acceptable_outputs"]),
                                     "disposition": "needs_review" if pending else "retained_exploratory_candidate",
                                     "eligible_as_reference_candidate": not pending, "training_ready": False})
                decisions.append({"id": current["id"], "status": "pending" if pending else "assistant_policy_reviewed",
                                  "target": None if pending else current["text"], "acceptable_outputs": current["acceptable_outputs"],
                                  "track": current["track"], "family": current["family"], "group_id": current["id"],
                                  "source_text_sha256": text_hash(current["text"]),
                                  "previous_refs_sha256": target_hash(current["acceptable_outputs"]),
                                  "reason": "Explicit synthetic assistant review; no model or rule label generation."})
        withdrawn = {"id": "withdrawn-fixture", "text": "这是合成的原始句。", "acceptable_outputs": ["这是合成的旧参考。"]}
        withdrawn_path = self.root / "withdrawn.jsonl"
        withdrawn_path.write_bytes(jsonl_bytes([withdrawn]))
        dispositions.append({"id": withdrawn["id"], "dataset": "external_regression_only",
                             "cases_file_sha256": sha(withdrawn_path.read_bytes()), "text_sha256": text_hash(withdrawn["text"]),
                             "acceptable_outputs_sha256": target_hash(withdrawn["acceptable_outputs"]),
                             "disposition": "withdrawn_for_future_use", "eligible_as_reference_candidate": False,
                             "training_ready": False})
        disposition_bytes = jsonl_bytes(dispositions)
        (self.audit_dir / "reference-dispositions.jsonl").write_bytes(disposition_bytes)
        self.audit = {"schema_version": 2, "audit_version": "reference-integrity-v2", "status": "verified_with_reference_erratum",
                      "critical_integrity_checks": {key: True for key in INTEGRITY_CHECKS}, "datasets": self.datasets,
                      "reference_dispositions": {"file": "reference-dispositions.jsonl", "sha256": sha(disposition_bytes), "count": len(dispositions)}}
        write_json(self.audit_dir / "data-audit.json", self.audit)
        new_refs = ["这是合成的待审候选。"]
        write_json(self.audit_dir / "reference-errata.json", {"items": [{"id": withdrawn["id"], "cases_path": str(withdrawn_path),
                   "cases_sha256": sha(withdrawn_path.read_bytes()), "old_acceptable_outputs": withdrawn["acceptable_outputs"],
                   "old_acceptable_outputs_sha256": target_hash(withdrawn["acceptable_outputs"]),
                   "candidate_acceptable_outputs": new_refs, "candidate_acceptable_outputs_sha256": target_hash(new_refs),
                   "candidate_eligible_for_future_supervision": False}]})
        self.recipe = {"schema_version": 1, "source_datasets": sorted(ORIGINAL_DATASETS),
                       "task_config_sha256": sha(self.config.read_bytes()), "annotation_policy_sha256": sha(self.policy.read_bytes()),
                       "origin": "assistant_original", "license": "Apache-2.0", "reviewer_role": "assistant",
                       "reviewer": "synthetic-test-reviewer", "human_reviewed": False,
                       "usage": "first_bounded_model_research_training_validation", "decisions": decisions}
        self.recipe_path = self.root / "recipe.json"
        write_json(self.recipe_path, self.recipe)

    def build(self, name="dataset"):
        return build_dataset(self.project, self.audit_dir, self.config, self.recipe_path, self.root / name)

    def test_build_records_new_roles_but_preserves_old_eligibility_and_rejects_errata(self):
        originals = {str(path): sha(path.read_bytes()) for path in self.root.rglob("*.jsonl")}
        manifest = self.build()
        output = self.root / "dataset"
        self.assertEqual(manifest["counts"]["original_candidates"], 6)
        self.assertEqual(manifest["counts"]["pending"], 1)
        self.assertEqual(manifest["counts"]["audit_rejected"], 1)
        self.assertFalse(any(manifest["split"]["cross_role_overlap"].values()))
        regressions = json.loads((output / "rejection_regression.json").read_text(encoding="utf-8"))
        self.assertEqual(len(regressions["checks"]), 2)
        self.assertTrue(all(row["actual"] == "reject" and not row["used_for_training"] for row in regressions["checks"]))
        for role in ("train", "validation"):
            rows = [json.loads(line) for line in (output / (role + ".jsonl")).read_text(encoding="utf-8").splitlines()]
            for row in rows:
                self.assertEqual(row["provenance"]["assigned_role"], role)
                self.assertFalse(row["provenance"]["original_audit_training_ready"])
                self.assertTrue(row["provenance"]["training_ready"])
                self.assertFalse(row["provenance"]["independent_test_eligible"])
                self.assertNotEqual(row["id"], "withdrawn-fixture")
                self.assertIn(row["target"], row["acceptable_outputs"])
        for path, digest in originals.items():
            self.assertEqual(sha(Path(path).read_bytes()), digest)
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(sha((output / name).read_bytes()), digest)
        self.build("repeat")
        for filename in ("train.jsonl", "validation.jsonl", "split_report.json"):
            self.assertEqual((output / filename).read_bytes(), (self.root / "repeat" / filename).read_bytes())

    def test_changed_audit_and_source_hashes_fail_before_output(self):
        for kind in ("failed-check", "disposition-hash", "case-hash"):
            with self.subTest(kind=kind):
                audit = copy.deepcopy(self.audit)
                if kind == "failed-check":
                    audit["critical_integrity_checks"][next(iter(INTEGRITY_CHECKS))] = False
                elif kind == "disposition-hash":
                    audit["reference_dispositions"]["sha256"] = "0" * 64
                else:
                    audit["datasets"][0]["hashes"]["cases.jsonl"] = "0" * 64
                write_json(self.audit_dir / "data-audit.json", audit)
                with self.assertRaises(ValueError):
                    self.build(kind)
                self.assertFalse((self.root / kind).exists())

    def test_recipe_cannot_restore_pending_reference_or_invent_target(self):
        for kind in ("restore-pending", "unknown-target", "policy-change", "missing-case", "real-origin"):
            with self.subTest(kind=kind):
                recipe = copy.deepcopy(self.recipe)
                if kind == "restore-pending":
                    row = next(row for row in recipe["decisions"] if row["id"] == "pending-fixture")
                    row.update(status="assistant_policy_reviewed", target="未经资格检查的新目标", acceptable_outputs=["未经资格检查的新目标"])
                elif kind == "unknown-target":
                    recipe["decisions"][0]["target"] = "not in explicit references"
                elif kind == "policy-change":
                    recipe["annotation_policy_sha256"] = "0" * 64
                elif kind == "missing-case":
                    recipe["decisions"].pop()
                else:
                    recipe["origin"] = "real_comment"
                write_json(self.recipe_path, recipe)
                with self.assertRaises(ValueError):
                    self.build(kind)
                self.assertFalse((self.root / kind).exists())

    def test_numeric_template_and_declared_groups_never_cross_roles(self):
        texts = ["请核对订单甲的12本书。", "请核对订单甲的３５本书。", "冬天风声很大。", "窗边有一盆竹子。", "明天出发去山里。"]
        rows = [{"id": str(i), "text": text, "track": "preservation", "group_id": str(i), "provenance": {}}
                for i, text in enumerate(texts)]
        rows[3]["group_id"] = rows[2]["group_id"]
        splits, report = grouped_split(rows)
        role_for = {row["id"]: role for role, values in splits.items() for row in values}
        self.assertEqual(role_for["0"], role_for["1"])
        self.assertEqual(role_for["2"], role_for["3"])
        self.assertFalse(any(report["cross_role_overlap"].values()))
        self.assertTrue(any(row["reason"] == "numeric_template" for row in report["joins"]))

    def test_existing_output_is_never_replaced(self):
        self.build()
        path = self.root / "dataset" / "manifest.json"
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
