"""CPU-only diagnostic evidence checks using original synthetic fixtures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from readcue.model_experiment import BASE_REPO, BASE_REVISION, DATASET_USAGES

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "development_diagnostic_test", PROJECT / "tools/diagnose_model_development.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def case(ident, text, target, role, *, refs=None, preserve=False, skill=None):
    row = {"id": ident, "text": text, "target": target,
           "acceptable_outputs": refs or [target],
           "track": "preservation" if preserve else "core_tn",
           "family": "raw_family", "group_id": role + "-component",
           "provenance": {"source_version": "synthetic_fixture"}}
    if skill is not None:
        row["skill_id"] = skill
    return row


def fixture_roles():
    return {"train": [
        case("t1", "请带2张纸。", "请带二张纸。", "train", refs=["请带二张纸。", "请带两张纸。"], skill="quantity"),
        case("t2", "原样保留 A9。", "原样保留 A9。", "train", preserve=True),
        case("t3", "请带3张卡。", "请带三张卡。", "train")],
        "validation": [case("v1", "请带4张图。", "请带四张图。", "validation"),
                       case("v2", "原样保留 B8。", "原样保留 B8。", "validation", preserve=True)]}


class DiagnosticMetricTests(unittest.TestCase):
    def test_inference_projection_excludes_all_supervision_and_grouping(self):
        roles = fixture_roles()
        original = deepcopy(roles)
        inputs = TOOL.make_inputs(roles)
        self.assertEqual([r["id"] for r in inputs], ["t1", "t2", "t3", "v1", "v2"])
        self.assertTrue(all(set(r) == {"id", "text"} for r in inputs))
        self.assertEqual(roles, original)

    def test_cross_role_duplicate_id_is_rejected(self):
        roles = fixture_roles()
        roles["validation"][0]["id"] = roles["train"][0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            TOOL.make_inputs(roles)

    def test_multiple_references_do_not_masquerade_as_fixed_target_match(self):
        results, details = TOOL.evaluate(PROJECT, fixture_roles(), [
            {"id": "t1", "prediction": "请带两张纸。", "status": "ok"}])
        overall = results["train"]["overall"]
        self.assertEqual((overall["total"], overall["completed"], overall["failed_or_missing"]), (3, 1, 2))
        self.assertEqual((overall["target_strict_correct"], overall["strict_correct"], overall["surface_correct"]), (0, 1, 1))
        self.assertEqual(details[0]["mechanical_category"], "reference_match")

    def test_missing_truncated_and_empty_outputs_remain_failures(self):
        results, details = TOOL.evaluate(PROJECT, fixture_roles(), [
            {"id": "t1", "prediction": "请带二张纸。", "status": "truncated"},
            {"id": "t2", "prediction": "   ", "status": "ok"},
            {"id": "v1", "prediction": None, "status": "ok"}])
        for role, total in (("train", 3), ("validation", 2)):
            self.assertEqual(results[role]["overall"]["total"], total)
            self.assertEqual(results[role]["overall"]["failed_or_missing"], total)
            self.assertEqual(results[role]["overall"]["surface_correct"], 0)
        self.assertTrue(all(r["mechanical_category"] == "execution_failure" for r in details))

    def test_surface_changes_and_retention_proxies_stay_separate(self):
        results, details = TOOL.evaluate(PROJECT, fixture_roles(), [
            {"id": "t1", "prediction": "请带二张纸.", "status": "ok"},
            {"id": "t2", "prediction": "原样保留 A9.", "status": "ok"},
            {"id": "t3", "prediction": "请带3张卡。", "status": "ok"},
            {"id": "v2", "prediction": "原样保留 B。", "status": "ok"}])
        train = results["train"]["overall"]
        self.assertEqual((train["strict_correct"], train["surface_correct"]), (0, 2))
        self.assertEqual((train["original_acceptable_total"], train["original_acceptable_changed"]), (1, 0))
        self.assertEqual((train["conversion_required_total"], train["conversion_required_unchanged"]), (2, 1))
        self.assertEqual(results["validation"]["overall"]["original_acceptable_changed"], 1)
        self.assertEqual(details[2]["mechanical_category"], "unchanged_required_conversion")
        self.assertEqual(details[4]["mechanical_category"], "changed_acceptable_original")

    def test_group_denominators_use_registered_metadata_without_skill_inference(self):
        results, _ = TOOL.evaluate(PROJECT, fixture_roles(), [])
        self.assertEqual(results["train"]["by_skill_id"]["unregistered"]["total"], 2)
        self.assertEqual(results["train"]["by_skill_id"]["quantity"]["total"], 1)
        self.assertEqual(results["train"]["overall"]["group_count"], 1)
        for role in ("train", "validation"):
            total = results[role]["overall"]["total"]
            for field, groups in results[role].items():
                if field.startswith("by_"):
                    self.assertEqual(sum(g["total"] for g in groups.values()), total)

    def test_duplicate_or_unknown_predictions_are_rejected(self):
        prediction = {"id": "t1", "prediction": "请带二张纸。", "status": "ok"}
        for rows in ([prediction, prediction], [dict(prediction, id="unknown")]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "Duplicate or unknown"):
                TOOL.evaluate(PROJECT, fixture_roles(), rows)


class DiagnosticEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.task_path = PROJECT / "configs/model_task_v1.json"
        self.task = json.loads(self.task_path.read_text(encoding="utf-8"))
        self.task_hash = TOOL.hash_file(self.task_path)
        self.policy_hash = TOOL.hash_file(PROJECT / self.task["annotation_policy"])
        self.roles = fixture_roles()
        for role, rows in self.roles.items():
            for row in rows:
                row["provenance"].update(origin="assistant_original", license="Apache-2.0",
                    assigned_role=role, reference_status="assistant_policy_reviewed",
                    usage=DATASET_USAGES["model-data-v2"], training_ready=True,
                    task_config_sha256=self.task_hash, annotation_policy_sha256=self.policy_hash)
            (self.dataset / (role + ".jsonl")).write_bytes(TOOL.jsonl(rows))
        self.write(self.dataset / "manifest.json", {
            "schema_version": 1, "dataset_version": "model-data-v2", "source_v1_manifest_sha256": "a" * 64,
            "roles": {r: r + ".jsonl" for r in self.roles},
            "task_config_sha256": self.task_hash, "annotation_policy_sha256": self.policy_hash,
            "files_sha256": {r + ".jsonl": TOOL.hash_file(self.dataset / (r + ".jsonl")) for r in self.roles}})
        self.adapter_dir = self.root / "train/checkpoints/selected"
        self.adapter_dir.mkdir(parents=True)
        (self.adapter_dir / "adapter_config.json").write_bytes(b"{}")
        (self.adapter_dir / "adapter_model.safetensors").write_bytes(b"synthetic fixture; never loaded")
        self.adapter = {"base_repo": BASE_REPO, "base_revision": BASE_REVISION,
            "task_config_sha256": self.task_hash, "annotation_policy_sha256": self.policy_hash,
            "training_config_sha256": "c" * 64, "dataset_version": "model-data-v2", "experiment_phase": "train",
            "epoch": 3, "optimizer_updates": 6, "validation_target_token_loss": 0.25,
            "dataset_manifest_sha256": TOOL.hash_file(self.dataset / "manifest.json"),
            "files_sha256": {p.name: TOOL.hash_file(p) for p in self.adapter_dir.iterdir()}}
        self.write(self.adapter_dir / "readcue_adapter.json", self.adapter)
        self.write(self.root / "train/selected_adapter.json", {
            "relative_directory": "checkpoints/selected",
            "validation_loss": 0.25,
            "metadata_sha256": TOOL.hash_file(self.adapter_dir / "readcue_adapter.json")})
        sources, _ = TOOL.execution_sources(PROJECT, PROJECT / "scripts/run_model_experiment.py")
        self.write(self.root / "train/base-model-manifest.json", {"repo": BASE_REPO, "revision": BASE_REVISION})
        self.training = {
            "status": "complete", "command": "train", "training": True,
            "independent_reload_verified": True, "independent_reload_validation_loss": 0.25,
            "dataset_manifest_sha256": self.adapter["dataset_manifest_sha256"],
            "base_repo": BASE_REPO, "base_revision": BASE_REVISION, "task_config_sha256": self.task_hash,
            "annotation_policy_sha256": self.policy_hash, "training_config_sha256": "c" * 64,
            "data_counts": {"train": 3, "validation": 2},
            "base_model_manifest_sha256": TOOL.hash_file(self.root / "train/base-model-manifest.json"),
            "runtime_lock_sha256": TOOL.hash_file(PROJECT / "requirements-model-v1.lock"),
            "source_sha256": {name: TOOL.digest(body) for name, body in sources.items()},
            "selected_adapter": {"directory": str(self.adapter_dir.resolve()), "epoch": 3,
                "optimizer_updates": 6, "validation_loss": 0.25,
                "metadata_sha256": TOOL.hash_file(self.adapter_dir / "readcue_adapter.json")}}
        self.write(self.root / "train/run.json", self.training)
        self.prepared = self.root / "prepared"
        self.receipt = TOOL.prepare(PROJECT, self.dataset, self.root / "train", self.task_path, self.prepared)
        self.run = self.root / "predictions"
        self.run.mkdir()
        sources, _ = TOOL.execution_sources(PROJECT, PROJECT / "scripts/run_model_experiment.py")
        for name, body in sources.items():
            path = self.run / "code" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        self.write(self.run / "base-model-manifest.json", {"repo": BASE_REPO, "revision": BASE_REVISION})
        (self.run / "inputs.original.jsonl").write_bytes((self.prepared / "inputs.jsonl").read_bytes())
        self.predictions = [{"id": r["id"], "prediction": r["target"], "status": "ok"}
                            for role in ("train", "validation") for r in self.roles[role]]
        (self.run / "predictions.jsonl").write_bytes(TOOL.jsonl(self.predictions))
        self.state = {"status": "complete", "command": "predict", "training": False,
            "optimizer_updates": 0, "external_api_cost_usd": 0, "inference_fields": ["text"],
            "output_transform": "none", "input_sha256": self.receipt["files_sha256"]["inputs.jsonl"],
            "adapter": self.adapter, "task_config_sha256": self.task_hash,
            "annotation_policy_sha256": self.policy_hash, "case_count": 5,
            "predictions_sha256": TOOL.hash_file(self.run / "predictions.jsonl"),
            "failed_cases": 0, "generation": self.task["generation"],
            "base_repo": BASE_REPO, "base_revision": BASE_REVISION, "training_config_sha256": "c" * 64,
            "source_sha256": {name: TOOL.digest(body) for name, body in sources.items()},
            "runtime_lock_sha256": TOOL.hash_file(PROJECT / "requirements-model-v1.lock"),
            "base_model_manifest_sha256": TOOL.hash_file(self.run / "base-model-manifest.json")}
        self.write(self.run / "run.json", self.state)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def score(self):
        return TOOL.score(PROJECT, self.prepared, self.run, self.root / "scoring")

    def test_prepare_copies_exact_data_bytes_and_gold_free_input(self):
        for name in ("manifest.json", "train.jsonl", "validation.jsonl"):
            self.assertEqual((self.prepared / "offline" / name).read_bytes(), (self.dataset / name).read_bytes())
        inputs = TOOL.read_jsonl(self.prepared / "inputs.jsonl")
        self.assertTrue(all(set(r) == {"id", "text"} for r in inputs))
        self.assertEqual(self.receipt["counts"], {"train": 3, "validation": 2})
        for name, digest in self.receipt["files_sha256"].items():
            self.assertEqual(TOOL.hash_file(self.prepared / name), digest)

    def test_unselected_checkpoint_directory_is_not_a_frozen_training_selection(self):
        with self.assertRaisesRegex(ValueError, "selected_adapter"):
            TOOL.prepare(PROJECT, self.dataset, self.adapter_dir, self.task_path, self.root / "unselected")
        self.assertFalse((self.root / "unselected").exists())

    def test_incomplete_training_run_is_rejected_before_preparation(self):
        self.write(self.root / "train/run.json", {"status": "started", "command": "train"})
        with self.assertRaisesRegex(ValueError, "completed training"):
            TOOL.prepare(PROJECT, self.dataset, self.root / "train", self.task_path, self.root / "incomplete")
        self.assertFalse((self.root / "incomplete").exists())

    def test_selected_descriptor_cannot_replace_original_training_selection(self):
        for field, value in (("metadata_sha256", "0" * 64), ("epoch", 2), ("optimizer_updates", 5),
                             ("validation_loss", 0.26), ("directory", str(self.root / "other"))):
            with self.subTest(field=field):
                training = deepcopy(self.training)
                training["selected_adapter"][field] = value
                self.write(self.root / "train/run.json", training)
                with self.assertRaisesRegex(ValueError, "completed training selection"):
                    TOOL.prepare(PROJECT, self.dataset, self.root / "train", self.task_path, self.root / "swapped")
                self.assertFalse((self.root / "swapped").exists())

    def test_missing_independent_reload_is_rejected(self):
        self.write(self.root / "train/run.json", dict(self.training, independent_reload_verified=False))
        with self.assertRaisesRegex(ValueError, "completed training"):
            TOOL.prepare(PROJECT, self.dataset, self.root / "train", self.task_path, self.root / "unverified")

    def test_preparation_embeds_original_training_and_base_manifest_hashes(self):
        self.assertEqual(self.receipt["selected_training"]["run_sha256"], TOOL.hash_file(self.root / "train/run.json"))
        self.assertEqual((self.prepared / "training-run.json").read_bytes(), (self.root / "train/run.json").read_bytes())
        self.assertEqual(self.receipt["selected_training"]["base_model_manifest_sha256"], self.training["base_model_manifest_sha256"])

    def test_matching_complete_evidence_scores_without_loading_weights(self):
        report = self.score()
        self.assertTrue(report["diagnostic_only"])
        self.assertEqual(report["results"]["train"]["overall"]["target_strict_correct"], 3)
        self.assertEqual(report["results"]["validation"]["overall"]["target_strict_correct"], 2)
        self.assertEqual((self.root / "scoring/mismatches.jsonl").read_bytes(), b"")

    def test_executed_scoring_source_is_archived_and_hash_bound(self):
        report = self.score()
        snapshot = self.root / "scoring/diagnostic_source.py"
        self.assertEqual(snapshot.read_bytes(), Path(TOOL.__file__).read_bytes())
        self.assertEqual(TOOL.hash_file(snapshot), report["source_sha256"])
        self.assertEqual(report["files_sha256"]["diagnostic_source.py"], report["source_sha256"])

    def test_prepared_bytes_cannot_drift_before_scoring(self):
        with (self.prepared / "offline/train.jsonl").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Prepared evidence changed"):
            self.score()
        self.assertFalse((self.root / "scoring").exists())

    def test_prediction_metadata_identity_drift_is_rejected(self):
        for field, value in (("case_count", 4), ("adapter", {}), ("training", True),
                             ("input_sha256", "0" * 64), ("inference_fields", ["text", "target"])):
            with self.subTest(field=field):
                self.write(self.run / "run.json", dict(self.state, **{field: value}))
                with self.assertRaisesRegex(ValueError, "Prediction identity mismatch"):
                    self.score()
                self.assertFalse((self.root / "scoring").exists())

    def test_generation_configuration_drift_is_rejected(self):
        self.write(self.run / "run.json", dict(self.state, generation={"do_sample": True, "max_new_tokens": 128}))
        with self.assertRaisesRegex(ValueError, "generation"):
            self.score()

    def test_source_snapshot_cannot_drift(self):
        name = next(iter(self.state["source_sha256"]))
        (self.run / "code" / name).write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "source snapshot"):
            self.score()

    def test_prediction_base_manifest_must_match_original_training(self):
        self.write(self.run / "base-model-manifest.json", {"repo": BASE_REPO, "revision": BASE_REVISION, "changed_asset": True})
        self.state["base_model_manifest_sha256"] = TOOL.hash_file(self.run / "base-model-manifest.json")
        self.write(self.run / "run.json", self.state)
        with self.assertRaisesRegex(ValueError, "frozen training identity: base_model_manifest"):
            self.score()

    def test_legacy_preparation_keeps_explicit_independent_audit_requirement(self):
        self.receipt.pop("selected_training")
        self.write(self.prepared / "preparation.json", self.receipt)
        report = self.score()
        self.assertEqual(report["selected_training_binding"], "legacy_preparation_requires_independent_selection_audit")

    def test_failed_case_count_cannot_hide_failed_prediction(self):
        self.predictions[0]["status"] = "truncated"
        (self.run / "predictions.jsonl").write_bytes(TOOL.jsonl(self.predictions))
        self.state["predictions_sha256"] = TOOL.hash_file(self.run / "predictions.jsonl")
        self.write(self.run / "run.json", self.state)
        with self.assertRaisesRegex(ValueError, "failed_cases"):
            self.score()

    def test_existing_output_and_repository_output_are_rejected(self):
        output = self.root / "scoring"
        output.mkdir()
        (output / "keep").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.score()
        self.assertEqual((output / "keep").read_bytes(), b"preserve")
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            TOOL.new_external_directory(PROJECT / "should-not-create-diagnostic-test-output", PROJECT)


if __name__ == "__main__":
    unittest.main()
