import importlib.util
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import shutil

from readcue.model_scoring import evaluate, paired_changes, validate_annotations

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frozen_scoring_test", ROOT / "scripts/score_zh_pilot.py")
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)
TOOL_SPEC = importlib.util.spec_from_file_location("model_score_tool", ROOT / "tools/score_model_experiment.py")
TOOL = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(TOOL)


class ModelScoringTests(unittest.TestCase):
    def fixture(self):
        return [{"id": "fixture", "text": "请带4张纸。", "acceptable_outputs": ["请带四张纸。"],
                 "track": "core_tn", "family": "test", "targets": [
                     {"source": "4", "start": 2, "end": 3, "acceptable_outputs": ["四"]}],
                 "protected_spans": [{"start": 0, "end": 2, "text": "请带"}]}]

    def run_score(self, rows):
        return evaluate(self.fixture(), rows, SCORER.score, SCORER.surface)

    def test_local_success_does_not_hide_unrelated_deletion(self):
        result = self.run_score([{"id": "fixture", "prediction": "请带四张。", "status": "ok"}])
        self.assertEqual(result["local_targets"]["matched"], 1)
        self.assertEqual(result["overall"]["surface_correct"], 0)
        self.assertTrue(result["mechanical_details"][0]["edits_outside_targets"])

    def test_missing_and_failure_stay_in_denominators(self):
        for rows in ([], [{"id": "fixture", "prediction": "请带四张纸。", "status": "truncated"}]):
            result = self.run_score(rows)
            self.assertEqual(result["total_cases"], 1)
            self.assertEqual(result["local_targets"]["failed"], 1)
            self.assertEqual(result["overall"]["strict_correct"], 0)

    def test_wide_change_stays_unresolved(self):
        result = self.run_score([{"id": "fixture", "prediction": "四", "status": "ok"}])
        self.assertEqual(result["local_targets"]["unresolved_alignment"], 1)

    def test_wrong_offsets_rejected(self):
        cases = self.fixture()
        cases[0]["targets"][0]["start"] = 1
        with self.assertRaises(ValueError):
            validate_annotations(cases)

    def test_replay_reference_conflict_rejected(self):
        cases = self.fixture()
        cases[0]["acceptable_outputs"] = ["请带五张纸。"]
        with self.assertRaises(ValueError):
            validate_annotations(cases)

    def test_paired_changes_use_same_ids(self):
        base = self.run_score([{"id": "fixture", "prediction": "请带4张纸。", "status": "ok"}])
        adapted = self.run_score([{"id": "fixture", "prediction": "请带四张纸。", "status": "ok"}])
        self.assertEqual(paired_changes(base, adapted)["new_matches"], ["fixture"])
        self.assertEqual(paired_changes(adapted, base)["regressions"], ["fixture"])


class ScoreEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train_dir = self.root / "train"
        self.train_dir.mkdir()

        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        self.write = write
        self.proof = self.root / "proof.json"
        inputs = b'{"id":"fixture","text":"fixture"}\n'
        input_hash = hashlib.sha256(inputs).hexdigest()
        proof = {"status": "frozen", "task_config_sha256": "a"*64, "files_sha256": {"inputs.jsonl": input_hash}}
        proof_sha = write(self.proof, proof)
        (self.train_dir / "heldout-proof.json").write_bytes(self.proof.read_bytes())
        cp = self.train_dir / "checkpoints/selected"
        cp.mkdir(parents=True)
        (cp / "adapter_config.json").write_text("{}")
        (cp / "adapter_model.safetensors").write_bytes(b"synthetic test fixture; not a model")
        adapter = {"base_repo": TOOL.BASE_REPO, "base_revision": TOOL.BASE_REVISION,
                   "task_config_sha256": "a"*64, "experiment_phase": "train", "dataset_manifest_sha256": "d"*64,
                   "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cp.iterdir()}}
        adapter_sha = write(cp / "readcue_adapter.json", adapter)
        selection = {"relative_directory": "checkpoints/selected", "metadata_sha256": adapter_sha}
        write(self.train_dir / "selected_adapter.json", selection)
        shared = {"base_repo": TOOL.BASE_REPO, "base_revision": TOOL.BASE_REVISION,
                  "task_config_sha256": "a"*64, "annotation_policy_sha256": "b"*64,
                  "training_config_sha256": "c"*64, "base_model_manifest_sha256": "e"*64,
                  "source_sha256": {"runner": "f"*64}}
        self.train = dict(shared, status="complete", command="train", training=True, independent_reload_verified=True,
                          optimizer_updates=1, heldout_proof_sha256=proof_sha, selected_adapter=selection,
                          dataset_manifest_sha256="d"*64)
        write(self.train_dir / "run.json", self.train)
        paths = {}
        for name in ("base", "adapted", "rules"):
            directory = self.root/name
            directory.mkdir()
            predictions = directory / "predictions.jsonl"
            predictions.write_bytes(b'{"id":"fixture","prediction":"fixture","status":"ok"}\n')
            (directory / "inputs.original.jsonl").write_bytes(inputs)
            record = dict(shared, status="complete", command="predict", training=False, inference_fields=["text"],
                          input_sha256=input_hash, output_transform="none", predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
                          generation={"do_sample": False, "max_new_tokens": 1024})
            if name == "adapted":
                record["adapter"] = adapter
            if name == "rules":
                record.update(policy_version="audit-corrected-v1", model_executed=False, candidate_audit_only=False,
                              inputs_sha256={"inputs": input_hash})
            write(directory / "run.json", record)
            paths[name] = predictions
        self.args = argparse.Namespace(training_run=self.train_dir, heldout_proof=self.proof, **paths)

    def test_matching_training_and_prediction_evidence_is_accepted(self):
        train, _, sources = TOOL.validate_run_evidence(self.args)
        self.assertEqual(train["command"], "train")
        self.assertEqual(set(sources), {"base", "adapted", "rules"})

    def test_preflight_cannot_masquerade_as_training(self):
        self.train["command"] = "preflight"
        self.write(self.train_dir / "run.json", self.train)
        with self.assertRaisesRegex(ValueError, "completed training"):
            TOOL.validate_run_evidence(self.args)

    def test_rebuilt_proof_cannot_replace_the_training_bound_version(self):
        proof = json.loads(self.proof.read_text())
        proof["files_sha256"]["inputs.jsonl"] = "0"*64
        self.write(self.proof, proof)
        with self.assertRaisesRegex(ValueError, "exact proof"):
            TOOL.validate_run_evidence(self.args)

    def test_swapping_base_and_adapted_predictions_is_rejected(self):
        self.args.base, self.args.adapted = self.args.adapted, self.args.base
        with self.assertRaisesRegex(ValueError, "system identity"):
            TOOL.validate_run_evidence(self.args)

    def add_previous_run(self):
        previous_dir = self.root / "previous-training"
        shutil.copytree(self.train_dir, previous_dir)
        cp = previous_dir / "checkpoints/selected"
        metadata = json.loads((cp / "readcue_adapter.json").read_text())
        metadata["dataset_manifest_sha256"] = "9" * 64
        selected = json.loads((previous_dir / "selected_adapter.json").read_text())
        selected["metadata_sha256"] = self.write(cp / "readcue_adapter.json", metadata)
        self.write(previous_dir / "selected_adapter.json", selected)
        previous_train = dict(self.train, dataset_manifest_sha256="9" * 64, selected_adapter=selected)
        self.write(previous_dir / "run.json", previous_train)
        prediction_dir = self.root / "previous"
        shutil.copytree(self.root / "adapted", prediction_dir)
        record = json.loads((prediction_dir / "run.json").read_text())
        record["adapter"] = metadata
        self.write(prediction_dir / "run.json", record)
        self.args.previous = prediction_dir / "predictions.jsonl"
        self.args.previous_training_run = previous_dir
        return previous_dir, prediction_dir

    def test_original_v1_adapter_is_bound_to_its_own_selection(self):
        self.add_previous_run()
        _, _, sources = TOOL.validate_run_evidence(self.args)
        self.assertIn("previous", sources)
        self.assertIn("previous_training", sources)

    def test_previous_predictions_cannot_use_the_new_adapter(self):
        _, prediction_dir = self.add_previous_run()
        current = json.loads((self.root / "adapted/run.json").read_text())
        record = json.loads((prediction_dir / "run.json").read_text())
        record["adapter"] = current["adapter"]
        self.write(prediction_dir / "run.json", record)
        with self.assertRaisesRegex(ValueError, "original v1 selected adapter"):
            TOOL.validate_run_evidence(self.args)

    def test_previous_training_selection_cannot_be_replaced(self):
        previous_dir, _ = self.add_previous_run()
        previous_train = json.loads((previous_dir / "run.json").read_text())
        previous_train["selected_adapter"]["metadata_sha256"] = "0" * 64
        self.write(previous_dir / "run.json", previous_train)
        with self.assertRaisesRegex(ValueError, "original validation choice"):
            TOOL.validate_run_evidence(self.args)

    def test_previous_predictions_require_training_identity(self):
        self.args.previous = self.args.adapted
        with self.assertRaisesRegex(ValueError, "previous-training-run"):
            TOOL.validate_run_evidence(self.args)

    def test_v2_proof_must_bind_the_actual_training_dataset(self):
        self.train["dataset_version"] = "model-data-v2"
        self.write(self.train_dir / "run.json", self.train)
        with self.assertRaisesRegex(ValueError, "different training dataset"):
            TOOL.validate_run_evidence(self.args)


if __name__ == "__main__":
    unittest.main()
