"""Core contracts for the bounded experiment, without Torch or model execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from readcue import model_experiment as experiment

PROJECT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    eos_token_id = 999999

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        assert [row["role"] for row in messages] == ["system", "user"]
        return "system:" + messages[0]["content"] + "\nuser:" + messages[1]["content"] + "\nassistant:<think>\n\n</think>\n\n"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [ord(character) for character in text]


def write_json(path, value):
    body = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def supervised_row(ident, role, text):
    return {"id": ident, "text": text, "target": "目标文本", "acceptable_outputs": ["目标文本"],
            "track": "context_reading", "family": "fixture", "group_id": ident + "-group",
            "provenance": {"origin": "assistant_original", "license": "Apache-2.0", "assigned_role": role,
                "reference_status": "assistant_policy_reviewed", "usage": "first_bounded_model_research_training_validation",
                "training_ready": True, "task_config_sha256": "a" * 64, "annotation_policy_sha256": "b" * 64}}


class ModelExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.train = [supervised_row("train-1", "train", "训练原文12元")]
        self.validation = [supervised_row("validation-1", "validation", "验证原文34元")]
        self.manifest = {"schema_version": 1, "dataset_version": "model-data-v1",
                         "task_config_sha256": "a" * 64, "annotation_policy_sha256": "b" * 64,
                         "roles": {"train": "train.jsonl", "validation": "validation.jsonl"}, "files_sha256": {}}
        self.write_dataset()

    def write_dataset(self):
        for role, rows in (("train", self.train), ("validation", self.validation)):
            body = experiment.serialize_jsonl(rows).encode("utf-8")
            (self.root / (role + ".jsonl")).write_bytes(body)
            self.manifest["files_sha256"][role + ".jsonl"] = hashlib.sha256(body).hexdigest()
        write_json(self.root / "manifest.json", self.manifest)

    def test_target_only_loss_masks_the_entire_non_thinking_prefix(self):
        row = {"id": "x", "text": "输入", "target": "完整目标"}
        tokenizer, task = FakeTokenizer(), {"system_prompt": "固定任务"}
        encoded = experiment.encode_supervised(tokenizer, task, row)
        prompt, prefix = experiment.prompt_tokens(tokenizer, task, row["text"])
        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))
        self.assertEqual(encoded["labels"][:len(prefix)], [-100] * len(prefix))
        self.assertEqual(encoded["labels"][len(prefix):], [ord(c) for c in row["target"]] + [tokenizer.eos_token_id])
        self.assertEqual(encoded["input_ids"][:len(prefix)], prefix)
        self.assertEqual(encoded["target_tokens"], len(row["target"]) + 1)

    def test_changed_token_boundary_and_overlength_are_rejected_without_truncation(self):
        row, task = {"id": "x", "text": "输入", "target": "目标"}, {"system_prompt": "任务"}
        tokenizer = FakeTokenizer()
        with self.assertRaisesRegex(ValueError, "without truncation"):
            experiment.encode_supervised(tokenizer, task, row, max_tokens=2)
        original = tokenizer.encode
        tokenizer.encode = lambda text, **kwargs: [42] if "assistant:" in text and text.endswith("目标") else original(text, **kwargs)
        with self.assertRaisesRegex(ValueError, "boundary"):
            experiment.encode_supervised(tokenizer, task, row)

    def test_partial_accumulation_window_has_full_target_token_weight(self):
        examples = [{"target_tokens": value} for value in range(1, 11)]
        windows = experiment.accumulation_windows(examples, 8)
        self.assertEqual([len(window) for window in windows], [8, 2])
        for window in windows:
            self.assertAlmostEqual(sum(experiment.target_weight(item, window) for item in window), 1.0)
        self.assertAlmostEqual(experiment.target_weight(windows[-1][0], windows[-1]), 9 / 19)

    def test_checkpoint_selection_keeps_earlier_ties_and_rejects_nan(self):
        self.assertTrue(experiment.checkpoint_is_better(2.0, None))
        self.assertTrue(experiment.checkpoint_is_better(1.0, 2.0))
        self.assertFalse(experiment.checkpoint_is_better(1.0, 1.0))
        self.assertFalse(experiment.checkpoint_is_better(2.0, 1.0))
        with self.assertRaises(ValueError):
            experiment.checkpoint_is_better(float("nan"), 1.0)

    def test_checkpoint_cleanup_is_limited_to_this_runs_generated_children(self):
        output = self.root / "new-run"
        checkpoint = output / "checkpoints/step-0001-preflight"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"synthetic-adapter")
        protected = self.root / "other-run/checkpoints/step-0001-preflight"
        protected.mkdir(parents=True)
        marker = protected / "adapter_model.safetensors"
        marker.write_bytes(b"keep-other-run")
        for target in (protected, output / "checkpoints", self.root):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "outside this new experiment"):
                    experiment._remove_old_checkpoint(target, output)
        self.assertEqual(marker.read_bytes(), b"keep-other-run")
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaises(ValueError):
                experiment._remove_old_checkpoint(checkpoint, output)
        self.assertTrue(checkpoint.exists())
        experiment._remove_old_checkpoint(checkpoint, output)
        self.assertFalse(checkpoint.exists())
        self.assertEqual(marker.read_bytes(), b"keep-other-run")

    def test_training_loader_never_opens_other_manifest_entries(self):
        self.manifest["files_sha256"]["heldout/cases.jsonl"] = "f" * 64
        write_json(self.root / "manifest.json", self.manifest)
        opened = []
        original = Path.read_bytes

        def read(path):
            opened.append(path.name)
            if path.name not in {"manifest.json", "train.jsonl", "validation.jsonl"}:
                raise AssertionError("Read a non-training file")
            return original(path)

        with patch.object(Path, "read_bytes", read):
            rows, _, _ = experiment.load_training_data(self.root, "a" * 64, "b" * 64)
        self.assertEqual(opened, ["manifest.json", "train.jsonl", "validation.jsonl"])
        self.assertEqual(rows["train"], self.train)
        opened.clear()
        with patch.object(Path, "read_bytes", read):
            experiment.load_training_data(self.root, "a" * 64, "b" * 64, include_validation=False)
        self.assertEqual(opened, ["manifest.json", "train.jsonl"])

    def test_train_validation_overlap_and_withdrawn_targets_are_rejected(self):
        self.validation[0]["group_id"] = self.train[0]["group_id"]
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "overlap: group"):
            experiment.load_training_data(self.root, "a" * 64, "b" * 64)
        self.validation[0]["group_id"] = "other"
        self.train[0]["provenance"]["reference_status"] = "withdrawn_for_future_use"
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "Unregistered or unapproved"):
            experiment.load_training_data(self.root, "a" * 64, "b" * 64)

    def test_target_membership_and_frozen_bytes_are_required(self):
        self.train[0]["target"] = "没有登记的候选"
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "registered training target"):
            experiment.load_training_data(self.root, "a" * 64, "b" * 64)
        (self.root / "train.jsonl").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            experiment.load_training_data(self.root, "a" * 64, "b" * 64)

    def test_v2_requires_explicit_lineage_and_registered_v2_usage(self):
        self.manifest.update(dataset_version="model-data-v2", source_v1_manifest_sha256="c" * 64)
        for row in self.train + self.validation:
            row["provenance"]["usage"] = "second_bounded_model_research_training_validation"
        self.write_dataset()
        rows, manifest, _ = experiment.load_training_data(self.root, "a" * 64, "b" * 64)
        self.assertEqual(manifest["dataset_version"], "model-data-v2")
        self.assertEqual(rows["train"], self.train)
        self.train[0]["provenance"]["usage"] = "first_bounded_model_research_training_validation"
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "Unregistered or unapproved"):
            experiment.load_training_data(self.root, "a" * 64, "b" * 64)
        self.manifest.pop("source_v1_manifest_sha256")
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "v1 source manifest"):
            experiment.load_training_data(self.root, "a" * 64, "b" * 64)

    def test_v2_usage_cannot_be_laundered_through_v1_manifest(self):
        self.train[0]["provenance"]["usage"] = "second_bounded_model_research_training_validation"
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "Unregistered or unapproved"):
            experiment.load_training_data(self.root, "a" * 64, "b" * 64)

    def test_heldout_proof_checks_hashes_without_opening_any_reference(self):
        proof = {"schema_version": 1, "status": "frozen", "task_config_sha256": "a" * 64,
                 "files_sha256": {name: "c" * 64 for name in ("inputs.jsonl", "cases.jsonl", "manifest.json")}}
        path = self.root / "proof.json"
        write_json(path, proof)
        self.assertEqual(experiment.load_heldout_proof(path, "a" * 64)[0], proof)
        self.assertFalse((self.root / "cases.jsonl").exists())
        proof["status"] = "draft"
        write_json(path, proof)
        with self.assertRaisesRegex(ValueError, "frozen held-out"):
            experiment.load_heldout_proof(path, "a" * 64)

    def test_import_and_help_do_not_load_ml_dependencies(self):
        code = "import sys; import readcue.model_experiment; assert not any(x in sys.modules for x in ['torch','transformers','peft','accelerate'])"
        subprocess.run([sys.executable, "-B", "-c", code], check=True)
        result = subprocess.run([sys.executable, "-B", "-m", "readcue.model_experiment", "--help"], capture_output=True, check=True)
        self.assertIn(b"preflight", result.stdout)

    def test_invalid_start_retains_failure_record_without_loading_a_model(self):
        project = Path(__file__).resolve().parents[1]
        output = self.root / "failed-run"
        args = argparse.Namespace(project=project, model_dir=self.root / "missing-model",
            task_config=self.root / "missing-task.json", output=output, dataset=None, heldout_proof=None,
            inputs=None, adapter=None, budget_root=self.root, command="train")
        with patch.object(experiment, "_runtime", side_effect=AssertionError("No model import expected")):
            with self.assertRaises(FileNotFoundError):
                experiment.execute(args)
        state = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["optimizer_updates"], 0)
        self.assertTrue((output / "error.log").exists())

    def command(self, output, *, project=PROJECT):
        return ["predict", "--project", str(project), "--model-dir", str(self.root / "missing-model"),
                "--task-config", str(self.root / "missing-task.json"), "--inputs", str(self.root / "inputs.jsonl"),
                "--output", str(output), "--budget-root", str(self.root)]

    def test_real_script_rejects_another_checkout_even_when_source_bytes_match(self):
        other = self.root / "other-checkout"
        for relative in experiment.SOURCE_FILES:
            target = other / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / relative, target)
        output = self.root / "mixed-checkout-run"
        result = subprocess.run([sys.executable, "-B", str(PROJECT / "scripts/run_model_experiment.py"),
                                 *self.command(output, project=other)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Script entrypoint must belong to --project", result.stderr)
        state = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["phase"], "identity_checks")
        self.assertNotIn("environment_sha256", state)

    def test_external_installed_modules_must_match_project_bytes(self):
        installation = self.root / "installed"
        for relative in experiment.SOURCE_FILES:
            if relative.startswith("src/readcue/"):
                target = installation / Path(relative).relative_to("src")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT / relative, target)
        environment = {**os.environ, "PYTHONPATH": str(installation)}
        output = self.root / "matched-installed-run"
        result = subprocess.run([sys.executable, "-B", "-m", "readcue.model_experiment", *self.command(output)],
                                env=environment, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)  # It passes identity, then encounters the deliberately missing task.
        self.assertNotIn("Loaded module differs", result.stderr)
        state = json.loads((output / "run.json").read_text(encoding="utf-8"))
        identity = state["execution_identity"]
        self.assertEqual(identity["entrypoint"]["kind"], "module")
        self.assertTrue(Path(identity["entrypoint"]["path"]).is_relative_to(installation))
        module = installation / "readcue/model_experiment.py"
        self.assertEqual(identity["entrypoint"]["sha256"], experiment.hash_file(module))
        module.write_bytes(module.read_bytes() + b"\n# Different installed revision.\n")
        output = self.root / "outdated-installed-run"
        result = subprocess.run([sys.executable, "-B", "-m", "readcue.model_experiment", *self.command(output)],
                                env=environment, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Loaded module differs from --project: src/readcue/model_experiment.py", result.stderr)
        state = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "failed")
        self.assertNotIn("environment_sha256", state)

    def test_imported_helper_bytes_are_also_bound_to_the_checkout(self):
        stale = self.root / "artifacts.py"
        stale.write_bytes((PROJECT / "src/readcue/artifacts.py").read_bytes() + b"\n# older helper\n")
        with patch.object(experiment.artifacts, "__file__", str(stale)):
            with self.assertRaisesRegex(ValueError, "Loaded module differs.*artifacts.py"):
                experiment.execution_sources(PROJECT)

    def test_explicit_shared_budget_is_required_before_creating_stage(self):
        for budget in (None, self.root / "stage"):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, "budget-root"):
                    experiment._new_output(self.root / "stage", PROJECT, [PROJECT], budget)
                self.assertFalse((self.root / "stage").exists())
        with self.assertRaisesRegex(ValueError, "overlap the shared runtime cache"):
            experiment._new_output(self.root / "runtime-cache-v1/stage", PROJECT, [PROJECT], self.root)
        self.assertFalse((self.root / "runtime-cache-v1").exists())

    def test_owned_cache_is_shared_temporary_files_are_counted_and_environment_restored(self):
        first, second = self.root / "preflight", self.root / "train"
        first.mkdir()
        second.mkdir()
        historical = first / "run.json"
        historical.write_bytes(b"preserved previous stage")
        first_paths = experiment.prepare_runtime_paths(self.root, first)
        cached = Path(first_paths["TORCH_HOME"]) / "cached.bin"
        cached.write_bytes(b"keep shared cache")
        paths = experiment.prepare_runtime_paths(self.root, second)
        self.assertEqual(paths["HF_HOME"], first_paths["HF_HOME"])
        self.assertNotEqual(paths["TMPDIR"], first_paths["TMPDIR"])
        self.assertTrue(all(Path(value).is_relative_to(self.root) for value in paths.values()))
        old_tempdir = tempfile.tempdir
        with patch.dict(os.environ, {name: "outside-the-budget" for name in paths}):
            tempfile.tempdir = "outside-the-budget"
            previous = experiment.activate_runtime_paths(paths)
            try:
                self.assertEqual(tempfile.gettempdir(), paths["TMPDIR"])
                self.assertEqual({name: os.environ[name] for name in paths}, paths)
                with tempfile.NamedTemporaryFile() as temporary:
                    self.assertTrue(Path(temporary.name).is_relative_to(second / "tmp"))
                    temporary.write(b"temporary bytes are counted")
                    temporary.flush()
                    used = experiment.directory_bytes(self.root)
                    with patch.dict(experiment.CONFIG, {"max_output_bytes": used - 1}):
                        with self.assertRaisesRegex(RuntimeError, "5 GiB"):
                            experiment.check_resources(self.root, experiment.time.monotonic(), 60)
            finally:
                experiment.restore_runtime_paths(previous)
                self.assertEqual(tempfile.tempdir, "outside-the-budget")
                tempfile.tempdir = old_tempdir
            self.assertTrue(all(os.environ[name] == "outside-the-budget" for name in paths))
        self.assertEqual(cached.read_bytes(), b"keep shared cache")
        self.assertEqual(historical.read_bytes(), b"preserved previous stage")

    def test_unowned_cache_and_preloaded_libraries_are_rejected(self):
        output = self.root / "stage"
        output.mkdir()
        cache = self.root / "runtime-cache-v1"
        cache.mkdir()
        prior = cache / "prior-record"
        prior.write_bytes(b"do not adopt or overwrite")
        with self.assertRaisesRegex(ValueError, "unowned runtime cache"):
            experiment.prepare_runtime_paths(self.root, output)
        self.assertEqual(prior.read_bytes(), b"do not adopt or overwrite")
        self.assertFalse((cache / "readcue-runtime.json").exists())
        with patch.dict(sys.modules, {"torch": object()}):
            with self.assertRaisesRegex(RuntimeError, "fresh process"):
                experiment.activate_runtime_paths({"TMPDIR": str(output)})

    def test_cache_link_outside_budget_is_rejected_without_writing_to_its_target(self):
        stage = self.root / "stage"
        stage.mkdir()
        paths = experiment.prepare_runtime_paths(self.root, stage)
        link = Path(paths["TORCH_HOME"])
        link.rmdir()  # This test's newly created, empty directory only.
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary).resolve()
            protected = outside / "existing-cache"
            protected.write_bytes(b"preserve outside cache")
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("Creating directory symlinks is unavailable on this host")
            try:
                with self.assertRaisesRegex(ValueError, "links or reparse"):
                    experiment.prepare_runtime_paths(self.root, stage)
                self.assertEqual(list(outside.iterdir()), [protected])
                self.assertEqual(protected.read_bytes(), b"preserve outside cache")
            finally:
                link.unlink()

    def test_windows_junctions_are_rejected_even_when_not_posix_symlinks(self):
        info = SimpleNamespace(st_mode=experiment.stat.S_IFDIR, st_file_attributes=0x400)
        with patch.object(Path, "lstat", return_value=info), \
                patch.object(experiment.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400, create=True):
            with self.assertRaisesRegex(ValueError, "links or reparse"):
                experiment._reject_link(self.root / "windows-junction")

    def test_budget_scan_does_not_silently_skip_unreadable_directories(self):
        def denied_walk(root, *, followlinks, onerror):
            onerror(PermissionError("budget directory is unreadable"))

        with patch.object(experiment.os, "walk", side_effect=denied_walk):
            with self.assertRaisesRegex(PermissionError, "budget directory is unreadable"):
                experiment.directory_bytes(self.root)

    def test_locked_dependencies_check_every_pin_and_ignore_unlisted_packages(self):
        lock = self.root / "runtime.lock"
        lock.write_text("# fixture\ntorch==2.6.0\nPEFT==0.17.1\nPyYAML==6.0.3\n", encoding="utf-8")
        versions = {"torch": "2.6.0", "peft": "0.17.1", "pyyaml": "6.0.3"}
        with patch.object(experiment.importlib.metadata, "version", side_effect=versions.__getitem__) as lookup:
            result = experiment.verify_runtime_dependencies(lock)
        self.assertEqual(result["installed_versions"], versions)
        self.assertEqual(result["lock_sha256"], experiment.hash_file(lock))
        self.assertEqual([call.args[0] for call in lookup.call_args_list], list(versions))
        lock.write_text("package_name==1.0\npackage-name==1.0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate runtime lock"):
            experiment.verify_runtime_dependencies(lock)

    def test_missing_or_drifted_dependency_fails_before_torch_import(self):
        lock = self.root / "runtime.lock"
        lock.write_text("torch==2.6.0\npeft==0.17.1\n", encoding="utf-8")
        original_import = __import__

        def forbid_models(name, *args, **kwargs):
            if name.split(".")[0] in experiment.ML_MODULES:
                raise AssertionError("A model library was imported before dependency verification")
            return original_import(name, *args, **kwargs)

        for missing in (False, True):
            def version(name):
                if name == "torch":
                    return "2.6.0"
                if missing:
                    raise experiment.importlib.metadata.PackageNotFoundError(name)
                return "0.18.0"

            with self.subTest(missing=missing), patch("builtins.__import__", side_effect=forbid_models), \
                    patch.object(experiment.importlib.metadata, "version", side_effect=version):
                with self.assertRaisesRegex(RuntimeError, "peft: expected 0.17.1, installed (missing|0.18.0)"):
                    experiment._runtime(20260928, lock)

    def test_runtime_version_failure_retains_paths_lock_and_actual_source_identity(self):
        output = self.root / "dependency-failure"
        model = self.root / "fake-model"
        model.mkdir()
        write_json(model / "readcue_manifest.json", {"fixture": True})
        inputs = self.root / "inputs.jsonl"
        inputs.write_text('{"id":"x","text":"ordinary input"}\n', encoding="utf-8")
        args = argparse.Namespace(project=PROJECT, model_dir=model,
            task_config=PROJECT / "configs/model_task_v1.json", output=output, dataset=None, heldout_proof=None,
            inputs=inputs, adapter=None, budget_root=self.root, command="predict")
        before = tempfile.tempdir
        with patch.object(experiment, "_verify_base", return_value={"fixture": True}), \
                patch.object(experiment.importlib.metadata, "version", return_value="wrong-version"), \
                patch.object(experiment, "_load_base", side_effect=AssertionError("No model may load")):
            with self.assertRaisesRegex(RuntimeError, "Locked runtime dependency mismatch"):
                experiment.execute(args)
        state = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["phase"], "loading")
        self.assertEqual(state["runtime_lock_sha256"], experiment.hash_file(output / experiment.RUNTIME_LOCK))
        self.assertEqual(state["runtime_paths"]["TMPDIR"], str(output / "tmp"))
        self.assertEqual(tempfile.tempdir, before)
        for relative, loaded in state["execution_identity"]["loaded_modules"].items():
            self.assertEqual(state["source_sha256"][relative], loaded["sha256"])
            self.assertEqual(experiment.hash_file(output / "code" / relative), loaded["sha256"])


if __name__ == "__main__":
    unittest.main()
