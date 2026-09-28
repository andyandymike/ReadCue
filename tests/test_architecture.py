import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

from readcue.artifacts import BASELINE_REPOS, fetch_model, hash_file, read_json, verify_model, verify_pilot
from readcue.baselines import run_baseline, validate_inputs
from readcue.evaluation import archive_pilot, score_files, validate_run_metadata

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / "reports/zh-pilot-v1"
CASES = PROJECT / "eval/zh_reading_pilot_v1/cases.jsonl"
INPUTS = PROJECT / "eval/zh_reading_pilot_v1/inputs.jsonl"
REVISION = "a" * 40


class AssetReader(io.BytesIO):
    def read(self, size=-1):
        if size < 0 or size > 1024 * 1024:
            raise AssertionError("Asset reads must use bounded chunks")
        return super().read(size)


def upstream_fixture(corrupt=None):
    files = {"config.json": b'{"test": true}\n', "tokenizer.json": b'{"fixture": 1}\n',
             "tokenizer_config.json": b'{"fixture": 2}\n', "model.safetensors": b"test weights\0\1"}
    siblings = []
    for name, body in files.items():
        record = {"rfilename": name, "size": len(body)}
        if name == "model.safetensors":
            record["lfs"] = {"sha256": hashlib.sha256(body).hexdigest()}
        else:
            record["blobId"] = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        siblings.append(record)
    metadata = {"sha": REVISION, "siblings": siblings}

    def open_fixture(url, timeout):
        if "/api/models/" in url:
            return io.BytesIO(json.dumps(metadata).encode())
        name = urlsplit(url).path.rsplit("/", 1)[-1]
        return AssetReader(files[name] + (b"corrupted" if name == corrupt else b""))

    return files, open_fixture


class ArchitectureTests(unittest.TestCase):
    def test_all_frozen_sources_and_historical_scores_are_unchanged(self):
        self.assertEqual(len(verify_pilot(PROJECT)["files_sha256"]), 7)
        for name in ("identity", "wetext", "tn", "qwen"):
            with self.subTest(baseline=name):
                actual = score_files(PROJECT, CASES, REPORT / f"{name}.jsonl")
                self.assertEqual(actual, read_json(REPORT / f"{name}.score.json"))

    def test_archived_results_and_metadata_are_reproduced(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "archive"
            result = archive_pilot(PROJECT, REPORT, output)
            self.assertEqual(result, read_json(REPORT / "summary.json"))
            self.assertEqual(read_json(output / "archive.json")["status"], "complete")
            for name in ("identity", "wetext", "tn", "qwen"):
                self.assertEqual(hash_file(output / f"{name}.jsonl"), hash_file(REPORT / f"{name}.jsonl"))

    def test_archive_rejects_wrong_baseline_and_revision(self):
        freeze = verify_pilot(PROJECT)
        original = read_json(REPORT / "tn.meta.json")
        for changes in ({"baseline": "qwen"}, {"revision": "0" * 40}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    validate_run_metadata("tn", REPORT / "tn.jsonl", {**original, **changes}, freeze)

    def test_wrapped_run_layout_archives_and_rejects_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("identity", "wetext", "tn", "qwen"):
                directory = root / "runs" / name
                directory.mkdir(parents=True)
                shutil.copyfile(REPORT / f"{name}.jsonl", directory / "predictions.jsonl")
                shutil.copyfile(REPORT / f"{name}.meta.json", directory / "predictions.meta.json")
                state = {"status": "complete", "baseline": name,
                         "predictions_sha256": hash_file(directory / "predictions.jsonl")}
                (directory / "run.json").write_text(json.dumps(state), encoding="utf-8")
            self.assertEqual(archive_pilot(PROJECT, root / "runs", root / "archive"), read_json(REPORT / "summary.json"))
            state["status"] = "failed"
            (directory / "run.json").write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaises(ValueError):
                archive_pilot(PROJECT, root / "runs", root / "failed_archive")
            self.assertFalse((root / "failed_archive").exists())

    def test_gold_fields_and_duplicate_ids_are_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "inputs.jsonl"
            examples = [[{"id": "a", "text": "一句话", "acceptable_outputs": ["答案"]}],
                        [{"id": "a", "text": "一句话"}, {"id": "a", "text": "另一句"}]]
            for rows in examples:
                source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_inputs(source)

    def test_child_loading_failure_has_durable_record_and_log(self):
        def failed_child(command, **kwargs):
            kwargs["stderr"].write("simulated model import failure\n")
            return SimpleNamespace(returncode=1)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaises(RuntimeError):
                run_baseline(PROJECT, INPUTS, "identity", output, executor=failed_child)
            state = read_json(output / "run.json")
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["phase"], "loading_or_inference")
            self.assertEqual(state["returncode"], 1)
            self.assertIn("simulated", (output / "stderr.log").read_text())
            self.assertFalse((output / "predictions.jsonl").exists())

    def test_identity_wrapper_runs_without_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "identity"
            state = run_baseline(PROJECT, INPUTS, "identity", output)
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["failed_cases"], 0)
            score = score_files(PROJECT, CASES, output / "predictions.jsonl")
            self.assertEqual(score["overall"]["surface_correct"], 32)

    def test_wrong_model_revision_fails_before_child_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, opener = upstream_fixture()
            fetch_model("tn", root / "model", REVISION, opener=opener)
            executor = Mock()
            with self.assertRaises(ValueError):
                run_baseline(PROJECT, INPUTS, "tn", root / "run", model_dir=root / "model", revision="b" * 40, executor=executor)
            executor.assert_not_called()
            state = read_json(root / "run/run.json")
            self.assertEqual((state["status"], state["phase"]), ("failed", "preflight"))

    def test_download_checks_git_and_lfs_digests_without_whole_file_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, opener = upstream_fixture()
            with patch.object(Path, "read_bytes", side_effect=AssertionError("No whole-file reads")):
                manifest = fetch_model("tn", temporary, REVISION, opener=opener)
                verified = verify_model(temporary, BASELINE_REPOS["tn"], REVISION)
            self.assertEqual(manifest, verified)
            self.assertTrue(manifest["files"][0]["upstream_git_blob_id"])
            self.assertTrue(manifest["files"][-1]["upstream_lfs_sha256"])

    def test_corrupt_download_is_not_published_or_marked_verified(self):
        for corrupt in ("config.json", "model.safetensors"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temporary:
                _, opener = upstream_fixture(corrupt)
                with self.assertRaises(ValueError):
                    fetch_model("tn", temporary, REVISION, opener=opener)
                self.assertFalse((Path(temporary) / corrupt).exists())
                self.assertFalse((Path(temporary) / "readcue_manifest.json").exists())
                self.assertEqual(list(Path(temporary).glob("*.partial")), [])

    def test_changed_cached_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, opener = upstream_fixture()
            fetch_model("tn", root, REVISION, opener=opener)
            modified = root / "config.json"
            modified.write_text("local change", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                fetch_model("tn", root, REVISION, opener=opener)
            self.assertEqual(modified.read_text(), "local change")
            with self.assertRaises(ValueError):
                verify_model(root, BASELINE_REPOS["tn"], REVISION)

    def test_unregistered_template_adapter_or_directory_is_rejected(self):
        for name in ("chat_template.jinja", "adapter_config.json", "chat_templates"):
            with self.subTest(entry=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                _, opener = upstream_fixture()
                fetch_model("tn", directory, REVISION, opener=opener)
                extra = directory / name
                if name == "chat_templates":
                    extra.mkdir()
                else:
                    extra.write_text("unregistered loader input", encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify_model(directory, BASELINE_REPOS["tn"], REVISION)
                with self.assertRaises(ValueError):
                    fetch_model("tn", directory, REVISION, opener=opener)
                self.assertTrue(extra.exists())

    def test_registered_asset_and_manifest_symlinks_are_rejected(self):
        for name in ("config.json", "readcue_manifest.json"):
            with self.subTest(entry=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = root / "model"
                _, opener = upstream_fixture()
                fetch_model("tn", directory, REVISION, opener=opener)
                source = directory / name
                target = root / name
                shutil.copyfile(source, target)
                source.unlink()
                try:
                    source.symlink_to(target)
                except (OSError, NotImplementedError) as error:
                    self.skipTest("Host does not permit file symlinks: " + str(error))
                with self.assertRaises(ValueError):
                    verify_model(directory, BASELINE_REPOS["tn"], REVISION)
                with self.assertRaises(ValueError):
                    fetch_model("tn", directory, REVISION, opener=opener)
                self.assertTrue(source.is_symlink())

    def test_importing_cli_does_not_import_model_libraries(self):
        code = "import sys; import readcue.cli; assert not any(x in sys.modules for x in ['torch', 'transformers', 'tn'])"
        subprocess.run([sys.executable, "-B", "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
