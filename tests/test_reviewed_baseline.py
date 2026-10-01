import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reviewed_runner", PROJECT / "scripts/run_zh_reviewed_baseline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReviewedBaselineTests(unittest.TestCase):
    def fixture(self, directory, rows=None):
        directory = Path(directory)
        inputs, settings = directory / "inputs.jsonl", directory / "settings.json"
        rows = rows if rows is not None else [{"id": "one", "text": "原文  A\u2028B\n第二行。"}]
        inputs.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        settings.write_text(json.dumps(dict(schema_version=1, experiment="bilibili-reviewed-v1",
                           system_prompt="只输出处理后的原文。", max_new_tokens=1024,
                           seed=20260928, temperature=0.7, top_p=0.8, top_k=20)), encoding="utf-8")
        return SimpleNamespace(inputs=str(inputs), settings=str(settings), output=str(directory / "run"),
                               baseline="identity", model_dir=None, revision=None, device="cuda")

    def test_identity_preserves_content_and_records_protocol_provenance(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = self.fixture(directory)
            state = runner.run(args)
            self.assertEqual(state["status"], "complete")
            output = Path(args.output)
            prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
            source = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
            self.assertEqual(prediction["prediction"], source["text"])
            self.assertEqual(prediction["status"], "ok")
            meta = json.loads((output / "predictions.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["settings_sha256"], runner.hash_file(args.settings))
            self.assertEqual(meta["input_sha256"], runner.hash_file(args.inputs))
            self.assertEqual(meta["predictions_sha256"], runner.hash_file(output / "predictions.jsonl"))
            self.assertEqual(meta["max_new_tokens"], 1024)
            self.assertEqual(len(meta["source_sha256"]), 4)
            self.assertEqual(meta["case_count"], 1)
            self.assertEqual(meta["failed_cases"], 0)

    def test_gold_fields_are_rejected_and_preflight_failure_is_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [{"id": "one", "text": "原文", "acceptable_outputs": ["参考"]}])
            with self.assertRaisesRegex(ValueError, "only id and text"):
                runner.run(args)
            state = runner.read_json(Path(args.output) / "run.json")
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["phase"], "preflight")
            self.assertFalse((Path(args.output) / "predictions.jsonl").exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = self.fixture(directory)
            runner.run(args)
            path = Path(args.output) / "run.json"
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.run(args)
            self.assertEqual(path.read_bytes(), before)

    def test_loading_failure_is_recorded_without_model_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            args.baseline = "wetext"
            def fail_loading(*unused, **unused_keywords):
                raise RuntimeError("synthetic loading failure")
            modules = {"tn": SimpleNamespace(), "tn.chinese": SimpleNamespace(),
                       "tn.chinese.normalizer": SimpleNamespace(Normalizer=fail_loading)}
            with patch.dict("sys.modules", modules), self.assertRaisesRegex(RuntimeError, "synthetic"):
                runner.run(args)
            state = runner.read_json(Path(args.output) / "run.json")
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["phase"], "loading")
            self.assertEqual(state["input_sha256"], runner.hash_file(args.inputs))

    def test_changed_settings_invalidate_results_and_keep_partial_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            with patch("builtins.print", side_effect=lambda *a, **kw: Path(args.settings).write_text("{}", encoding="utf-8")):
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    runner.run(args)
            self.assertTrue((Path(args.output) / "predictions.jsonl").exists())
            self.assertFalse((Path(args.output) / "predictions.meta.json").exists())
            self.assertEqual(runner.read_json(Path(args.output) / "run.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
