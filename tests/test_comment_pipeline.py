"""Synthetic policy/TN composition tests; no model or evaluation labels loaded."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("comment_pipeline", PROJECT / "scripts/run_comment_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class CommentPipelineTests(unittest.TestCase):
    def fixture(self, directory, rows):
        directory = Path(directory)
        inputs, resources, settings = (directory / name for name in ("inputs.jsonl", "resources.json", "settings.json"))
        inputs.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        pipeline.write_json(resources, dict(platform_emotes=["笑"], unicode_emotes=["😊"], metadata={"source": "original synthetic fixture"}))
        pipeline.write_json(settings, dict(schema_version=1, experiment="bilibili-reviewed-v1",
                                          system_prompt="Synthetic unused TN prompt.", max_new_tokens=1024,
                                          seed=20260928, temperature=0.7, top_p=0.8, top_k=20))
        return SimpleNamespace(inputs=str(inputs), resources=str(resources), settings=str(settings),
                               output=str(directory / "run"), model_dir=str(directory / "mock-model"),
                               revision="a" * 40, device="cuda")

    def executor(self, results, *, mutation=None, omit_last=False):
        def execute(command, **kwargs):
            self.assertEqual(command[0], pipeline.sys.executable)
            self.assertEqual(command[1], str(PROJECT / "scripts/run_zh_reviewed_baseline.py"))
            options = dict(zip(command[2::2], command[3::2]))
            self.assertEqual(options["--baseline"], "tn")
            rows = pipeline.read_jsonl(options["--inputs"])
            self.assertTrue(all(set(row) == {"id", "text"} for row in rows))
            output = Path(options["--output"])
            output.mkdir()
            predictions = []
            for row in rows:
                prediction, raw, status = results.get(row["id"], (row["text"], "", "ok"))
                predictions.append(dict(id=row["id"], prediction=prediction, raw_output=raw,
                                        status=status, latency_ms=2.5, generated_tokens=3, input_tokens=9))
            if omit_last:
                predictions = predictions[:-1]
            pipeline._write_jsonl(output / "predictions.jsonl", predictions)
            sources = {name: pipeline.hash_file(PROJECT / name) for name in
                       ("scripts/run_zh_reviewed_baseline.py", "scripts/run_zh_baseline.py",
                        "src/readcue/artifacts.py", "src/readcue/baselines.py")}
            meta = dict(experiment="bilibili-reviewed-v1", baseline="tn", training=False, external_api_cost_usd=0,
                        input_sha256=pipeline.hash_file(options["--inputs"]),
                        settings_sha256=pipeline.hash_file(options["--settings"]),
                        runner_sha256=sources["scripts/run_zh_reviewed_baseline.py"], source_sha256=sources,
                        revision=options["--revision"], model_revision=options["--revision"],
                        model_manifest_sha256="b" * 64, case_count=len(rows), load_seconds=3.5, warmup_seconds=0.25,
                        predictions_sha256=pipeline.hash_file(output / "predictions.jsonl"))
            pipeline.write_json(output / "predictions.meta.json", meta)
            pipeline.write_json(output / "run.json", {**meta, "status": "complete"})
            if mutation:
                mutation()
            return SimpleNamespace(returncode=0)
        return execute

    def test_composition_preserves_all_ids_failures_and_atomic_guard_fallback(self):
        rows = [dict(id="ordinary", text="回复 @甲 :价格30元[笑]"),
                dict(id="protected", text="代码 `v2` 价格30元"),
                dict(id="nondigit", text="保持语气"), dict(id="empty", text="[笑]"),
                dict(id="truncated", text="结果20元"), dict(id="negation", text="不超过30元")]
        results = {"ordinary": ("价格三十元", "30->三十", "ok"),
                   "protected": ("代码 `v二` 价格三十元", "2->二\n30->三十", "ok"),
                   "nondigit": ("不要乱改", "保持语气->不要乱改", "ok"),
                   "truncated": ("", "20->二", "truncated"),
                   "negation": ("三十元", "不超过30元->三十元", "ok")}
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, rows)
            state = pipeline.run(args, executor=self.executor(results))
            self.assertEqual(state["status"], "complete")
            output = Path(args.output)
            final = pipeline.read_jsonl(output / "predictions.jsonl")
            self.assertEqual([row["id"] for row in final], [row["id"] for row in rows])
            by_id = {row["id"]: row for row in final}
            self.assertEqual(by_id["ordinary"]["prediction"], "价格三十元")
            self.assertTrue(by_id["ordinary"]["policy_edits"])
            self.assertEqual(by_id["ordinary"]["tn_edits"][0]["before"], "30")
            self.assertEqual(by_id["protected"]["prediction"], "代码 `v2` 价格30元")
            self.assertEqual(by_id["protected"]["tn_prediction"], "代码 `v二` 价格三十元")
            self.assertEqual(by_id["protected"]["status"], "tn_guard_rejected")
            self.assertEqual(by_id["protected"]["tn_status"], "ok")
            self.assertEqual(by_id["empty"]["status"], "empty_after_policy")
            self.assertEqual(by_id["truncated"]["status"], "truncated")
            self.assertEqual(by_id["negation"]["status"], "tn_guard_rejected")
            self.assertEqual(state["failed_cases"], 5)
            self.assertEqual(state["guard_rejected_cases"], 3)
            self.assertEqual(state["truncated_cases"], 1)
            self.assertEqual(state["load_seconds"], 3.5)
            self.assertEqual(state["warmup_seconds"], 0.25)
            self.assertEqual(len(pipeline.read_jsonl(output / "transformed_inputs.jsonl")), 5)
            self.assertEqual(state["predictions_sha256"], pipeline.hash_file(output / "predictions.jsonl"))

    def test_han_preservation_and_sequential_anchor_boundaries(self):
        protected = pipeline.load_policy().protected_spans
        examples = [("不超过30元", "不超过30元->三十元", "三十元", True),
                    ("不超过30元", "不超过30元->不超过三十元", "不超过三十元", False),
                    ("没交20元", "没交20元->交没二十元", "交没二十元", True),
                    ("共10元再加10元", "10->十\n10->十", "共十元再加十元", False),
                    ("先10元再20元", "20->二十\n10->十", "先十元再二十元", True),
                    ("纯文字", "纯文字->别的", "别的", True),
                    ("保持", "", "保持", False)]
        for text, raw, prediction, reject in examples:
            with self.subTest(text=text, raw=raw):
                edits, reasons = pipeline.guarded_tn_edits(text, raw, prediction, protected)
                self.assertEqual(bool(reasons), reject)

    def test_code_url_quotes_and_brackets_reject_but_adjacent_digits_pass(self):
        protected = pipeline.load_policy().protected_spans
        for text in ("代码 `v2`", "地址 https://example.invalid/a2", "原样“2”", "索引[2]"):
            with self.subTest(text=text):
                edits, reasons = pipeline.guarded_tn_edits(text, "2->二", text.replace("2", "二"), protected)
                self.assertIn("protected_span_overlap", {reason["code"] for reason in reasons})
        edits, reasons = pipeline.guarded_tn_edits("代码 `v2` 收费20元", "20->二十", "代码 `v2` 收费二十元", protected)
        self.assertFalse(reasons)

    def test_empty_only_inputs_skip_model_and_stay_unsuccessful(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="empty", text="[笑]")])
            def must_not_run(*unused, **unused_keywords):
                self.fail("Empty-only policy output must not load a model")
            result = pipeline.run(args, executor=must_not_run)
            self.assertTrue(result["tn_skipped_no_nonempty_inputs"])
            self.assertEqual(result["failed_cases"], 1)
            self.assertEqual(Path(args.output, "transformed_inputs.jsonl").read_text(encoding="utf-8"), "")

    def test_gold_input_preflight_failure_is_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="one", text="文字", acceptable_outputs=["答案"])])
            with self.assertRaisesRegex(ValueError, "only id and text"):
                pipeline.run(args)
            result = pipeline.read_json(Path(args.output) / "run.json")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["phase"], "preflight")

    def test_subprocess_failure_keeps_parent_failure_and_rules_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="one", text="30元")])
            with self.assertRaisesRegex(RuntimeError, "TN component failed"):
                pipeline.run(args, executor=lambda *a, **kw: SimpleNamespace(returncode=3))
            result = pipeline.read_json(Path(args.output) / "run.json")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["phase"], "tn_component")
            self.assertEqual(result["tn_returncode"], 3)
            self.assertTrue(Path(args.output, "rules_only/predictions.jsonl").exists())

    def test_missing_child_id_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="one", text="30元")])
            with self.assertRaisesRegex(ValueError, "every transformed input"):
                pipeline.run(args, executor=self.executor({}, omit_last=True))
            self.assertEqual(pipeline.read_json(Path(args.output) / "run.json")["status"], "failed")

    def test_changed_resources_invalidate_run_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="one", text="文字")])
            mutation = lambda: Path(args.resources).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resources or settings changed"):
                pipeline.run(args, executor=self.executor({}, mutation=mutation))
            path = Path(args.output) / "run.json"
            before = path.read_bytes()
            self.assertFalse(Path(args.output, "predictions.meta.json").exists())
            with self.assertRaises(FileExistsError):
                pipeline.run(args)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
