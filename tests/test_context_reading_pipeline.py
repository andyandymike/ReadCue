"""Synthetic multi-stage composition checks; no evaluation corpus or model."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


pipeline = module("context_pipeline", PROJECT / "scripts/run_context_reading_pipeline.py")
prior = module("context_test_prior_fixtures", PROJECT / "tests/test_reading_pipeline.py")


class ContextReadingPipelineTests(unittest.TestCase):
    def fixture(self, directory, rows):
        args = prior.ReadingPipelineTests().fixture(directory, rows)
        resources = dict(schema_version=1, policy_version="synthetic-context-test-v1",
            provenance=[dict(id="fixture", url="https://example.invalid/source", title="Original synthetic convention")],
            slang=[dict(id="test-slang", surface="u1s1", case_sensitive=False, spoken_form="有一说一",
                        evidence=["fixture"], reading_basis="project_convention")],
            school_codes=[dict(id="test-school-a", surface="985", spoken_form="九八五", evidence=["fixture"], reading_basis="project_convention"),
                          dict(id="test-school-b", surface="211", spoken_form="二幺幺", evidence=["fixture"], reading_basis="project_convention")],
            context=dict(education_terms=["高校", "大学", "学历"], project_suffixes=["高校", "大学", "工程"], window_codepoints=64))
        path = Path(directory) / "context-resources.json"
        pipeline.write_json(path, resources)
        args.context_resources = str(path)
        return args

    def execute_fixture(self, results, **kwargs):
        return prior.fixtures.CommentPipelineTests().executor(results, **kwargs)

    def test_context_generation_is_protected_without_blocking_separate_quantities(self):
        rows = [dict(id="bad", text="回复 @同学 :u1s1，费用30元。[笑]"),
                dict(id="good", text="u1s1，费用30元。"),
                dict(id="school", text="报名985高校需要2张照片。"),
                dict(id="quantity", text="转动360度，支付20元。"),
                dict(id="unknown", text="项目xy7保持。"), dict(id="empty", text="[笑]")]
        fake = self.execute_fixture({
            "bad": ("有一说一，费用三千元。", "有一说一，费用30元->有一说一，费用三千元", "ok"),
            "good": ("有一说一，费用三十元。", "30->三十", "ok"),
            "school": ("报名九八五高校需要二三张照片。", "九八五高校需要2->九八五高校需要二三", "ok"),
            "quantity": ("转动三百六十度，支付二十元。", "360->三百六十\n20->二十", "ok")})
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, rows)
            state = pipeline.run(args, executor=fake)
            records = {row["id"]: row for row in pipeline.component.read_jsonl(Path(args.output) / "predictions.jsonl")}
            self.assertEqual(set(records), {row["id"] for row in rows})
            self.assertEqual(records["bad"]["rules_prediction"], "有一说一，费用30元。")
            self.assertEqual(records["bad"]["prediction"], "有一说一，费用30元。")
            self.assertEqual(records["bad"]["tn_prediction"], "有一说一，费用三千元。")
            self.assertEqual(records["bad"]["status"], "tn_guard_rejected")
            self.assertEqual(records["bad"]["tn_status"], "ok")
            self.assertIn("protected_span_overlap", {item["code"] for item in records["bad"]["tn_guard_reasons"]})
            self.assertTrue(records["bad"]["comment_edits"])
            self.assertTrue(records["bad"]["context_edits"])
            self.assertEqual(records["good"]["prediction"], "有一说一，费用三十元。")
            self.assertEqual(records["good"]["status"], "ok")
            self.assertEqual(records["school"]["status"], "tn_guard_rejected")
            self.assertEqual(records["school"]["prediction"], "报名九八五高校需要2张照片。")
            self.assertEqual(records["quantity"]["prediction"], "转动三百六十度，支付二十元。")
            self.assertFalse(records["quantity"]["context_edits"])
            self.assertEqual(records["unknown"]["prediction"], "项目xy7保持。")
            self.assertEqual(records["unknown"]["status"], "ok")
            self.assertEqual(records["empty"]["status"], "empty_after_policy")
            self.assertEqual(state["guard_rejected_cases"], 2)
            self.assertEqual(state["failed_cases"], 3)
            self.assertEqual(state["transformed_case_count"], 5)
            self.assertEqual(state["context_resources_sha256"], pipeline.hash_file(args.context_resources))
            self.assertEqual(state["warmup_seconds"], 0.25)

    def test_quoted_and_code_literals_survive_all_layers_and_guard_rejection(self):
        rows = [dict(id="literal", text='原样保留“u1s1”和“985”，代码`211`也别动。')]
        prediction = '原样保留“u一s1”和“985”，代码`211`也别动。'
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, rows)
            state = pipeline.run(args, executor=self.execute_fixture({"literal": (prediction, "1->一", "ok")}))
            row = pipeline.component.read_jsonl(Path(args.output) / "predictions.jsonl")[0]
            self.assertFalse(row["context_edits"])
            self.assertEqual(row["prediction"], rows[0]["text"])
            self.assertEqual(row["status"], "tn_guard_rejected")
            self.assertEqual(state["failed_cases"], 1)

    def test_tn_failures_remain_failures_and_warning_alone_is_not_a_failure(self):
        rows = [dict(id="truncated", text="金额30元。"),
                dict(id="failed", text="金额40元。"),
                dict(id="ambiguous", text="985")]
        fake = self.execute_fixture({
            "truncated": ("金额三十元。", "30->三十", "truncated"),
            "failed": ("金额四十元。", "40->四十", "inference_error")})
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, rows)
            state = pipeline.run(args, executor=fake)
            records = {row["id"]: row for row in pipeline.component.read_jsonl(Path(args.output) / "predictions.jsonl")}
            self.assertEqual(list(records), [row["id"] for row in rows])
            self.assertEqual(records["truncated"]["status"], "truncated")
            self.assertEqual(records["failed"]["status"], "inference_error")
            self.assertTrue(records["ambiguous"]["context_warnings"])
            self.assertEqual(records["ambiguous"]["status"], "ok")
            self.assertEqual(state["failed_cases"], 2)
            self.assertEqual(state["truncated_cases"], 1)

    def test_edit_replay_and_generated_span_offsets_are_checked(self):
        source = "前ab，中c9后。"
        result = dict(text="前有一说一，中九八五后。", policy_version="fixture", warnings=[], protected_spans=[],
            edits=[dict(start=1, end=3, original="ab", replacement="有一说一"),
                   dict(start=5, end=7, original="c9", replacement="九八五")])
        pipeline.checked_policy_result(source, result, "context")
        self.assertEqual(pipeline.context_output_protection(result), [[1, 5], [7, 10]])
        broken = {**result, "text": "没有编辑记录的错改"}
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            pipeline.checked_policy_result(source, broken, "context")
        with self.assertRaisesRegex(ValueError, "outside"):
            pipeline.context_output_protection({**result, "protected_spans": [[0, 1000]]})

    def test_malformed_context_result_fails_before_tn_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="one", text="普通正文。")])
            bad = dict(text="偷偷删了正文", edits=[], warnings=[], policy_version="fixture", protected_spans=[])
            with patch.object(pipeline.context_reading_policy, "preprocess", return_value=bad):
                with self.assertRaisesRegex(ValueError, "reconstruct"):
                    pipeline.run(args, executor=lambda *a, **k: self.fail("Do not infer invalid stage output"))
            self.assertEqual(pipeline.read_json(Path(args.output) / "run.json")["status"], "failed")

    def test_context_resource_mutation_invalidates_run_and_old_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="one", text="普通正文。")])
            mutation = lambda: Path(args.context_resources).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed: context_resources"):
                pipeline.run(args, executor=self.execute_fixture({}, mutation=mutation))
            path = Path(args.output) / "run.json"
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                pipeline.run(args)
            self.assertEqual(path.read_bytes(), before)

    def test_gold_input_is_rejected_before_preprocessing_or_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="one", text="普通正文。", acceptable_outputs=["参考"])])
            with self.assertRaisesRegex(ValueError, "only id and text"):
                pipeline.run(args, executor=lambda *a, **k: self.fail("No gold input allowed"))
            state = pipeline.read_json(Path(args.output) / "run.json")
            self.assertEqual(state["phase"], "preflight")
            self.assertEqual(state["status"], "failed")


if __name__ == "__main__":
    unittest.main()
