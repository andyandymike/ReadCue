"""Original multi-stage semantic fixtures with a mocked frozen TN component."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_context_reading_pipeline as prior

pipeline = prior.module("semantic_context_pipeline", prior.PROJECT / "scripts/run_semantic_context_pipeline.py")


class SemanticContextPipelineTests(unittest.TestCase):
    def fixture(self, directory, rows):
        # Reuse setup only: do not replace globals used by the frozen tests.
        args = prior.ContextReadingPipelineTests().fixture(directory, rows)
        resources = dict(schema_version=1, policy_version="synthetic-semantic-test-v1",
            provenance={"source": "Original integration fixture, not evaluation labels"},
            entries=[dict(id="apple", surface="🍎", spoken_form="苹果",
                          evidence=[dict(url="https://example.invalid/apple", claim="Fixture noun name")])])
        path = Path(directory) / "semantic-resources.json"
        pipeline.write_json(path, resources)
        args.semantic_resources = str(path)
        # Membership in this removal list must lose to the lexical noun list.
        emotes = pipeline.read_json(args.resources)
        emotes["unicode_emotes"].append("🍎")
        pipeline.write_json(args.resources, emotes)
        return args

    def executor(self, results, **kwargs):
        return prior.ContextReadingPipelineTests().execute_fixture(results, **kwargs)

    def rows(self, args):
        return {row["id"]: row for row in pipeline.component.read_jsonl(Path(args.output) / "predictions.jsonl")}

    def test_noun_survives_removal_and_offsets_follow_reply_and_english_expansion(self):
        text = "回复 @甲 :QRS像个🍎，价格30元[笑]"
        fake = self.executor({
            "rejected": ("Q R S像个苹果，价格三千元", "苹果，价格30元->苹果，价格三千元", "ok"),
            "accepted": ("Q R S像个苹果，价格三十元", "30->三十", "ok")})
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="rejected", text=text), dict(id="accepted", text=text)])
            state = pipeline.run(args, executor=fake)
            records = self.rows(args)
            row = records["rejected"]
            self.assertEqual(row["comment_text"], "QRS像个🍎，价格30元")
            self.assertEqual(row["semantic_text"], "QRS像个苹果，价格30元")
            self.assertEqual(row["rules_prediction"], "Q R S像个苹果，价格30元")
            self.assertEqual(row["status"], "tn_guard_rejected")
            self.assertEqual(row["prediction"], row["rules_prediction"])
            self.assertEqual(row["tn_prediction"], "Q R S像个苹果，价格三千元")
            self.assertIn("protected_span_overlap", {r["code"] for r in row["tn_guard_reasons"]})
            annotation = row["comment_semantic_annotations"][0]
            self.assertEqual(annotation["original_start"], text.index("🍎"))
            self.assertEqual(annotation["start"], row["comment_text"].index("🍎"))
            self.assertTrue(annotation["resolved_by_semantic"])
            self.assertTrue(row["comment_warnings"])
            self.assertFalse(row["semantic_warnings"])
            start = row["rules_prediction"].index("苹果")
            self.assertTrue(any(a <= start and start + 2 <= b for a, b in row["context_protected_spans"]))
            self.assertEqual(records["accepted"]["status"], "ok")
            self.assertEqual(records["accepted"]["prediction"], "Q R S像个苹果，价格三十元")
            self.assertEqual(state["experiment"], "semantic-context-v4")
            self.assertEqual(state["failed_cases"], 1)
            self.assertEqual(state["semantic_resources_sha256"], pipeline.hash_file(args.semantic_resources))
            self.assertEqual(state["semantic_resources_policy_version"], "synthetic-semantic-test-v1")
            self.assertEqual(state["semantic_resource_provenance"], pipeline.read_json(args.semantic_resources)["provenance"])
            self.assertEqual(state["semantic_entry_count"], 1)
            for stage in ("comment", "semantic", "reading", "context"):
                self.assertIn(stage + "_policy_version", state)
                self.assertIn(stage + "_edits", row)
                self.assertIn(stage + "_warnings", row)
            for name in ("scripts/run_semantic_context_pipeline.py", "src/readcue/comment_policy_v2.py",
                         "src/readcue/semantic_emoji_policy.py", "src/readcue/context_reading_policy_v3.py",
                         "src/readcue/comment_policy.py"):
                self.assertEqual(state["source_sha256"][name], pipeline.hash_file(prior.PROJECT / name))

    def test_unknown_and_unresolved_pictures_cannot_be_deleted_by_numeric_anchor(self):
        rows = [dict(id="unknown", text="回复 @乙 :QRS看到🦉，数字30保留。"),
                dict(id="unresolved", text="QRS看到🍎，数字30保留。")]
        fake = self.executor({
            "unknown": ("Q R S看到，数字三十保留。", "🦉，数字30->，数字三十", "ok"),
            "unresolved": ("Q R S看到，数字三十保留。", "🍎，数字30->，数字三十", "ok")})
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, rows)
            state = pipeline.run(args, executor=fake)
            records = self.rows(args)
            for identity, emote in (("unknown", "🦉"), ("unresolved", "🍎")):
                row = records[identity]
                self.assertEqual(row["status"], "tn_guard_rejected")
                self.assertIn(emote, row["prediction"])
                self.assertNotIn(emote, row["tn_prediction"])
                self.assertFalse(row["semantic_edits"])
                self.assertFalse(row["comment_semantic_annotations"][0]["resolved_by_semantic"])
                self.assertIn("protected_span_overlap", {r["code"] for r in row["tn_guard_reasons"]})
            self.assertTrue(records["unresolved"]["semantic_warnings"])
            self.assertEqual(state["failed_cases"], 2)

    def test_school_identity_and_book_quantities_are_independent(self):
        text = "她是985在读，采购211本教材，花费30元。"
        fake = self.executor({"school": (
            "她是九八五在读，采购二百一十一本教材，花费三十元。", "211->二百一十一\n30->三十", "ok")})
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="school", text=text)])
            pipeline.run(args, executor=fake)
            row = self.rows(args)["school"]
            self.assertEqual(row["rules_prediction"], "她是九八五在读，采购211本教材，花费30元。")
            self.assertEqual(row["prediction"], "她是九八五在读，采购二百一十一本教材，花费三十元。")
            self.assertEqual(row["status"], "ok")

    def test_partial_overlap_in_later_policy_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="overlap", text="🦉甲")])
            bad = dict(text="甲", edits=[dict(start=0, end=2, original="🦉甲", replacement="甲")],
                       warnings=[], policy_version="bad-fixture")
            with patch.object(pipeline.reading_policy, "preprocess", return_value=bad):
                with self.assertRaisesRegex(ValueError, "partly overlaps"):
                    pipeline.run(args, executor=lambda *a, **k: self.fail("No inference after destructive overlap"))
            self.assertEqual(pipeline.read_json(Path(args.output) / "run.json")["status"], "failed")

    def test_semantic_resource_validation_and_mutation_fail_durably(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="bad", text="普通文本。")])
            bad = pipeline.read_json(args.semantic_resources)
            bad["entries"][0]["spoken_form"] = "Apple"
            pipeline.write_json(args.semantic_resources, bad)
            with self.assertRaisesRegex(ValueError, "Chinese noun"):
                pipeline.run(args, executor=lambda *a, **k: self.fail("Invalid resources must not reach TN"))
            self.assertEqual(pipeline.read_json(Path(args.output) / "run.json")["status"], "failed")
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="one", text="普通文本。")])
            mutation = lambda: Path(args.semantic_resources).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed: semantic_resources"):
                pipeline.run(args, executor=self.executor({}, mutation=mutation))
            path = Path(args.output) / "run.json"
            before = path.read_bytes()
            self.assertEqual(pipeline.read_json(path)["status"], "failed")
            with self.assertRaises(FileExistsError):
                pipeline.run(args)
            self.assertEqual(path.read_bytes(), before)

    def test_empty_failed_and_truncated_cases_keep_their_original_ids_and_failures(self):
        rows = [dict(id="empty", text="[笑]"), dict(id="truncated", text="价格30元。"),
                dict(id="failed", text="价格40元。"), dict(id="pending", text="看到🦉。")]
        fake = self.executor({"truncated": ("价格三十元。", "30->三十", "truncated"),
                              "failed": ("价格四十元。", "40->四十", "inference_error")})
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, rows)
            state = pipeline.run(args, executor=fake)
            records = self.rows(args)
            self.assertEqual(list(records), [r["id"] for r in rows])
            self.assertEqual([r["status"] for r in records.values()],
                             ["empty_after_policy", "truncated", "inference_error", "ok"])
            self.assertTrue(records["pending"]["comment_warnings"])
            self.assertEqual(state["failed_cases"], 3)
            self.assertEqual(state["truncated_cases"], 1)

    def test_gold_fields_are_rejected_before_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id="one", text="普通文本。", acceptable_outputs=["标签"])])
            with self.assertRaisesRegex(ValueError, "only id and text"):
                pipeline.run(args, executor=lambda *a, **k: self.fail("Gold input forbidden"))
            self.assertEqual(pipeline.read_json(Path(args.output) / "run.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
