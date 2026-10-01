"""Reuse frozen composition checks and test the bounded degree/count revision."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_context_reading_pipeline as prior

pipeline = prior.module("context_pipeline_v2", prior.PROJECT / "scripts/run_context_reading_pipeline_v2.py")


class ContextReadingPipelineV2Tests(prior.ContextReadingPipelineTests):
    def setUp(self):
        # Inherited checks resolve the original module global at call time.
        self.runner_patch = patch.object(prior, "pipeline", pipeline)
        self.runner_patch.start()
        self.addCleanup(self.runner_patch.stop)

    def test_degree_compound_and_true_book_count_keep_separate_readings(self):
        rows = [dict(id="degree", text="她在985本硕连读，同时订了211本教材。"),
                dict(id="books", text="学院采购985本科普书，另付30元。")]
        fake = self.execute_fixture({
            "degree": ("她在九八五本硕连读，同时订了二百一十一本教材。", "211->二百一十一", "ok"),
            "books": ("学院采购九百八十五本科普书，另付三十元。", "985->九百八十五\n30->三十", "ok")})
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, rows)
            state = pipeline.run(args, executor=fake)
            records = {row["id"]: row for row in pipeline.component.read_jsonl(Path(args.output) / "predictions.jsonl")}
            self.assertEqual(records["degree"]["rules_prediction"], "她在九八五本硕连读，同时订了211本教材。")
            self.assertEqual(records["degree"]["prediction"], "她在九八五本硕连读，同时订了二百一十一本教材。")
            self.assertEqual(records["books"]["rules_prediction"], rows[1]["text"])
            self.assertEqual(records["books"]["prediction"], "学院采购九百八十五本科普书，另付三十元。")
            self.assertEqual(state["failed_cases"], 0)
            self.assertEqual(state["experiment"], "context-reading-v3.1")
            self.assertIn("scripts/run_context_reading_pipeline_v2.py", state["source_sha256"])
            self.assertIn("src/readcue/context_reading_policy_v2.py", state["source_sha256"])
            self.assertNotIn("scripts/run_context_reading_pipeline.py", state["source_sha256"])
            self.assertNotIn("src/readcue/context_reading_policy.py", state["source_sha256"])


if __name__ == "__main__":
    unittest.main()
