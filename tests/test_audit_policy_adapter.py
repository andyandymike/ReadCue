"""Original regressions for withdrawn inferences; no evaluation files loaded."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from readcue import audit_policy_adapter as policy
from test_context_reading_policy_v3 import fixture as context_fixture
from test_reading_policy import fixture as reading_fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_rules_test_runner", ROOT / "scripts/run_audited_rules.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def resources():
    return dict(resources=dict(metadata={"fixture": True}, platform_emotes=["笑"], unicode_emotes=["😊"]),
        lexicon=reading_fixture(), context_resources=context_fixture(),
        semantic_resources=dict(schema_version=1, policy_version="synthetic", provenance="Synthetic fixture",
            entries=[dict(id=str(i), surface=surface, spoken_form=name, evidence="Synthetic fixture")
                     for i, (surface, name) in enumerate((("🌹", "玫瑰"), ("🔨", "锤子"), ("🍋", "柠檬")))]))


class AuditPolicyTests(unittest.TestCase):
    def apply(self, text):
        result = policy.preprocess(text, **resources())
        current = text
        for stage in result["stages"].values():
            self.assertEqual(policy._rebuild(current, stage["edits"])[0], stage["text"])
            current = stage["text"]
        self.assertEqual(current, result["text"])
        return result

    def test_school_window_cannot_override_quantity_connectors(self):
        for text in ("学校的学费是985。", "大学采购教材，金额为211。", "学校预算为985。"):
            with self.subTest(text=text):
                result = self.apply(text)
                self.assertEqual(result["text"], text)
                self.assertTrue(result["warnings"])
                self.assertEqual(result["protected_spans"], [])
                number = "985" if "985" in text else "211"
                spoken = "九百八十五" if number == "985" else "二百一十一"
                _, reasons = runner.guarded_tn_edits(text, number + "->" + spoken,
                    text.replace(number, spoken), lambda _: result["protected_spans"])
                self.assertEqual(reasons, [])

    def test_identifiers_nicknames_and_distant_school_words_are_not_rewritten(self):
        for text in ("学校统一使用账号211。", "学校机房型号为985。", "昵称是985本科。",
                     "学校的资料编号是985。", "我的网名是u1s1。", "学校数学题：985除以211。",
                     "985。另一句话谈到了学校。", "学历不限，记录211。"):
            with self.subTest(text=text):
                result = self.apply(text)
                self.assertEqual(result["text"], text)
                self.assertTrue(result["warnings"])
                self.assertTrue(result["protected_spans"])

    def test_explicit_local_school_forms_and_ordinary_units_still_work(self):
        for text, expected in (("985高校与211工程", "九八五高校与二幺幺工程"),
                               ("她985本硕毕业，他211在读。", "她九八五本硕毕业，他二幺幺在读。"),
                               ("u1s1，985硕博。", "有一说一，九八五硕博。"),
                               ("985名大学生，211所大学，转动360度。", "985名大学生，211所大学，转动360度。")):
            self.assertEqual(self.apply(text)["text"], expected)

    def test_withdrawn_school_list_element_stays_protected(self):
        result = self.apply("985/211高校")
        self.assertEqual(result["text"], "985/二幺幺高校")
        fragments = [result["text"][a:b] for a, b in result["protected_spans"]]
        self.assertIn("985", fragments)
        self.assertIn("二幺幺", fragments)
        _, reasons = runner.guarded_tn_edits(result["text"], "985->九百八十五",
            "九百八十五/二幺幺高校", lambda _: result["protected_spans"])
        self.assertIn("protected_span_overlap", [reason["code"] for reason in reasons])

    def test_word_tail_does_not_create_an_emoji_noun_slot(self):
        for text in ("欢迎报名🌹", "登记表已经实名🌹", "这次顺利签名🌹",
                     "事情都办得妥当🌹", "这是我的头像🌹", "做事应当实事求是🌹"):
            result = self.apply(text)
            self.assertEqual(result["text"], text)
            self.assertTrue(any("word_tail" in warning for warning in result["warnings"]))
            self.assertIn("🌹", [result["text"][a:b] for a, b in result["protected_spans"]])
        self.assertEqual(self.apply("递我一把🔨，它像🍋一般酸。")['text'], "递我一把锤子，它像柠檬一般酸。")

    def test_reading_generated_letters_are_protected_from_numeric_whole_sentence_edit(self):
        result = self.apply("使用QRS，费用30元。")
        text = result["text"]
        self.assertEqual(text, "使用Q R S，费用30元。")
        _, reasons = runner.guarded_tn_edits(text, text + "->使用，费用三十元。",
            "使用，费用三十元。", lambda _: result["protected_spans"])
        self.assertIn("protected_span_overlap", [reason["code"] for reason in reasons])
        _, reasons = runner.guarded_tn_edits(text, "30->三十", text.replace("30", "三十"),
            lambda _: result["protected_spans"])
        self.assertEqual(reasons, [])

    def test_multi_stage_offsets_and_literal_protection(self):
        result = self.apply('回复 @甲 :QRS拿一把🔨，985本科，成本30元[笑]；“u1s1”与`211`。')
        self.assertEqual(result["text"], 'Q R S拿一把锤子，九八五本科，成本30元；“u1s1”与`211`。')
        fragments = [result["text"][a:b] for a, b in result["protected_spans"]]
        for fragment in ("Q R S", "锤子", "九八五", "“u1s1”", "`211`"):
            self.assertIn(fragment, fragments)

    def test_frozen_modules_remain_unchanged(self):
        expected = {"context_reading_policy_v3.py": "d0705cc760ea96a9692617d1b9dd843158619cd2087c5e2dc9a21916c07f4bd5",
                    "reading_policy.py": "11bf1b87aa30f3150f75fdd94599ce597f6f107603079e3e743457b9b86b2bba",
                    "comment_policy.py": "9d8b680eadaea0778e7d62c7918483a3ae459c102e34c1ce5bdc385b05e46f3a"}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / "src/readcue" / name).read_bytes()).hexdigest(), digest)

    def setup_cli(self, directory):
        root = Path(directory)
        values = {key: root / (key + ".json") for key in resources()}
        for key, value in resources().items():
            values[key].write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        inputs = root / "inputs.jsonl"
        inputs.write_text(json.dumps(dict(id="one", text="使用QRS，费用30元。"), ensure_ascii=False) + "\n", encoding="utf-8")
        return argparse.Namespace(inputs=str(inputs), output=str(root / "out"), tn_candidates=None,
                                  **{key: str(path) for key, path in values.items()})

    def test_cli_rules_only_is_a_distinct_run_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_cli(directory)
            state = runner.run(args)
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["policy_version"], "audit-corrected-v1")
            self.assertFalse(state["model_executed"])
            self.assertFalse(state["historical_results_replaced"])
            row = json.loads((Path(args.output) / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "rules_only")
            with self.assertRaises(FileExistsError):
                runner.run(args)

    def test_cli_checks_existing_candidate_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_cli(directory)
            path = Path(directory) / "candidate.jsonl"
            source = "使用Q R S，费用30元。"
            path.write_text(json.dumps(dict(id="one", status="ok", raw_output=source + "->使用，费用三十元。",
                                           prediction="使用，费用三十元。"), ensure_ascii=False) + "\n", encoding="utf-8")
            args.tn_candidates = str(path)
            state = runner.run(args)
            row = json.loads((Path(args.output) / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "candidate_rejected")
            self.assertEqual(row["prediction"], source)
            self.assertFalse(state["model_executed"])

    def test_cli_rejects_gold_inputs_and_wrong_candidate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_cli(directory)
            Path(args.inputs).write_text(json.dumps(dict(id="one", text="x", accepted_outputs=["x"])), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.run(args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_cli(directory)
            path = Path(directory) / "candidate.jsonl"
            path.write_text(json.dumps(dict(id="wrong", prediction="x", raw_output="", status="ok")), encoding="utf-8")
            args.tn_candidates = str(path)
            with self.assertRaises(ValueError):
                runner.run(args)


if __name__ == "__main__":
    unittest.main()
