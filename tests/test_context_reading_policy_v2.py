"""Original positive and negative segmentation cases for the isolated v2 fix."""

import hashlib
from pathlib import Path
import unittest

from readcue.context_reading_policy import preprocess as frozen_preprocess
from readcue.context_reading_policy_v2 import POLICY_VERSION, preprocess


def fixture():
    return {"schema_version": 1, "policy_version": "synthetic-context-v1",
            "provenance": [{"id": "synthetic", "url": "https://example.invalid/source"}],
            "slang": [{"id": "slang", "surface": "u1s1", "case_sensitive": False,
                       "spoken_form": "有一说一", "evidence": ["synthetic"], "reading_basis": "Fixture convention"}],
            "school_codes": [
                {"id": "school985", "surface": "985", "spoken_form": "九八五", "evidence": ["synthetic"],
                 "reading_basis": "Fixture identifier convention"},
                {"id": "school211", "surface": "211", "spoken_form": "二幺幺", "evidence": ["synthetic"],
                 "reading_basis": "User convention, not uniquely official"}],
            "context": {"window_codepoints": 64, "education_terms": ["大学", "高校", "学校", "学历", "本科", "毕业"],
                        "project_suffixes": ["工程"]}}


class ContextReadingPolicyV2Tests(unittest.TestCase):
    def apply(self, text):
        result = preprocess(text, resources=fixture())
        self.assertEqual(result["policy_version"], POLICY_VERSION)
        rebuilt, cursor = "", 0
        for edit in result["edits"]:
            self.assertGreaterEqual(edit["start"], cursor)
            self.assertEqual(edit["original"], text[edit["start"]:edit["end"]])
            rebuilt += text[cursor:edit["start"]] + edit["replacement"]
            cursor = edit["end"]
        self.assertEqual(result["text"], rebuilt + text[cursor:])
        return result

    def test_educational_compounds_do_not_become_book_quantities(self):
        cases = (("她是211本科毕业", "她是二幺幺本科毕业"),
                 ("同事的985本硕经历在这里", "同事的九八五本硕经历在这里"),
                 ("谈的是211本博连读", "谈的是二幺幺本博连读"),
                 ("985本硕", "九八五本硕"),
                 ("211\t本科背景", "二幺幺\t本科背景"),
                 ("这所985本科学制四年", "这所九八五本科学制四年"))
        for original, expected in cases:
            with self.subTest(text=original):
                result = self.apply(original)
                self.assertEqual(result["text"], expected)
                self.assertEqual(result["warnings"], [])

    def test_real_book_quantities_stay_on_the_quantity_path(self):
        for text in ("学校收到211本书", "大学购买985本教材", "985本科普读物送到了大学",
                     "大学有211本科技期刊", "211本硕士论文在学校", "学校保存985本博士论文",
                     "大学读到985本科学读物", "学校领了211本科学教材", "985本科学书放在学校"):
            with self.subTest(text=text):
                result = self.apply(text)
                self.assertEqual(result["text"], text)
                self.assertEqual(result["edits"], [])
                self.assertEqual(result["warnings"], [])

    def test_explicit_quantity_prefixes_take_precedence_over_compounds(self):
        for text in ("数量：211本科教材", "学校总计985本硕材料", "学校编号211本博方案",
                     "大学费用985本科学习用书", "学校排名985本科专业"):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text)["text"], text)

    def test_compound_text_inside_literals_remains_literal(self):
        text = '“985本硕毕业” `211本科` [985本博] https://example.invalid/211本科 @985本硕'
        result = self.apply(text)
        self.assertEqual(result["text"], text)
        self.assertEqual(result["warnings"], [])

    def test_multiple_compounds_keep_auditable_offsets_and_output_guard(self):
        original = "🙂985本科与211本博；学校领了985本教材。"
        result = self.apply(original)
        self.assertEqual(result["text"], "🙂九八五本科与二幺幺本博；学校领了985本教材。")
        self.assertEqual([(edit["start"], edit["end"]) for edit in result["edits"]], [(1, 4), (7, 10)])
        self.assertEqual([result["text"][start:end] for start, end in result["protected_spans"]], ["九八五", "二幺幺"])

    def test_other_policy_behavior_matches_frozen_v1(self):
        for text in ("u1s1，985 211。学历只是一个方面。", "985 211", "985名大学生，211所大学",
                     "大学数学作业：计算985/211", "标识是u1s1，字段值为u1s1", "985元学费，211路公交",
                     "360借条，360度，J D K 八 tomcat 九。", "985本科普书", "211本硕士论文"):
            with self.subTest(text=text):
                before = frozen_preprocess(text, resources=fixture())
                after = self.apply(text)
                before.pop("policy_version")
                after.pop("policy_version")
                self.assertEqual(after, before)

    def test_frozen_v1_source_is_unchanged(self):
        path = Path(__file__).resolve().parents[1] / "src/readcue/context_reading_policy.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "69412b2fc689e94279a63457fd57222e42f01aa1331b99843a01ef52c3dcfd16")


if __name__ == "__main__":
    unittest.main()
