"""Original policy boundary fixtures, isolated from all evaluation materials."""

import copy
import unittest

from readcue.context_reading_policy import preprocess, validate_resources


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
            "context": {"window_codepoints": 64, "education_terms": ["大学", "高校", "院校", "学校", "学历", "本科", "毕业"],
                        "project_suffixes": ["工程"]}}


class ContextReadingPolicyTests(unittest.TestCase):
    def apply(self, text, resources=None):
        result = preprocess(text, resources=resources or fixture())
        rebuilt, cursor = "", 0
        for edit in result["edits"]:
            self.assertGreaterEqual(edit["start"], cursor)
            self.assertEqual(edit["original"], text[edit["start"]:edit["end"]])
            rebuilt += text[cursor:edit["start"]] + edit["replacement"]
            cursor = edit["end"]
        self.assertEqual(result["text"], rebuilt + text[cursor:])
        for start, end in result["protected_spans"]:
            self.assertGreaterEqual(start, 0)
            self.assertGreater(end, start)
            self.assertLessEqual(end, len(result["text"]))
        return result

    def test_complete_slang_and_preserved_whitespace(self):
        text = "🙂u1s1， 这个办法可行。\nU1S1也是我的看法\t "
        result = self.apply(text)
        self.assertEqual(result["text"], "🙂有一说一， 这个办法可行。\n有一说一也是我的看法\t ")
        self.assertEqual(result["edits"][0]["start"], 1)
        self.assertEqual(result["warnings"], [])

    def test_slang_substrings_identifiers_and_literals_do_not_expand(self):
        for text in ("user_u1s1 u1s1_extra xu1s1 u1s1.js u1s1-test", "变量u1s1", "u1s1=3", "参数：u1s1",
                     "标识是u1s1", "字段值为u1s1", "用户名设为u1s1",
                     "u1s1的值", "`u1s1`", '“u1s1” "u1s1"', "https://example.invalid/u1s1", "@u1s1 [u1s1]"):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text)["text"], text)
        self.assertTrue(self.apply("变量u1s1")["warnings"])

    def test_school_context_and_neighboring_codes(self):
        for text, expected in (("他从985大学毕业", "他从九八五大学毕业"),
                               ("她的本科学历是211", "她的本科学历是二幺幺"),
                               ("谈到985 211。学历只是一个方面。", "谈到九八五 二幺幺。学历只是一个方面。"),
                               ("985/211高校", "九八五/二幺幺高校"),
                               ("985和211工程", "九八五和二幺幺工程")):
            with self.subTest(text=text):
                result = self.apply(text)
                self.assertEqual(result["text"], expected)
                self.assertEqual(result["warnings"], [])

    def test_bare_codes_stay_ambiguous_and_guarded(self):
        result = self.apply("985 211")
        self.assertEqual(result["text"], "985 211")
        self.assertEqual(len(result["warnings"]), 2)
        self.assertEqual(result["protected_spans"], [(0, 3), (4, 7)])

    def test_units_override_education_words(self):
        for text in ("985名大学生", "211所大学", "211路公交到学校", "985元学费",
                     "大学采购985台设备", "学校长985米宽211米", "高校预算985万元", "985年成立的学校",
                     "学校收了211份表格", "学校要211美元", "学校增长211%", "学校排名985", "大学收费：985",
                     "大学门牌985号", "高校第211次会议", "学校有211.5人", "学校尺寸985×211", "学校费用985,211"):
            with self.subTest(text=text):
                result = self.apply(text)
                self.assertEqual(result["text"], text)
                self.assertEqual(result["edits"], [])

    def test_dates_dimensions_decimal_ranges_and_identifiers(self):
        for text in ("高校数据985.211", "高校日期2026/9/211", "学校参数985-211", "高校比例985:211",
                     "学校尺寸985x211", "高校尺寸985×211", "学校数量985+211", "大学的tag-985", "学校型号A211"):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text)["text"], text)

    def test_arithmetic_overrides_school_context_for_known_code_pair(self):
        for text in ("大学数学作业：计算985/211", "大学题目中的比值：985/211", "学校练习985/211的结果",
                     "高校数据985 / 3", "高校数据985 - 211", "大学练习985 + 211"):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text)["text"], text)

    def test_literal_context_does_not_leak_to_neighboring_numbers(self):
        for text in ('“大学” 985', "[学历] 211", "`学校` 985", "https://example.invalid/大学 211"):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text)["text"], text)
        text = "学校\n985 211"
        self.assertEqual(self.apply(text)["text"], text)
        text = "学校" + "甲" * 70 + "985"
        self.assertEqual(self.apply(text)["text"], text)

    def test_existing_guard_spans_and_all_generated_readings_survive(self):
        result = self.apply("u1s1，985大学。u2s2 `123` 211元 九八五 二幺幺")
        values = [result["text"][start:end] for start, end in result["protected_spans"]]
        self.assertIn("有一说一", values)
        self.assertIn("九八五", values)
        self.assertIn("二幺幺", values)
        self.assertIn("u2s2", values)
        self.assertIn("`123`", values)
        self.assertNotIn("211", values)

    def test_output_coordinates_shift_for_ambiguous_code(self):
        result = self.apply("u1s1 985")
        self.assertEqual(result["text"], "有一说一 985")
        self.assertEqual(result["protected_spans"], [(0, 4), (5, 8)])

    def test_brand_quantity_and_existing_english_readings_unchanged(self):
        text = "360借条，旋转360度，jdk8tomcat9；J D K 八 tomcat 九。"
        result = self.apply(text)
        self.assertEqual(result["text"], text)
        self.assertEqual(result["edits"], [])
        self.assertEqual(result["warnings"], [])

    def test_resources_are_validated_and_not_mutated(self):
        resources = fixture()
        before = copy.deepcopy(resources)
        self.apply("u1s1 985高校", resources)
        self.assertEqual(resources, before)
        invalid = fixture()
        invalid["school_codes"][0]["evidence"] = ["absent"]
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_resources(invalid)
        invalid = fixture()
        invalid["context"]["window_codepoints"] = 10000
        with self.assertRaisesRegex(ValueError, "bounded"):
            validate_resources(invalid)


if __name__ == "__main__":
    unittest.main()
