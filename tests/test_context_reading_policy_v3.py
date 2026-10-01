"""Original boundary tests for v3, without reading any evaluation material."""

import hashlib
from pathlib import Path
import unittest

from readcue.context_reading_policy_v2 import preprocess as frozen_preprocess
from readcue.context_reading_policy_v3 import POLICY_VERSION, preprocess


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


class ContextReadingPolicyV3Tests(unittest.TestCase):
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

    def test_adjacent_education_identities_need_no_other_school_word(self):
        for text, expected in (("她走的是211硕博连读路线", "她走的是二幺幺硕博连读路线"),
                               ("他目前985在读", "他目前九八五在读"),
                               ("同事是985硕博", "同事是九八五硕博"),
                               ("这位211在读研究生分享了经验", "这位二幺幺在读研究生分享了经验"),
                               ("211\t硕博都参加交流", "二幺幺\t硕博都参加交流"),
                               ("985 在读，先记录下来", "九八五 在读，先记录下来")):
            with self.subTest(text=text):
                result = self.apply(text)
                self.assertEqual(result["text"], expected)
                self.assertEqual(result["warnings"], [])

    def test_quantity_and_year_rules_stay_ahead_of_identity_context(self):
        for text in ("有985名硕博参加", "211个在读学生", "学校人数985在读", "费用：211在读期间支付",
                     "985年在读者的信息", "学校第211在读登记表", "大学预算985元供在读学生使用",
                     "人数211硕博", "大学总计985硕博", "编号211在读记录"):
            with self.subTest(text=text):
                result = self.apply(text)
                self.assertEqual(result["text"], text)
                self.assertEqual(result["edits"], [])

    def test_arithmetic_is_not_a_school_identity(self):
        for text in ("大学计算985/211在读记录的比值", "求值985在读数据中出现", "算式：211硕博资料中出现",
                     "大学数据985 - 211在读者使用", "学校统计985.211在读数据"):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text)["text"], text)

    def test_reading_technical_prefixes_do_not_match_enrolment(self):
        for suffix in ("在读取数据", "在读写文件", "在读数器中", "在读卡器上", "在读盘时", "在读码程序中",
                       "在读秒时", "在读音表中", "在读报时", "在读书时"):
            text = "学校的985" + suffix
            with self.subTest(text=text):
                result = self.apply(text)
                self.assertEqual(result["text"], text)
                self.assertTrue(result["warnings"])

    def test_educational_materials_do_not_become_person_identity(self):
        for suffix in ("硕博论文", "硕博教材", "硕博读物", "硕博书籍", "硕博期刊", "硕博资料"):
            text = "高校的211" + suffix
            with self.subTest(text=text):
                result = self.apply(text)
                self.assertEqual(result["text"], text)
                self.assertTrue(result["warnings"])

    def test_explicit_nicknames_and_identifiers_remain_literal(self):
        for text in ("昵称985在读", "网名是211硕博", "用户名：985在读", "账号为211硕博", "型号985在读",
                     "标识是211硕博", "字段值为985在读", "@985在读 @211硕博", "tag-985在读"):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text)["text"], text)

    def test_quoted_code_urls_and_brackets_remain_protected(self):
        text = '“985在读” `211硕博` [985在读] https://example.invalid/211硕博'
        self.assertEqual(self.apply(text)["text"], text)

    def test_identity_is_local_not_an_unbounded_context_keyword(self):
        for text in ("985。另一件事是在读资料", "211\n硕博都在场", "985 211", "985211在读"):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text)["text"], text)

    def test_generated_spans_and_original_offsets_remain_auditable(self):
        result = self.apply("🙂211硕博与985在读；费用211元。")
        self.assertEqual(result["text"], "🙂二幺幺硕博与九八五在读；费用211元。")
        self.assertEqual([(edit["start"], edit["end"]) for edit in result["edits"]], [(1, 4), (7, 10)])
        self.assertEqual([result["text"][start:end] for start, end in result["protected_spans"]], ["二幺幺", "九八五"])

    def test_other_policy_behavior_matches_frozen_v2(self):
        for text in ("u1s1，985 211。学历只是一个方面。", "985本硕毕业，211本博连读", "她是211本科毕业",
                     "大学购买985本教材，收到211本硕士论文", "985名大学生，211所大学", "大学数学作业：计算985/211",
                     "标识是u1s1，字段值为u1s1", "985元学费，211路公交", "360借条，360度，J D K 八 tomcat 九。"):
            with self.subTest(text=text):
                before = frozen_preprocess(text, resources=fixture())
                after = self.apply(text)
                before.pop("policy_version")
                after.pop("policy_version")
                self.assertEqual(after, before)

    def test_frozen_sources_remain_unchanged(self):
        root = Path(__file__).resolve().parents[1] / "src/readcue"
        hashes = {"context_reading_policy.py": "69412b2fc689e94279a63457fd57222e42f01aa1331b99843a01ef52c3dcfd16",
                  "context_reading_policy_v2.py": "f120db643f8c734d0d636697783d58177ed553f5380234065fc0b296d28227b9"}
        for filename, expected in hashes.items():
            self.assertEqual(hashlib.sha256((root / filename).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
