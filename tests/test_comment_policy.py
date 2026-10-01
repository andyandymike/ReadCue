"""Original synthetic boundary cases; no evaluation samples or labels loaded."""

import unittest

from readcue.comment_policy import POLICY_VERSION, preprocess, protected_spans


class CommentPolicyTests(unittest.TestCase):
    def apply(self, text, platform=None, unicode=None):
        result = preprocess(text, platform_emotes=platform or set(),
                            unicode_emotes=unicode or set())
        self.assertEqual(result["policy_version"], POLICY_VERSION)
        cursor = 0
        rebuilt = ""
        for edit in result["edits"]:
            self.assertGreaterEqual(edit["start"], cursor)
            self.assertEqual(text[edit["start"]:edit["end"]], edit["original"])
            rebuilt += text[cursor:edit["start"]] + edit["replacement"]
            cursor = edit["end"]
        self.assertEqual(rebuilt + text[cursor:], result["text"])
        return result

    def test_prefix_preserves_body_and_line_breaks(self):
        text = " \t回复 @某个昵称 ： \t正文  保留\r\n回复 @另一个人: 后文  "
        result = self.apply(text)
        self.assertEqual(result["text"], "正文  保留\r\n回复 @另一个人: 后文  ")
        self.assertEqual(len(result["edits"]), 1)
        self.assertEqual(result["edits"][0]["reason"], "omit_reply_prefix")

    def test_prefix_never_crosses_a_line(self):
        for text in ("回复 @某人\n正文: 后文", "回复 @某人\r正文：后文",
                     "回复 @某人\u2028正文：后文", "回复 @某人\u2029正文: 后文",
                     "第一行\n回复 @某人: 后文", "回复 @ : 后文", "请回复 @某人: 后文"):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text)["text"], text)
        self.assertEqual(self.apply("回复 @某人:\n正文")["text"], "\n正文")

    def test_quoted_prefix_code_urls_and_mentions_are_preserved(self):
        texts = (
            '“回复 @某人: [微笑]😊”',
            '"回复 @某人: [微笑]😊"',
            "'回复 @某人: [微笑]😊'",
            "「回复 @某人: [微笑]😊」",
            "`回复 @某人: [微笑]😊`",
            "``包含`的[微笑]😊``",
            "```text\n回复 @某人: [微笑]😊\n```",
            "`未结束的[微笑]😊",
            "https://example.invalid/[微笑]/😊?q=1",
            "www.example.invalid/[微笑]/😊",
            "正文 @某人 保持原样",
            "正文 @某人😊 保持原样",
            "正文 @某人[微笑] 保持原样",
        )
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(self.apply(text, {"微笑"}, {"😊"})["text"], text)
        self.assertEqual(self.apply("正文 @某人 😊", unicode={"😊"})["text"], "正文 @某人 ")

    def test_platform_allowlist_only_and_literal_brackets(self):
        text = "[微笑]正文[变量] [1,2,3] [不在名单] [[微笑]] [数组[微笑]😊]"
        expected = "正文[变量] [1,2,3] [不在名单] [[微笑]] [数组[微笑]😊]"
        self.assertEqual(self.apply(text, {"微笑"}, {"😊"})["text"], expected)
        self.assertEqual(self.apply("[未闭合 😊", {"微笑"}, {"😊"})["text"], "[未闭合 😊")

    def test_emoji_removal_does_not_join_ascii_words(self):
        for text, expected in (("a[微笑]b", "a b"), ("1[微笑][微笑]2", "1 2"),
                               ("A[微笑]😊B", "A B"), ("A😊B", "A B"),
                               ("a[微笑] b", "a b"), ("中[微笑]文", "中文")):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text, {"微笑"}, {"😊"})["text"], expected)

    def test_unicode_longest_complete_sequence_and_unknowns(self):
        allowed = {"😊", "🙂", "🙂\u200d↔️", "✌️", "✌️🏻", "🇨🇳"}
        text = "好😊🙂\u200d↔️✌️🏻🇨🇳 新🙂\u200d未知 ✌🏽 🇨🇦 🔧"
        result = self.apply(text, unicode=allowed)
        self.assertEqual(result["text"], "好 新🙂\u200d未知 ✌🏽 🇨🇦 🔧")
        self.assertEqual([edit["original"] for edit in result["edits"]],
                         ["😊", "🙂\u200d↔️", "✌️🏻", "🇨🇳"])

    def test_unknown_modifier_joiner_keycap_and_tag_sequences_remain_intact(self):
        cases = (
            ("😊🏽", {"😊"}),
            ("🙂\u200d😊", {"😊", "🙂"}),
            ("😊\ufe0f", {"😊"}),
            ("😊\u0301", {"😊"}),
            ("1\ufe0f\u20e3", {"1"}),
            ("🏴\U000e0067\U000e0062\U000e007f", {"🏴"}),
            ("🇨🇦", {"🇨"}),
            ("😊\u200d", {"😊"}),
            ("\u200d😊", {"😊"}),
        )
        for text, allowed in cases:
            with self.subTest(text=text):
                self.assertEqual(self.apply(text, unicode=allowed)["text"], text)

    def test_classifier_preserves_possible_meaning_and_warns(self):
        for classifier in ("个", "只", "把", "颗", "条", "朵", "枚", "台"):
            for emote, platform, unicode in (("😊", set(), {"😊"}),
                                              ("[微笑]", {"微笑"}, set())):
                text = f"给你一{classifier} {emote}"
                result = self.apply(text, platform, unicode)
                self.assertEqual(result["text"], text)
                self.assertEqual(result["edits"], [])
                self.assertEqual(len(result["warnings"]), 1)
                self.assertTrue(result["warnings"][0].startswith("needs_review:"))

    def test_predicate_preserves_emoji_as_possible_noun(self):
        for before in ("他是", "它像", "我当", "已经成为", "后来变成"):
            for emote, platform, unicode in (("🤖", set(), {"🤖"}),
                                              ("[机器人]", {"机器人"}, set())):
                text = before + emote
                result = self.apply(text, platform, unicode)
                self.assertEqual(result["text"], text)
                self.assertEqual(result["edits"], [])
                self.assertEqual(len(result["warnings"]), 1)
                self.assertTrue(result["warnings"][0].startswith("needs_review:predicate_before_emote:"))
        self.assertEqual(self.apply("他是 🤖", unicode={"🤖"})["text"], "他是 🤖")

    def test_mixed_edits_keep_original_codepoint_offsets_and_no_reformatting(self):
        text = "回复 @测试😊:  好😊！\n  alpha[微笑]beta\t\n末尾  "
        result = self.apply(text, {"微笑"}, {"😊"})
        self.assertEqual(result["text"], "好！\n  alpha beta\t\n末尾  ")
        self.assertEqual([edit["reason"] for edit in result["edits"]],
                         ["omit_reply_prefix", "omit_unicode_emote", "omit_platform_emote"])
        self.assertEqual(result["edits"][1]["start"], text.index("😊", text.index("好")))

    def test_protected_spans_can_be_used_on_preprocessed_text(self):
        text = '前 `a[微笑]` 中 https://example.invalid/[x] “引文😊” 后 [1,[2]] @用户😊'
        spans = protected_spans(text)
        values = [text[start:end] for start, end in spans]
        self.assertEqual(values, ["`a[微笑]`", "https://example.invalid/[x]", "“引文😊”", "[1,[2]]", "@用户😊"])
        for before, after in zip(spans, spans[1:]):
            self.assertLess(before[1], after[0])

    def test_no_implicit_dictionary_number_or_case_rewrites(self):
        text = "ABC xyz v17 123 456 [unknown] @正文 ☃ 🛠️  \n"
        self.assertEqual(self.apply(text)["text"], text)

    def test_invalid_allowlist_entries_fail_closed(self):
        for platform, unicode in (({""}, set()), ({"[微笑]"}, set()),
                                  ({"跨\n行"}, set()), (set(), {""})):
            with self.assertRaises(ValueError):
                preprocess("原文", platform_emotes=platform, unicode_emotes=unicode)


if __name__ == "__main__":
    unittest.main()
