"""Original synthetic behavior cases; no evaluation data or resources loaded."""

import unittest

from readcue.comment_policy_v2 import POLICY_VERSION, preprocess, protected_spans


class CommentPolicyV2Tests(unittest.TestCase):
    def apply(self, text, *, platform=None, unicode=None, lexical=None):
        result = preprocess(text, platform_emotes=platform or set(),
                            unicode_emotes=unicode or set(), lexical_emotes=lexical)
        self.assertEqual(result["policy_version"], POLICY_VERSION)
        chunks, cursor = [], 0
        for edit in result["edits"]:
            self.assertGreaterEqual(edit["start"], cursor)
            self.assertEqual(text[edit["start"]:edit["end"]], edit["original"])
            chunks.extend((text[cursor:edit["start"]], edit["replacement"]))
            cursor = edit["end"]
        self.assertEqual("".join(chunks) + text[cursor:], result["text"])
        for item in result["semantic_annotations"]:
            self.assertEqual(text[item["original_start"]:item["original_end"]],
                             item["surface"])
            self.assertEqual(result["text"][item["start"]:item["end"]],
                             item["surface"])
            self.assertTrue(any(left <= item["start"] < item["end"] <= right
                                for left, right in result["semantic_protected_spans"]))
            self.assertTrue(item["needs_review"])
        return result

    def test_comparison_slot_keeps_picture_even_if_removal_list_contains_it(self):
        result = self.apply("这段说明写得💩一样绕😊", unicode={"💩", "😊"})
        self.assertEqual(result["text"], "这段说明写得💩一样绕")
        item, = result["semantic_annotations"]
        self.assertEqual(item["reason"], "comparison_after_emote")
        self.assertTrue(item["contextual_noun_slot"])
        self.assertFalse(item["in_lexical_emotes"])
        self.assertEqual([edit["original"] for edit in result["edits"]], ["😊"])

    def test_classifier_and_predicate_slots_keep_complete_pictures(self):
        cases = (("给一颗🍎😊", "给一颗🍎", "classifier_before_emote"),
                 ("外形像🐈😊", "外形像🐈", "predicate_before_emote"),
                 ("改成了一个🧱😊", "改成了一个🧱", "classifier_before_emote"))
        for text, expected, reason in cases:
            with self.subTest(text=text):
                result = self.apply(text, unicode={"🍎", "🐈", "🧱", "😊"},
                                    lexical={"🍎", "🐈", "🧱"})
                self.assertEqual(result["text"], expected)
                self.assertEqual(result["semantic_annotations"][0]["reason"], reason)

    def test_lexical_membership_overrides_removal_but_is_not_a_reading(self):
        result = self.apply("收到了🍋！", unicode={"🍋"}, lexical={"🍋"})
        self.assertEqual(result["text"], "收到了🍋！")
        self.assertEqual(result["edits"], [])
        item, = result["semantic_annotations"]
        self.assertEqual(item["reason"], "lexical_emote")
        self.assertTrue(item["in_lexical_emotes"])
        self.assertFalse(item["contextual_noun_slot"])
        self.assertNotIn("reading", item)

    def test_known_object_genitive_is_distinct_from_emotional_deque(self):
        result = self.apply("🍎的香气😊", unicode={"🍎", "😊"}, lexical={"🍎"})
        self.assertEqual(result["text"], "🍎的香气")
        self.assertEqual(result["semantic_annotations"][0]["reason"],
                         "lexical_noun_modifier")
        emotion = self.apply("好的😊的确如此", unicode={"😊"}, lexical={"🍎"})
        self.assertEqual(emotion["text"], "好的的确如此")
        self.assertEqual(emotion["semantic_annotations"], [])
        obj = self.apply("🍎的确很香", unicode={"🍎"}, lexical={"🍎"})
        self.assertEqual(obj["semantic_annotations"][0]["reason"], "lexical_emote")

    def test_clear_emotion_sequences_still_omit_at_any_sentence_position(self):
        result = self.apply("😊今天😊开工[轻笑]，明天继续😊！",
                            platform={"轻笑"}, unicode={"😊"}, lexical={"🍎"})
        self.assertEqual(result["text"], "今天开工，明天继续！")
        self.assertEqual(len(result["edits"]), 4)
        self.assertEqual(result["semantic_protected_spans"], [])
        self.assertEqual(result["warnings"], [])

    def test_unknown_medial_picture_is_protected_without_external_list(self):
        result = self.apply("试试🧰再接着写😊", unicode={"😊"})
        self.assertEqual(result["text"], "试试🧰再接着写")
        self.assertEqual(result["semantic_protected_spans"], [(2, 3)])
        self.assertEqual(result["semantic_annotations"][0]["reason"],
                         "unknown_unicode_emote")
        self.assertTrue(result["warnings"][0].startswith("needs_review:"))

    def test_unlisted_larger_unicode_sequences_are_not_partially_removed(self):
        sequences = ("✌🏽", "😊\ufe0f", "🙂\u200d😊", "🇨🇦",
                     "1\ufe0f\u20e3", "🏴\U000e0067\U000e0062\U000e007f")
        for seq in sequences:
            with self.subTest(seq=seq):
                result = self.apply("前" + seq + "后", unicode={"✌", "😊", "🙂", "🇨", "1", "🏴"})
                self.assertEqual(result["text"], "前" + seq + "后")
                item, = result["semantic_annotations"]
                self.assertEqual(item["surface"], seq)
                self.assertEqual((item["start"], item["end"]), (1, 1 + len(seq)))

    def test_output_offsets_account_for_reply_emoji_and_ascii_boundary_space(self):
        text = "回复 @读者🐈: A[轻笑]B；给一颗🍎😊，还有🧰。"
        result = self.apply(text, platform={"轻笑"}, unicode={"😊", "🍎"},
                            lexical={"🍎", "🐈"})
        self.assertEqual(result["text"], "A B；给一颗🍎，还有🧰。")
        apple, box = result["semantic_annotations"]
        self.assertEqual((apple["start"], apple["end"]), (7, 8))
        self.assertEqual((box["start"], box["end"]), (11, 12))
        self.assertEqual(apple["original_start"], text.index("🍎"))
        self.assertEqual([item["surface"] for item in result["semantic_annotations"]],
                         ["🍎", "🧰"])
        self.assertEqual([edit["replacement"] for edit in result["edits"]], ["", " ", ""])

    def test_adjacent_semantic_annotations_are_individual_with_merged_guard(self):
        result = self.apply("🍎🧱😊", unicode={"😊"}, lexical={"🍎", "🧱"})
        self.assertEqual(result["text"], "🍎🧱")
        self.assertEqual(result["semantic_protected_spans"], [(0, 2)])
        self.assertEqual([(x["start"], x["end"]) for x in result["semantic_annotations"]],
                         [(0, 1), (1, 2)])

    def test_platform_noun_slot_retains_token_but_other_platform_emote_omits(self):
        result = self.apply("[轻笑]一样的标记，[轻笑]收尾", platform={"轻笑"})
        self.assertEqual(result["text"], "[轻笑]一样的标记，收尾")
        item, = result["semantic_annotations"]
        self.assertEqual(item["surface"], "[轻笑]")
        self.assertTrue(item["contextual_noun_slot"])
        self.assertFalse(item["in_lexical_emotes"])

    def test_literals_unknown_brackets_and_body_mentions_are_not_pending(self):
        texts = ('“🍎像💩一样😊”', '"🍎😊"', "`🍎😊`", "`未闭合🍎😊",
                 "https://example.invalid/🍎/😊", "www.example.invalid/🧰",
                 "正文 @人物🐈 继续", "[未知🍎😊]", "[数组[轻笑]🍎]", "[未闭合🍎")
        for text in texts:
            with self.subTest(text=text):
                result = self.apply(text, platform={"轻笑"}, unicode={"💩", "😊"},
                                    lexical={"🍎", "🐈", "💩"})
                self.assertEqual(result["text"], text)
                self.assertEqual(result["semantic_annotations"], [])
                self.assertEqual(result["warnings"], [])
                self.assertTrue(protected_spans(result["text"]))

    def test_prefix_never_crosses_newline_and_body_whitespace_is_preserved(self):
        text = "回复 @甲: \t正文\n回复 @乙:  后文  "
        self.assertEqual(self.apply(text)["text"], "正文\n回复 @乙:  后文  ")
        for text in ("回复 @甲\n正文: 后文", "回复 @甲\u2028正文: 后文",
                     "回复 @ : 后文", "请回复 @甲: 后文"):
            with self.subTest(text=text):
                self.assertEqual(self.apply(text)["text"], text)

    def test_semantic_context_does_not_cross_newline(self):
        result = self.apply("像\n🍎\n一样", unicode={"🍎"}, lexical={"🍎"})
        self.assertEqual(result["text"], "像\n🍎\n一样")
        self.assertFalse(result["semantic_annotations"][0]["contextual_noun_slot"])

    def test_no_dictionary_names_numbers_or_case_are_invented(self):
        text = "收到🍎 abc ABC v23 45.6！"
        self.assertEqual(self.apply(text, lexical={"🍎"})["text"], text)

    def test_invalid_new_lexical_entries_fail_closed(self):
        for value in ({""}, {None}, {1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                preprocess("正文", platform_emotes=set(), unicode_emotes=set(),
                           lexical_emotes=value)


if __name__ == "__main__":
    unittest.main()
