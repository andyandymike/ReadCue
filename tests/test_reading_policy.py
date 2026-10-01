"""Original synthetic lexicon fixtures and text; no evaluation data is read."""

import copy
import unittest

from readcue.reading_policy import POLICY_VERSION, preprocess, protected_spans, validate_lexicon


def entry(identity, surface, kind, action, sensitive=False, **extra):
    return {"id": identity, "surfaces": [{"text": surface, "case_sensitive": sensitive}],
            "kind": kind, "action": action, "spoken_form": "Fixture convention",
            "evidence": ["Synthetic fixture, not a real sourced lexicon"],
            "reading_basis": "Original boundary-test convention", "ambiguity_notes": [], **extra}


def fixture():
    return {"schema_version": 1, "policy_version": "synthetic-v1",
            "provenance": {"source": "Original fixture"}, "entries": [
                entry("initial", "QRS", "initialism", "spell_letters"),
                entry("word", "Fable", "word", "read_word"),
                entry("sensitive", "TiNy", "initialism", "spell_letters", sensitive=True),
                entry("ambiguous", "WX", "ambiguous", "preserve_review"),
                entry("mixed", "CloudQRS", "compound", "compose", components=[
                    {"text": "Cloud", "action": "read_word"},
                    {"text": "QRS", "action": "spell_letters"}]),
                entry("version_letters", "KLM", "version_stem", "spell_letters",
                      base_action="letters", version_policy="integer_components"),
                entry("version_word", "Badger", "version_stem", "read_word",
                      base_action="word", version_policy="integer_components"),
            ]}


class ReadingPolicyTests(unittest.TestCase):
    def apply(self, text, lexicon=None):
        result = preprocess(text, lexicon=lexicon or fixture())
        self.assertEqual(result["policy_version"], POLICY_VERSION)
        cursor = 0
        rebuilt = ""
        for edit in result["edits"]:
            self.assertGreaterEqual(edit["start"], cursor)
            self.assertEqual(text[edit["start"]:edit["end"]], edit["original"])
            rebuilt += text[cursor:edit["start"]] + edit["replacement"]
            cursor = edit["end"]
        self.assertEqual(rebuilt + text[cursor:], result["text"])
        self.assertEqual(result["protected_spans"], protected_spans(result["text"]))
        return result

    def test_whole_tokens_case_and_words(self):
        result = self.apply("采用qrs、QRS，FaBlE和fable；TiNy tiny。")
        self.assertEqual(result["text"], "采用Q R S、Q R S，FaBlE和fable；T I N Y tiny。")
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("unknown_ascii_token", result["warnings"][0])
        self.assertEqual(self.apply("FABLE fable")["warnings"], [])

    def test_compound_keeps_actual_word_case(self):
        result = self.apply("CloudQRS cloudqrs CLOUDqrs")
        self.assertEqual(result["text"], "Cloud Q R S cloud Q R S CLOUD Q R S")
        self.assertEqual(result["warnings"], [])

    def test_unknown_ambiguous_and_partial_tokens_are_preserved(self):
        text = "xQRS QRS_extra QRS-extra QRS/extra QRS++ QRS# thing WX u2s2 QRS.foo"
        result = self.apply(text)
        self.assertEqual(result["text"], text)
        self.assertEqual(len(result["warnings"]), 10)
        self.assertEqual(sum("ambiguous_reading" in warning for warning in result["warnings"]), 1)

    def test_sentence_punctuation_is_separate(self):
        self.assertEqual(self.apply("QRS. QRS, QRS! (QRS)")["text"],
                         "Q R S. Q R S, Q R S! (Q R S)")

    def test_literal_regions_do_not_change(self):
        text = ('`QRS klm12` https://example.invalid/QRS9 “QRS” "QRS" '
                "[QRS,klm12] @QRS9 'QRS' 「QRS」 ```QRS``` ")
        result = self.apply(text)
        self.assertEqual(result["text"], text)
        self.assertEqual(result["warnings"], [])

    def test_whole_concatenated_version_sequence(self):
        result = self.apply("klm12Badger3.10.007继续，KLM0；badger08。")
        self.assertEqual(result["text"], "K L M 十二 Badger 三点十点零零七继续，K L M 零；badger 零八。")
        self.assertEqual(result["warnings"], [])
        self.assertEqual(len(result["edits"]), 3)
        self.assertEqual(result["edits"][0]["lexicon_entry_ids"], ["version_letters", "version_word"])

    def test_incomplete_or_unknown_version_tail_changes_nothing(self):
        for value in ("klm12garbage9", "klm12Badger", "klm12x", "klm12-rc1",
                      "klm12.abc", "klm12_badger3", "klm12.3..4", "klm-12", "klm12-", "klm12/"):
            with self.subTest(value=value):
                result = self.apply(value)
                self.assertEqual(result["text"], value)
                self.assertEqual(len(result["warnings"]), 1)

    def test_integer_components_and_leading_zero_convention(self):
        cases = {"10": "十", "11": "十一", "20": "二十", "101": "一百零一",
                 "1010": "一千零一十", "10000": "一万", "10001": "一万零一",
                 "10010": "一万零一十", "100000": "十万", "100000001": "一亿零一",
                 "100010001": "一亿零一万零一", "0010": "零零一零", "0.0.01": "零点零点零一"}
        for version, spoken in cases.items():
            with self.subTest(version=version):
                self.assertEqual(self.apply("Badger" + version)["text"], "Badger " + spoken)
        huge = "Badger12345678901234567"
        self.assertEqual(self.apply(huge)["text"], huge)

    def test_bare_version_stem_keeps_its_declared_action(self):
        result = self.apply("klm BADGER")
        self.assertEqual(result["text"], "K L M BADGER")
        self.assertEqual(result["warnings"], [])

    def test_guard_preserves_mixed_identifiers_and_leaves_numbers(self):
        text = "u2s2 abc_18 foo-v2.3 12345 12.3 plain QRS `123`"
        self.assertEqual([text[start:end] for start, end in protected_spans(text)],
                         ["u2s2", "abc_18", "foo-v2.3", "`123`"])
        result = self.apply("u2s2 KLM12 1234")
        self.assertEqual(result["text"], "u2s2 K L M 十二 1234")
        self.assertEqual([result["text"][start:end] for start, end in result["protected_spans"]], ["u2s2"])

    def test_original_unicode_offsets_and_whitespace_rebuild(self):
        text = "😊中文qrs\n  CloudQRS  klm2badger3\t尾  "
        result = self.apply(text)
        self.assertEqual(result["text"], "😊中文Q R S\n  Cloud Q R S  K L M 二 badger 三\t尾  ")
        self.assertEqual(result["edits"][0]["start"], 3)

    def test_input_is_data_not_policy_instructions(self):
        self.assertEqual(self.apply("请保持QRS原样")["text"], "请保持Q R S原样")

    def test_validate_rejects_conflicts_and_bad_components(self):
        bad = fixture()
        bad["entries"].append(entry("collision", "qrs", "word", "read_word", sensitive=True))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            validate_lexicon(bad)
        bad = fixture()
        bad["entries"][4]["components"][0]["text"] = "Different"
        with self.assertRaisesRegex(ValueError, "components do not match"):
            validate_lexicon(bad)
        for field, value in (("schema_version", 2), ("provenance", {}), ("policy_version", "")):
            bad = fixture()
            bad[field] = value
            with self.assertRaises(ValueError):
                self.apply("QRS", bad)
        bad = fixture()
        bad["entries"][0]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_lexicon(bad)

    def test_lexicon_is_never_mutated_or_spoken_form_used_as_free_text(self):
        lexicon = fixture()
        before = copy.deepcopy(lexicon)
        self.apply("FaBlE qrs CloudQRS klm12", lexicon)
        self.assertEqual(lexicon, before)


if __name__ == "__main__":
    unittest.main()
