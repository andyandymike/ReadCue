from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from readcue import semantic_emoji_policy as policy


def resource():
    return {"schema_version": 1, "policy_version": "fixture", "provenance": "synthetic names",
            "entries": [{"id": str(i), "surface": surface, "spoken_form": name, "evidence": "fixture"}
                        for i, (surface, name) in enumerate([("💩", "大便"), ("🔨", "锤子"), ("🍋", "柠檬")])]}


class SemanticEmojiTests(unittest.TestCase):
    def test_explicit_noun_slots_and_ambiguous_suffix(self):
        cases = [("这是💩一样的说明。", "这是大便一样的说明。"),
                 ("请递我一把🔨。", "请递我一把锤子。"),
                 ("他像🍋一般酸。", "他像柠檬一般酸。"),
                 ("今天写完了💩", "今天写完了💩")]
        for original, expected in cases:
            result = policy.preprocess(original, resources=resource())
            self.assertEqual(result["text"], expected)
            self.assertTrue(result["protected_spans"])
        self.assertTrue(result["warnings"])

    def test_code_quotes_mentions_are_literal(self):
        text = '把`一把🔨`和“像💩一样”原样保存，通知@一只🍋。'
        result = policy.preprocess(text, resources=resource())
        self.assertEqual(result["text"], text)
        self.assertEqual(result["edits"], [])

    def test_complete_emoji_sequence_only(self):
        for text in ['一把🔨\u200d🔥', '一把🔨🏽']:
            self.assertEqual(policy.preprocess(text, resources=resource())["text"], text)
        self.assertEqual(policy.preprocess('一把🔨\ufe0f。', resources=resource())["text"], '一把锤子。')

    def test_edits_replay_and_output_protection_after_multiple_lengths(self):
        text = '💩一样，拿一把🔨，🍋一般。'
        result = policy.preprocess(text, resources=resource())
        cursor, chunks = 0, []
        for edit in result['edits']:
            self.assertEqual(text[edit['start']:edit['end']], edit['original'])
            chunks.extend([text[cursor:edit['start']], edit['replacement']]); cursor = edit['end']
        self.assertEqual(''.join(chunks)+text[cursor:],result['text'])
        self.assertEqual([result['text'][a:b] for a,b in result['protected_spans']],['大便','锤子','柠檬'])

    def test_protection_moves_over_prefix_edits_and_rejects_partial_removal(self):
        source='ai一把🔨'
        result={'text':'A I一把🔨','edits':[{'start':0,'end':2,'replacement':'A I'}]}
        self.assertEqual(policy.remap_protection([(4,5)],source,result),[(5,6)])
        for result in [{'text':'ai一把','edits':[{'start':4,'end':5,'replacement':''}]},
                       {'text':'ai物','edits':[{'start':2,'end':5,'replacement':'物'}]}]:
            with self.assertRaises(ValueError):policy.remap_protection([(4,5)],source,result)

    def test_invalid_or_duplicate_resource_fails(self):
        for bad in [{}, {**resource(),'entries':resource()['entries']*2}]:
            with self.assertRaises(ValueError):policy.validate_resources(bad)

    def test_resource_cannot_hide_ordinary_text_in_a_picture_surface(self):
        for surface in ['foo🔨', '🔨abc', '正文💩', '💩文字', '🔨\u200da', '🔨\u200d。', '🔨💩']:
            bad = resource()
            bad['entries'][0]['surface'] = surface
            with self.subTest(surface=surface):
                with self.assertRaises(ValueError):policy.validate_resources(bad)


if __name__ == '__main__':unittest.main()
