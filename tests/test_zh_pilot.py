import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_zh_pilot import score, surface
from run_zh_baseline import parse_edits


def case(cid='a', text='预算12元。', refs=None, track='core_tn'):
    return dict(id=cid, family='f', track=track, text=text,
                acceptable_outputs=refs or ['预算十二元。'])


class EvaluationTests(unittest.TestCase):
    def test_missing_and_empty_predictions_count_as_failure(self):
        result = score([case(), case('b')], [dict(id='a', prediction='', status='ok')])
        self.assertEqual(result['overall']['total'], 2)
        self.assertEqual(result['overall']['surface_correct'], 0)
        self.assertEqual(result['overall']['failed_or_missing'], 2)

    def test_no_silent_duplicate_or_extra_ids(self):
        for rows in [[dict(id='x')], [dict(id='a'), dict(id='a')]]:
            with self.assertRaises(ValueError): score([case()], rows)

    def test_multiple_readings_and_full_denominator(self):
        result = score([case(refs=['预算十二元。','预算拾贰元。']), case('b')],
                       [dict(id='a', prediction='预算拾贰元.', status='ok')])
        self.assertEqual(result['overall']['surface_accuracy'], 0.5)
        self.assertEqual(result['families']['all_correct'], 0)

    def test_surface_does_not_erase_semantics(self):
        self.assertEqual(surface('金额１２元，已收。'), surface('金额12元,已收.'))
        self.assertNotEqual(surface('不能改'), surface('能改'))
        self.assertNotEqual(surface('user_id'), surface('userid'))
        self.assertNotEqual(surface('12%'), surface('12'))
        self.assertNotEqual(surface('Thank you'), surface('Thankyou'))
        self.assertNotEqual(surface('user_id'), surface('user_ id'))
        self.assertEqual(surface('使用 WiFi 上传'), surface('使用WiFi上传'))

    def test_failure_reason_survives_empty_output(self):
        result = score([case()], [dict(id='a', prediction='', status='truncated')])
        self.assertEqual(result['cases'][0]['status'], 'truncated')
        self.assertFalse(result['cases'][0]['output_valid'])

    def test_preservation_and_parse_failure(self):
        c = case(text='不改。', refs=['不改。'], track='preservation')
        result = score([c], [dict(id='a', prediction='不改。', status='parse_error')])
        self.assertEqual(result['overall']['surface_accuracy'], 0)
        self.assertEqual(result['overall']['preservation_violation_rate'], 1)

    def test_edits_are_ordered_and_nonempty(self):
        self.assertEqual(parse_edits('12元，12号。', '12->十二\n12->十二'), '十二元，十二号。')
        self.assertEqual(parse_edits('保持。', ''), '保持。')
        for raw in ['missing->一', '12->', 'unstructured']:
            with self.assertRaises(ValueError): parse_edits('12元', raw)


if __name__ == '__main__': unittest.main()
