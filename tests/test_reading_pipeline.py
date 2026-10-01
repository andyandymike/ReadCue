"""Composition checks using invented terms and mocked TN, without eval data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


pipeline = module('reading_pipeline', PROJECT / 'scripts/run_reading_pipeline.py')
fixtures = module('original_pipeline_fixtures', PROJECT / 'tests/test_comment_pipeline.py')


class ReadingPipelineTests(unittest.TestCase):
    def fixture(self, directory, rows):
        args = fixtures.CommentPipelineTests().fixture(directory, rows)
        lexicon = dict(schema_version=1, policy_version='synthetic-test-v1',
            provenance={'source': 'Invented test acronym, not an evaluation reference'},
            entries=[dict(id='qrs', surfaces=[dict(text='QRS', case_sensitive=False)],
                kind='initialism', action='spell_letters', spoken_form='Q R S',
                reading_basis='project_convention',
                evidence=[dict(url='synthetic:test', title='Original fixture', evidence_type='user_policy', claim='Synthetic spelling test')])])
        path = Path(directory) / 'lexicon.json'
        path.write_text(json.dumps(lexicon), encoding='utf-8')
        args.lexicon = str(path)
        return args

    def test_ordering_provenance_and_mixed_identifier_guard(self):
        rows = [dict(id='letters', text='回复 @甲 :用QRS检查，价格30元[笑]'),
                dict(id='identifier', text='abc2价格30元'), dict(id='empty', text='[笑]')]
        fake = fixtures.CommentPipelineTests().executor({
            'letters': ('用Q R S检查，价格三十元', '30->三十', 'ok'),
            'identifier': ('abc二价格三十元', '2->二\n30->三十', 'ok')})
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, rows)
            state = pipeline.run(args, executor=fake)
            self.assertEqual(state['status'], 'complete')
            self.assertTrue(state['pronunciation_dictionary_used'])
            self.assertEqual(state['lexicon_sha256'], pipeline.hash_file(args.lexicon))
            result = {row['id']: row for row in pipeline.read_jsonl(Path(args.output) / 'predictions.jsonl')}
            self.assertEqual(set(result), {'letters', 'identifier', 'empty'})
            self.assertEqual(result['letters']['prediction'], '用Q R S检查，价格三十元')
            self.assertTrue(result['letters']['comment_edits'])
            self.assertTrue(result['letters']['reading_edits'])
            self.assertEqual(result['identifier']['status'], 'tn_guard_rejected')
            self.assertEqual(result['identifier']['prediction'], 'abc2价格30元')
            self.assertEqual(result['empty']['status'], 'empty_after_policy')
            self.assertEqual(state['failed_cases'], 2)
            self.assertEqual(state['transformed_case_count'], 2)
            with self.assertRaises(FileExistsError):
                pipeline.run(args, executor=fake)

    def test_lexicon_mutation_leaves_failure_record(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id='one', text='QRS')])
            fake = fixtures.CommentPipelineTests().executor({}, mutation=lambda: Path(args.lexicon).write_text('{}'))
            with self.assertRaisesRegex(ValueError, 'changed: lexicon'):
                pipeline.run(args, executor=fake)
            self.assertEqual(pipeline.read_json(Path(args.output) / 'run.json')['status'], 'failed')

    def test_gold_input_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, [dict(id='one', text='QRS', acceptable_outputs=['Q R S'])])
            with self.assertRaises(ValueError):
                pipeline.run(args, executor=lambda *a, **k: self.fail('Must not invoke TN'))
            self.assertEqual(pipeline.read_json(Path(args.output) / 'run.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
