"""Versioned comment, lexical reading and frozen local TN composition."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'scripts'))
from readcue.artifacts import hash_file, read_json, read_jsonl, verify_pilot, write_json
from readcue.baselines import validate_inputs
from readcue import comment_policy, reading_policy
import run_comment_pipeline as component


def now():
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def run(args, *, executor=subprocess.run):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(experiment='reading-policy-v2', baseline='comment-reading-tn',
                 status='started', phase='preflight', started_at_utc=now(),
                 training=False, external_api_cost_usd=0)
    write_json(output / 'run.json', state)
    try:
        verify_pilot(PROJECT)
        paths = {name: Path(getattr(args, name)).resolve()
                 for name in ('inputs', 'resources', 'lexicon', 'settings')}
        watched = {name: hash_file(path) for name, path in paths.items()}
        inputs = validate_inputs(paths['inputs'])
        resources = component.validate_resources(read_json(paths['resources']))
        lexicon = read_json(paths['lexicon'])
        reading_policy.validate_lexicon(lexicon)
        settings = component.validate_kernel_settings(paths['settings'])
        if not args.model_dir or not args.revision:
            raise ValueError('Explicit local model directory and revision required')
        source_paths = ('scripts/run_reading_pipeline.py', 'src/readcue/reading_policy.py',
                        'scripts/run_comment_pipeline.py', 'src/readcue/comment_policy.py',
                        'scripts/run_zh_reviewed_baseline.py', 'scripts/run_zh_baseline.py',
                        'src/readcue/artifacts.py', 'src/readcue/baselines.py')
        sources = {name: hash_file(PROJECT / name) for name in source_paths}
        meta = dict(experiment=state['experiment'], baseline=state['baseline'],
                    started_at_utc=state['started_at_utc'], training=False, external_api_cost_usd=0,
                    input_sha256=watched['inputs'], resources_sha256=watched['resources'],
                    lexicon_sha256=watched['lexicon'], settings_sha256=watched['settings'],
                    settings=settings, source_sha256=sources, case_count=len(inputs),
                    inference_fields=['id', 'text'], pronunciation_dictionary_used=True,
                    emote_resource_lists_used=True, lexicon_entry_count=len(lexicon['entries']),
                    lexicon_provenance=lexicon.get('provenance'),
                    comment_policy_version=comment_policy.POLICY_VERSION,
                    guard='v1 sequential TN guard plus unresolved mixed ASCII identifiers',
                    edit_coordinates='comment_edits index original text; reading_edits index comment_text; tn_edits index rules prediction',
                    component_experiment_semantics='bilibili-reviewed-v1 identifies the unchanged TN computation component, not this parent experiment or dataset')
        state.update(phase='preprocessing', **meta)
        write_json(output / 'run.json', state)
        rows, transformed = [], []
        for item in inputs:
            started = time.perf_counter()
            comment = comment_policy.preprocess(item['text'],
                platform_emotes=set(resources['platform_emotes']), unicode_emotes=set(resources['unicode_emotes']))
            reading = reading_policy.preprocess(comment['text'], lexicon=lexicon)
            if not isinstance(reading.get('text'), str):
                raise ValueError('Reading policy must return text')
            record = dict(id=item['id'], prediction=reading['text'], raw_output=reading['text'],
                          status='ok' if reading['text'].strip() else 'empty_after_policy',
                          comment_text=comment['text'], comment_edits=comment['edits'],
                          reading_edits=reading['edits'], comment_warnings=comment['warnings'],
                          reading_warnings=reading['warnings'], policy_version=reading['policy_version'],
                          latency_ms=(time.perf_counter() - started) * 1000, generated_tokens=None)
            rows.append(record)
            if record['status'] == 'ok':
                transformed.append(dict(id=item['id'], text=reading['text']))
        versions = {row['policy_version'] for row in rows}
        if len(versions) != 1:
            raise ValueError('Reading policy version changed')
        meta['reading_policy_version'] = next(iter(versions))
        rules = output / 'rules_only'
        rules.mkdir()
        write_jsonl(rules / 'predictions.jsonl', rows)
        transformed_path = output / 'transformed_inputs.jsonl'
        write_jsonl(transformed_path, transformed)
        meta.update(rules_predictions_sha256=hash_file(rules / 'predictions.jsonl'),
                    transformed_inputs_sha256=hash_file(transformed_path), transformed_case_count=len(transformed),
                    empty_after_policy=len(rows) - len(transformed))
        rules_meta = {**meta, 'baseline': 'comment-reading-rules',
                      'predictions_sha256': meta['rules_predictions_sha256'],
                      'failed_cases': meta['empty_after_policy']}
        write_json(rules / 'predictions.meta.json', rules_meta)
        write_json(rules / 'run.json', {**rules_meta, 'status': 'complete'})
        state.update(phase='tn_component', **meta)
        write_json(output / 'run.json', state)
        predictions, child_meta = [], None
        if transformed:
            command = [sys.executable, str(PROJECT / 'scripts/run_zh_reviewed_baseline.py'),
                       '--inputs', str(transformed_path), '--baseline', 'tn', '--output', str(output / 'tn'),
                       '--settings', str(paths['settings']), '--model-dir', str(Path(args.model_dir).resolve()),
                       '--revision', args.revision, '--device', args.device]
            state['command'] = command
            write_json(output / 'run.json', state)
            with (output / 'tn.stdout.log').open('x', encoding='utf-8') as stdout, (output / 'tn.stderr.log').open('x', encoding='utf-8') as stderr:
                process = executor(command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=False)
            state['tn_returncode'] = process.returncode
            if process.returncode:
                raise RuntimeError('TN component failed; see logs and tn/run.json')
            predictions, child_meta = component._validate_component(output, transformed,
                meta['transformed_inputs_sha256'], meta['settings_sha256'], args, sources)
        else:
            (output / 'tn').mkdir()
            write_json(output / 'tn/run.json', dict(status='skipped_empty_input', case_count=0))
        state['phase'] = 'guard_and_merge'
        write_json(output / 'run.json', state)
        by_id = {row['id']: row for row in predictions}
        merged = []
        for row in rows:
            record = {**row, 'rules_prediction': row['prediction'], 'policy_latency_ms': row['latency_ms'],
                      'tn_status': 'not_run_empty_after_policy', 'tn_prediction': None,
                      'tn_edits': [], 'tn_guard_reasons': [], 'tn_latency_ms': 0, 'guard_latency_ms': 0}
            if row['id'] in by_id:
                pred = by_id[row['id']]
                record.update(prediction=pred.get('prediction', ''), raw_output=pred.get('raw_output', ''),
                              status=pred.get('status', 'missing'), tn_status=pred.get('status', 'missing'),
                              tn_prediction=pred.get('prediction', ''), generated_tokens=pred.get('generated_tokens'),
                              input_tokens=pred.get('input_tokens'), tn_latency_ms=pred.get('latency_ms', 0))
                if record['status'] == 'ok':
                    started = time.perf_counter()
                    edits, reasons = component.guarded_tn_edits(row['prediction'], record['raw_output'],
                        record['prediction'], reading_policy.protected_spans)
                    record.update(tn_edits=edits, tn_guard_reasons=reasons,
                                  guard_latency_ms=(time.perf_counter() - started) * 1000)
                    if reasons:
                        record.update(status='tn_guard_rejected', prediction=row['prediction'])
                record['latency_ms'] = record['policy_latency_ms'] + record['tn_latency_ms'] + record['guard_latency_ms']
            merged.append(record)
        write_jsonl(output / 'predictions.jsonl', merged)
        meta.update(predictions_sha256=hash_file(output / 'predictions.jsonl'),
                    failed_cases=sum(row['status'] != 'ok' for row in merged),
                    guard_rejected_cases=sum(row['status'] == 'tn_guard_rejected' for row in merged),
                    truncated_cases=sum(row['tn_status'] == 'truncated' for row in merged),
                    tn_component=child_meta, load_seconds=(child_meta or {}).get('load_seconds', 0),
                    inference_seconds=sum(row['latency_ms'] for row in merged) / 1000)
        for name, path in paths.items():
            if hash_file(path) != watched[name]:
                raise ValueError('Frozen input/resource/settings changed: ' + name)
        if hash_file(transformed_path) != meta['transformed_inputs_sha256']:
            raise ValueError('Transformed inputs changed')
        if any(hash_file(PROJECT / name) != digest for name, digest in sources.items()):
            raise ValueError('Inference source changed')
        meta['finished_at_utc'] = now()
        write_json(output / 'predictions.meta.json', meta)
        state.update(status='complete', phase='complete', **meta)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state['finished_at_utc'] = now()
        write_json(output / 'run.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'resources', 'lexicon', 'settings', 'model-dir', 'revision', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    run(parser.parse_args())
