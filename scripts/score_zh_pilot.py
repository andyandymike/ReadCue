"""Score the authored Chinese pilot; missing/failed cases stay in the denominator."""
import argparse
from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path
import unicodedata


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def surface(text):
    text = unicodedata.normalize('NFC', text)
    table = str.maketrans({'。': '.', '“': '"', '”': '"', '‘': "'", '’': "'", '、': ','})
    text = ''.join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in text)
    text = ' '.join(text.translate(table).split())
    # Chinese layout spaces are cosmetic; Latin/number/identifier word boundaries are not.
    return re.sub(r'(?<=[\u3400-\u9fff]) +| +(?=[\u3400-\u9fff])', '', text)


def validate_cases(cases):
    seen = set()
    for case in cases:
        if not case.get('id') or case['id'] in seen:
            raise ValueError('Missing/duplicate case id')
        seen.add(case['id'])
        refs = case.get('acceptable_outputs')
        if not isinstance(refs, list) or not refs or any(not isinstance(x, str) or not x.strip() for x in refs):
            raise ValueError('References must be a nonempty list of nonempty strings')
        if not case.get('family') or not case.get('track') or not case.get('text'):
            raise ValueError('Missing family, track or text')
        if case['track'] == 'preservation' and any(surface(x) != surface(case['text']) for x in refs):
            raise ValueError('Preservation references must preserve source content')
    if not cases:
        raise ValueError('No cases')


def summarize(rows):
    total = len(rows)
    ok = sum(r['status'] == 'ok' for r in rows)
    strict = sum(r['strict_correct'] for r in rows)
    normalized = sum(r['surface_correct'] for r in rows)
    preservation = [r for r in rows if r['track'] == 'preservation']
    changed = sum(r['status'] == 'ok' and r['changed_source'] for r in preservation)
    missing = sum(r['status'] != 'ok' for r in preservation)
    return dict(total=total, completed=ok, failed_or_missing=total-ok,
                strict_correct=strict, strict_accuracy=strict/total if total else None,
                surface_correct=normalized, surface_accuracy=normalized/total if total else None,
                preservation_total=len(preservation), preservation_changed=changed,
                preservation_failed_or_missing=missing,
                preservation_violation_rate=(changed+missing)/len(preservation) if preservation else None)


def score(cases, predictions):
    validate_cases(cases)
    ids = {c['id'] for c in cases}
    by_id = {}
    for pred in predictions:
        pid = pred.get('id')
        if pid not in ids or pid in by_id:
            raise ValueError('Duplicate or unknown prediction id: ' + str(pid))
        by_id[pid] = pred
    rows = []
    for case in cases:
        pred = by_id.get(case['id'], {})
        output = pred.get('prediction', '')
        status = pred.get('status', 'missing')
        output_valid = isinstance(output, str) and bool(output.strip())
        if not output_valid:
            output = ''
            if status == 'ok':
                status = 'empty_or_invalid'
        valid = status == 'ok'
        rows.append(dict(id=case['id'], family=case['family'], track=case['track'],
                         text=case['text'], acceptable_outputs=case['acceptable_outputs'],
                         prediction=output, status=status, output_valid=output_valid,
                         strict_correct=valid and output in case['acceptable_outputs'],
                         surface_correct=valid and surface(output) in {surface(x) for x in case['acceptable_outputs']},
                         changed_source=surface(output) != surface(case['text'])))
    tracks, families = defaultdict(list), defaultdict(list)
    for row in rows:
        tracks[row['track']].append(row)
        families[row['family']].append(row)
    return dict(overall=summarize(rows), by_track={k:summarize(v) for k,v in sorted(tracks.items())},
                families=dict(total=len(families), all_correct=sum(all(r['surface_correct'] for r in v) for v in families.values())),
                cases=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = score(read_jsonl(args.cases), read_jsonl(args.predictions))
    result['cases_sha256'] = hashlib.sha256(Path(args.cases).read_bytes()).hexdigest()
    result['predictions_sha256'] = hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['by_track'], ensure_ascii=False))


if __name__ == '__main__':
    main()
