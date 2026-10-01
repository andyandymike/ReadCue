"""Fetch bounded public emoji metadata, independent of inference/evaluation cases."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


SOURCES = {
    'bilibili-basic.json': 'https://api.bilibili.com/x/emote/package?business=reply&ids=1',
    'emoji-test-16.0.txt': 'https://www.unicode.org/Public/emoji/16.0/emoji-test.txt',
}


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, url in SOURCES.items():
        request = urllib.request.Request(url, headers={'User-Agent': 'ReadCue-research/0.1'})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError('Unexpected resource size: ' + name)
            records[name] = dict(url=url, status=response.status, bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(), fetched_at_utc=datetime.now(timezone.utc).isoformat())
        (output / name).write_bytes(raw)
    basic = json.loads((output / 'bilibili-basic.json').read_text(encoding='utf-8'))
    if basic.get('code') != 0:
        raise ValueError('Public platform metadata request failed')
    packages = (basic.get('data') or {}).get('packages') or []
    platform = set()
    for package in packages:
        if package.get('id') != 1:
            raise ValueError('Unrequested package received')
        for emote in package.get('emote') or []:
            token = emote.get('text', '')
            if token.startswith('[') and token.endswith(']') and '\n' not in token:
                platform.add(token[1:-1])
    if not platform:
        raise ValueError('Public basic package contained no usable tokens')
    unicode_emotes, subgroup = set(), None
    for line in (output / 'emoji-test-16.0.txt').read_text(encoding='utf-8').splitlines():
        if line.startswith('# subgroup: '):
            subgroup = line.split(': ', 1)[1]
        if not line or line.startswith('#') or ';' not in line:
            continue
        selected = (subgroup.startswith('face-') or subgroup in {'cat-face', 'monkey-face', 'person-gesture'}
                    or subgroup.startswith('hand-') or subgroup == 'hands')
        codepoints, description = line.split(';', 1)
        if selected and not description.strip().startswith('component'):
            unicode_emotes.add(''.join(chr(int(cp, 16)) for cp in codepoints.split()))
    resources = dict(schema_version=1, policy_version='comment-policy-v1',
        platform_emotes=sorted(platform), unicode_emotes=sorted(unicode_emotes),
        metadata=dict(created_at_utc=datetime.now(timezone.utc).isoformat(), sources=records,
            selection='Basic Bilibili package 1; Unicode 16.0 face-*, cat-face, monkey-face, person-gesture, hand-*, hands',
            source_independence='No evaluation inputs, gold references, annotations or predictions read',
            exclusions='Objects, flags, numbers, operators, hearts and generic symbols are not automatic deletion targets',
            unicode_terms='https://www.unicode.org/license.txt',
            platform_terms='Public platform token metadata only; no image downloads or downstream licensing grant inferred'))
    path = output / 'resources.json'
    path.write_text(json.dumps(resources, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(path=str(path), platform_emotes=len(platform), unicode_emotes=len(unicode_emotes),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), sources=records), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    build(parser.parse_args().output)
