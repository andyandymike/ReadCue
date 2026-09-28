"""Fetch two public baseline models at resolved revisions; never execute remote code."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPOS = {'tn':'leeoxiang/qwen3-0.6b-zh-tn', 'qwen':'Qwen/Qwen3-0.6B'}
ALLOWED = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
           'vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json',
           'model.safetensors', 'README.md', 'LICENSE', 'chat_template.jinja'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', choices=REPOS, required=True)
    p.add_argument('--revision', help='Reuse a recorded commit; omitted only for first asset resolution')
    p.add_argument('--destination', required=True)
    args = p.parse_args()
    repo = REPOS[args.baseline]
    ref = args.revision or 'main'
    with urlopen(f'https://huggingface.co/api/models/{repo}/revision/{ref}?blobs=true', timeout=60) as response:
        metadata = json.load(response)
    revision = metadata['sha']
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = dict(repo=repo, revision=revision, files=[])
    print(f'{repo}@{revision}', flush=True)
    for item in metadata['siblings']:
        name = item['rfilename']
        if name not in ALLOWED:
            continue
        path = destination / name
        expected = item.get('lfs', {}).get('sha256')
        if path.exists():
            data = path.read_bytes()
            valid = hashlib.sha256(data).hexdigest() == expected if expected else (
                hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == item.get('blobId'))
            if not valid:
                raise FileExistsError(f'Refusing to overwrite unverified existing file: {path}')
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.partial')
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true'
            with urlopen(url, timeout=60) as response, partial.open('wb') as output:
                count, reported = 0, 0
                while chunk := response.read(1024*1024):
                    output.write(chunk)
                    count += len(chunk)
                    if count - reported >= 100*1024*1024:
                        print(f'{name}: {count//(1024*1024)} MiB', flush=True)
                        reported = count
            partial.rename(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and sha != expected:
            raise ValueError(f'Upstream SHA256 mismatch: {name}')
        manifest['files'].append(dict(name=name, bytes=path.stat().st_size, sha256=sha,
                                      upstream_lfs_sha256=expected))
        print(f'verified {name}: {path.stat().st_size} bytes', flush=True)
    (destination/'readcue_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
