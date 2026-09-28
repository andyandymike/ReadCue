"""Small, dependency-free helpers for verifiable experiment assets."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from urllib.parse import quote
from urllib.request import urlopen

PILOT_MANIFEST = "eval/zh_reading_pilot_v1/freeze.json"
BASELINE_REPOS = {"tn": "leeoxiang/qwen3-0.6b-zh-tn", "qwen": "Qwen/Qwen3-0.6B"}
MODEL_FILES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "model.safetensors", "README.md", "LICENSE", "chat_template.jinja",
}
REQUIRED_MODEL_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"}


def _plain_file(path):
    info = path.lstat()
    return (stat.S_ISREG(info.st_mode)
            and not getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _check_model_directory(directory, names, *, require_files=False):
    """Loaders may consume unlisted templates/adapters; reject all extra entries."""
    expected = set(names)
    allowed = expected | {"readcue_manifest.json"}
    entries = list(directory.iterdir())
    for entry in entries:
        if entry.name not in allowed:
            raise ValueError("Unexpected model directory entry; use a clean asset directory: " + entry.name)
        if not _plain_file(entry):
            raise ValueError("Model assets must be ordinary files, not links or directories: " + entry.name)
    if require_files and not expected.issubset(entry.name for entry in entries):
        raise ValueError("Model directory is missing registered assets")


def hash_file(path, *, git_blob=False):
    """Hash large files with bounded memory; Git hashes include their blob header."""
    path = Path(path)
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(b"blob " + str(path.stat().st_size).encode("ascii") + b"\0")
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    """Publish a complete JSON file, leaving its prior version intact on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def project_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest path escapes project: " + str(relative))
    return path


def verify_pilot(root):
    """Do not update historical hashes to make a modified experiment pass."""
    freeze = read_json(project_path(root, PILOT_MANIFEST))
    for name, expected in freeze["files_sha256"].items():
        if hash_file(project_path(root, name)) != expected:
            raise ValueError("Frozen pilot file changed: " + name)
    return freeze


def _source_digest(record):
    lfs = record.get("lfs") or {}
    expected = lfs.get("sha256")
    git_blob = expected is None
    expected = record.get("blobId") if git_blob else expected
    length = 40 if git_blob else 64
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, expected):
        raise ValueError("Missing valid upstream digest for " + record["rfilename"])
    return expected, git_blob


def fetch_model(baseline, destination, revision=None, *, opener=urlopen):
    """Fetch only the two current public baselines, verifying before publication."""
    repo = BASELINE_REPOS[baseline]
    ref = quote(revision or "main", safe="")
    with opener(f"https://huggingface.co/api/models/{repo}/revision/{ref}?blobs=true", timeout=60) as response:
        metadata = json.load(response)
    resolved = metadata["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Upstream did not resolve to a commit")
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision) and revision != resolved:
        raise ValueError("Resolved revision differs from requested commit")
    records = [item for item in metadata["siblings"] if item["rfilename"] in MODEL_FILES]
    names = [item["rfilename"] for item in records]
    if len(names) != len(set(names)) or not REQUIRED_MODEL_FILES.issubset(names):
        raise ValueError("Missing required model files or duplicate asset names")
    # Validate all advertised hashes before creating partial downloads.
    for record in records:
        _source_digest(record)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _check_model_directory(destination, names)
    manifest = {"repo": repo, "revision": resolved, "files": []}
    for record in records:
        name = record["rfilename"]
        path = destination / name
        expected, git_blob = _source_digest(record)
        if path.exists():
            if hash_file(path, git_blob=git_blob) != expected:
                raise FileExistsError("Existing asset differs from upstream: " + str(path))
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".partial", dir=destination)
            temporary = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    with opener(f"https://huggingface.co/{repo}/resolve/{resolved}/{name}?download=true", timeout=60) as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(chunk)
                if hash_file(temporary, git_blob=git_blob) != expected:
                    raise ValueError("Upstream digest mismatch: " + name)
                # Never publish unverified bytes under the final asset name.
                if path.exists():
                    raise FileExistsError("Asset appeared during download: " + str(path))
                temporary.rename(path)
            finally:
                temporary.unlink(missing_ok=True)
        manifest["files"].append({"name": name, "bytes": path.stat().st_size,
            "sha256": hash_file(path), "upstream_lfs_sha256": None if git_blob else expected,
            "upstream_git_blob_id": expected if git_blob else None})
    _check_model_directory(destination, names, require_files=True)
    write_json(destination / "readcue_manifest.json", manifest)
    _check_model_directory(destination, names, require_files=True)
    return manifest


def verify_model(directory, repo, revision, *, trusted=None):
    """Match labels and every local asset to recorded, verified source content."""
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Inference requires a resolved HF commit, not a moving branch")
    directory = Path(directory)
    manifest_path = directory / "readcue_manifest.json"
    if not _plain_file(manifest_path):
        raise ValueError("Model manifest must be an ordinary file, not a link")
    manifest = read_json(manifest_path)
    if manifest.get("repo") != repo or manifest.get("revision") != revision:
        raise ValueError("Model manifest repo/revision does not match the requested baseline")
    if trusted and (trusted["repo"] != repo or trusted["revision"] != revision):
        raise ValueError("Trusted model manifest has a different identity")
    pinned = {item["name"]: item["sha256"] for item in (trusted or {}).get("files", [])}
    names = [item["name"] for item in manifest["files"]]
    if len(names) != len(set(names)) or not REQUIRED_MODEL_FILES.issubset(names):
        raise ValueError("Missing required model files or duplicate asset names")
    if pinned and set(names) != set(pinned):
        raise ValueError("Model asset list differs from the frozen source manifest")
    _check_model_directory(directory, names, require_files=True)
    for item in manifest["files"]:
        name = item["name"]
        if name not in MODEL_FILES:
            raise ValueError("Unexpected model file: " + name)
        path = project_path(directory, name)
        actual = hash_file(path)
        if actual != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("Model asset changed: " + name)
        if name in pinned:
            if actual != pinned[name]:
                raise ValueError("Model asset differs from frozen source: " + name)
        elif item.get("upstream_lfs_sha256"):
            if actual != item["upstream_lfs_sha256"]:
                raise ValueError("Model LFS digest mismatch: " + name)
        elif item.get("upstream_git_blob_id"):
            if hash_file(path, git_blob=True) != item["upstream_git_blob_id"]:
                raise ValueError("Model Git blob digest mismatch: " + name)
        else:
            raise ValueError("No source digest for asset; fetch it through ReadCue: " + name)
    return manifest
