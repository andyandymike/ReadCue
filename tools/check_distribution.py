"""Check release archives without extracting or importing their contents."""
import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 10 * 1024 * 1024
FORBIDDEN_PARTS = {"models", "data", "runs", "checkpoints", "outputs", "wandb",
                   ".git", ".cache", "__pycache__"}
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf", ".pyc"}
FORBIDDEN_NAMES = {".netrc", "_netrc", ".pypirc", "credentials.json",
                   "service-account.json", "service_account.json"}


def check_members(members):
    total = 0
    for name, size in members:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Unsafe archive path: " + name)
        if any(part in FORBIDDEN_PARTS or part.startswith((".venv", "venv-")) for part in path.parts):
            raise ValueError("Local runtime artifact in distribution: " + name)
        if path.suffix in FORBIDDEN_SUFFIXES | {".pem", ".key"} or path.name in FORBIDDEN_NAMES or path.name.startswith(".env"):
            raise ValueError("Unexpected artifact in distribution: " + name)
        if size > MAX_FILE_BYTES:
            raise ValueError("Unexpectedly large file in research release: " + name)
        total += size
    if total > MAX_ARCHIVE_BYTES:
        raise ValueError("Distribution exceeds the 10 MiB research-source limit")


def check_wheel(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        check_members((item.filename, item.file_size) for item in entries)
        names = set(archive.namelist())
        if len(names) != len(entries):
            raise ValueError("Duplicate wheel members")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("Expected one package metadata file")
        metadata_name = metadata_names[0]
        metadata_root = metadata_name.rsplit("/", 1)[0]
        for name in names:
            if not (name.startswith("readcue/") and name.endswith(".py")) and name != "readcue/review.html" and not name.startswith(metadata_root + "/"):
                raise ValueError("Unexpected payload in core wheel: " + name)
        for name in ("readcue/__init__.py", "readcue/__main__.py", "readcue/cli.py", "readcue/review.html",
                     metadata_root + "/entry_points.txt", metadata_root + "/licenses/LICENSE"):
            if name not in names:
                raise ValueError("Missing wheel member: " + name)
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if metadata["Name"] != "readcue" or metadata["License-Expression"] != "Apache-2.0":
            raise ValueError("Unexpected package identity/license metadata")
        if metadata.get_all("Requires-Dist"):
            raise ValueError("Core wheel must have no mandatory runtime dependencies")
        if not metadata.get_all("Project-URL"):
            raise ValueError("Missing project links in package metadata")


def check_sdist(path):
    with tarfile.open(path, "r:gz") as archive:
        entries = archive.getmembers()
        if any(not (item.isfile() or item.isdir()) for item in entries):
            raise ValueError("Source archive contains a link or special file")
        check_members((item.name, item.size) for item in entries)
        roots = {PurePosixPath(item.name).parts[0] for item in entries}
        if len(roots) != 1:
            raise ValueError("Source archive must have a single root")
        root = roots.pop() + "/"
        members = {item.name[len(root):]: item for item in entries if item.isfile()}
        if len(members) != sum(item.isfile() for item in entries):
            raise ValueError("Duplicate source members")
        required = ("LICENSE", "README.md", "CONTRIBUTING.md", "SECURITY.md", "pyproject.toml",
                    "src/readcue/cli.py", "src/readcue/review.html", "tests/test_architecture.py", "tests/test_zh_pilot.py",
                    "eval/zh_reading_pilot_v1/freeze.json", "reports/zh-pilot-v1/summary.json",
                    "scripts/archive_zh_pilot.py", "docs/REPRODUCING_ZH_PILOT.md",
                    "tools/synthesize_reading_demo.ps1")
        for name in required:
            if name not in members:
                raise ValueError("Incomplete research source archive: " + name)
        freeze = json.load(archive.extractfile(members["eval/zh_reading_pilot_v1/freeze.json"]))
        for name, expected in freeze["files_sha256"].items():
            if name not in members:
                raise ValueError("Missing frozen source: " + name)
            actual = hashlib.sha256(archive.extractfile(members[name]).read()).hexdigest()
            if actual != expected:
                raise ValueError("Frozen evidence changed inside source archive: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheels, sources = list(args.dist.glob("*.whl")), list(args.dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        parser.error("Expected exactly one wheel and one source archive in --dist")
    check_wheel(wheels[0])
    check_sdist(sources[0])
    print("Distribution check passed: dependency-free wheel and complete frozen research sources.")


if __name__ == "__main__":
    main()
