"""Inspect a pinned wheel cache; only --download permits network access or writes."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).absolute().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/maintenance/2026-09-28-wheel-cache.json"
ALLOWED_HOST = "files.pythonhosted.org"


def normalized_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not parsed.path.startswith("/packages/") or parsed.fragment):
        raise ValueError(f"Refusing non-official wheel URL: {url}")


class OfficialRedirectsOnly(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_ordinary_path(path: Path) -> None:
    """Reject linked/reparse ancestors as well as linked destination files."""
    relative = path.relative_to(ROOT)
    current = ROOT
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError(f"Refusing linked/reparse path: {current}")


def file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def read_manifest(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported wheel manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest must contain a nonempty files list")
    seen: set[str] = set()
    for item in files:
        relative = PurePosixPath(item["path"])
        if (relative.parts[:2] != (".cache", "linux-wheels") or len(relative.parts) != 3
                or "\\" in item["path"] or not relative.name.endswith(".whl")
                or item["path"] != relative.as_posix()):
            raise ValueError(f"Wheel destination is outside the permitted cache: {item['path']}")
        if item["path"] in seen:
            raise ValueError(f"Duplicate manifest path: {item['path']}")
        seen.add(item["path"])
        if not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise ValueError(f"Invalid expected size for {item['path']}")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError(f"Invalid SHA-256 for {item['path']}")
        check_url(item["url"])
        if unquote(urlsplit(item["url"]).path.rsplit("/", 1)[-1]) != relative.name:
            raise ValueError(f"Source/destination filename mismatch: {item['path']}")
        if not item.get("package") or not item.get("version"):
            raise ValueError(f"Missing package identity: {item['path']}")
        datetime.fromisoformat(item["mtime_utc"].replace("Z", "+00:00"))
    return files


def inspect(item: dict) -> tuple[Path, bool]:
    target = ROOT.joinpath(*PurePosixPath(item["path"]).parts)
    check_ordinary_path(target)
    if target.exists():
        if not target.is_file() or file_digest(target) != (item["bytes"], item["sha256"]):
            raise FileExistsError(f"Refusing to overwrite conflicting file: {target}")
        return target, True
    return target, False


def publish_without_overwrite(temporary: Path, target: Path) -> None:
    if os.name == "nt":
        # Windows rename fails if target exists, including a concurrently created file.
        temporary.rename(target)
    else:
        # POSIX rename would overwrite. Link/unlink provides a no-clobber publication.
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()


def download(item: dict, target: Path) -> None:
    check_ordinary_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    check_ordinary_path(target)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=target.name + ".", suffix=".partial", delete=False
        ) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            size = 0
            request = Request(item["url"], headers={"Accept-Encoding": "identity"})
            with build_opener(OfficialRedirectsOnly).open(request, timeout=60) as response:
                check_url(response.url)
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(chunk)
                    if size > item["bytes"]:
                        raise ValueError(f"Download exceeded expected size: {target.name}")
                    output.write(chunk)
                    digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if (size, digest.hexdigest()) != (item["bytes"], item["sha256"]):
            raise ValueError(f"Downloaded size/SHA-256 does not match: {target.name}")
        check_ordinary_path(target)
        timestamp = datetime.fromisoformat(item["mtime_utc"].replace("Z", "+00:00")).timestamp()
        os.utime(temporary, (timestamp, timestamp))
        publish_without_overwrite(temporary, target)
        print(f"Restored and verified: {item['path']} ({size:,} bytes)")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--package", action="append", help="Select a package; may be repeated")
    parser.add_argument("--download", action="store_true", help="Download missing verified wheels")
    args = parser.parse_args()
    try:
        files = read_manifest(args.manifest)
        if args.package:
            requested = {normalized_package(name) for name in args.package}
            available = {normalized_package(item["package"]) for item in files}
            if missing := requested - available:
                raise ValueError(f"Packages not in manifest: {', '.join(sorted(missing))}")
            files = [item for item in files if normalized_package(item["package"]) in requested]
        # Inspect all selected destinations before making any changes.
        plan = [(item, *inspect(item)) for item in files]
        missing_bytes = sum(item["bytes"] for item, _, exists in plan if not exists)
        print(f"{'Download' if args.download else 'Dry run'}: {len(plan)} selected wheels; "
              f"{missing_bytes:,} missing bytes")
        for item, target, exists in plan:
            if exists:
                print(f"Already verified: {item['path']}")
            elif args.download:
                download(item, target)
            else:
                print(f"Would restore: {item['path']} ({item['bytes']:,} bytes)")
        if not args.download:
            print("No network access or writes. Add --download to restore missing wheels.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Restore stopped: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
