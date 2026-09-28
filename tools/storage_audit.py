"""Read-only repository file-size audit; never traverse links or reparse points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from datetime import datetime, timezone


def is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def audit(root: Path) -> dict:
    root = root.absolute()
    if is_link(root.lstat()) or not root.is_dir():
        raise ValueError(f"Audit root must be an ordinary directory: {root}")
    totals: dict[str, dict[str, int]] = {}
    skipped: list[str] = []
    errors: list[dict[str, str]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            # Recheck queued directories before opening them.
            if is_link(directory.lstat()):
                skipped.append(str(directory.relative_to(root)))
                continue
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    relative = path.relative_to(root)
                    try:
                        info = entry.stat(follow_symlinks=False)
                        if is_link(info):
                            skipped.append(str(relative))
                        elif stat.S_ISDIR(info.st_mode):
                            pending.append(path)
                        elif stat.S_ISREG(info.st_mode):
                            bucket = totals.setdefault(relative.parts[0], {"files": 0, "bytes": 0})
                            bucket["files"] += 1
                            bucket["bytes"] += info.st_size
                        else:
                            skipped.append(str(relative))
                    except OSError as exc:
                        errors.append({"path": str(relative), "error": str(exc)})
        except OSError as exc:
            errors.append({"path": str(directory.relative_to(root)), "error": str(exc)})
    groups = sorted(
        ({"path": name, **counts} for name, counts in totals.items()),
        key=lambda row: (-row["bytes"], row["path"]),
    )
    return {
        "root": str(root),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "logical bytes of regular files; hard links are counted per path",
        "total_files": sum(row["files"] for row in groups),
        "total_bytes": sum(row["bytes"] for row in groups),
        "groups": groups,
        "skipped_links_or_special_files": sorted(skipped),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout; create no file")
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Root: {result['root']}")
        print(f"{'Logical GiB':>12} {'Files':>9}  Path")
        for row in result["groups"]:
            print(f"{row['bytes'] / (1024 ** 3):12.4f} {row['files']:9d}  {row['path']}")
        print(f"Total: {result['total_bytes']:,} bytes / {result['total_files']:,} files")
        print(f"Skipped links/special files: {len(result['skipped_links_or_special_files'])}")
        for path in result["skipped_links_or_special_files"]:
            print(f"  skipped: {path}")
        for error in result["errors"]:
            print(f"  error: {error['path']}: {error['error']}")
        print("Logical sizes only; linked targets are excluded. Files may change during a scan.")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
