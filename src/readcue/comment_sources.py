"""Offline import of verified Bilibili CSV captures, preserving comment text."""
import csv
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import io
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from urllib.parse import quote


CHINA_TIME = timezone(timedelta(hours=8))
REQUIRED_COLUMNS = {"bv_id", "oid", "rpid", "root_rpid", "parent_rpid", "ctime", "message"}
POSITIVE_ID = re.compile(r"[1-9][0-9]*")
NONNEGATIVE_ID = re.compile(r"(?:0|[1-9][0-9]*)")
SCIENTIFIC = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+")
BVID = re.compile(r"BV[1-9A-HJ-NP-Za-km-z]{10}")


def _is_link(info):
    return (stat.S_ISLNK(info.st_mode)
            or bool(getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _source_path(root, relative):
    """Manifest paths are portable relative paths, with no links or ADS."""
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or "\0" in relative or PurePosixPath(relative).is_absolute()
            or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("Invalid source-relative path")
    current = root
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        info = current.lstat()
        if _is_link(info):
            raise ValueError("Source paths must not contain links or reparse points: " + relative)
        expected = stat.S_ISREG if index == len(parts) - 1 else stat.S_ISDIR
        if not expected(info.st_mode):
            raise ValueError("Source path is not an ordinary file: " + relative)
    if not current.resolve().is_relative_to(root):
        raise ValueError("Source path escapes capture")
    return current


def _date(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError(name + " must be an ISO calendar date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Invalid " + name) from error


def _configuration(config):
    if not isinstance(config, dict):
        raise ValueError("Source configuration must be an object")
    source_id = config.get("source_id")
    repo, revision = config.get("repo"), config.get("revision")
    if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", source_id):
        raise ValueError("Invalid source_id")
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Expected a GitHub owner/repository")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Source revision must be a fixed 40-character commit")
    license_info = config.get("license")
    if (not isinstance(license_info, dict)
            or any(not isinstance(license_info.get(key), str) or not license_info[key].strip()
                   for key in ("status", "note"))):
        raise ValueError("Source license requires explicit status and note")
    capture = _date(config.get("capture_date"), "capture_date")
    cutoff = _date(config["max_date"], "max_date") if "max_date" in config else capture
    if not date(2009, 1, 1) <= cutoff <= capture < date.max:
        raise ValueError("max_date must be between 2009-01-01 and capture_date")
    lower = int(datetime(2009, 1, 1, tzinfo=CHINA_TIME).timestamp())
    upper = int(datetime.combine(cutoff + timedelta(days=1), time(), CHINA_TIME).timestamp())
    return source_id, repo, revision, {key: license_info[key] for key in ("status", "note")}, lower, upper


def _verify_files(root, config, repo, revision, license_info):
    data_files, evidence_files = config.get("files"), config.get("evidence_files", [])
    if not isinstance(data_files, list) or not data_files or not isinstance(evidence_files, list):
        raise ValueError("Source requires a nonempty files list and an optional evidence_files list")
    verified, seen = [], set()
    for role, files in (("data", data_files), ("evidence", evidence_files)):
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError("Source file manifest entry must be an object")
            relative = entry.get("path")
            path = _source_path(root, relative)
            identity = str(path.resolve()).casefold()
            if identity in seen:
                raise ValueError("Duplicate source file path: " + relative)
            seen.add(identity)
            size, sha256, git_sha = entry.get("bytes"), entry.get("sha256"), entry.get("git_blob_sha")
            if (not isinstance(size, int) or isinstance(size, bool) or size < 0
                    or not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256)
                    or not isinstance(git_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", git_sha)):
                raise ValueError("Invalid source file hashes or byte count: " + relative)
            body = path.read_bytes()
            actual_sha256 = hashlib.sha256(body).hexdigest()
            actual_git_sha = hashlib.sha1(b"blob " + str(len(body)).encode("ascii") + b"\0" + body).hexdigest()
            if len(body) != size or actual_sha256 != sha256 or actual_git_sha != git_sha:
                raise ValueError("Source file bytes/hash mismatch: " + relative)
            receipt = {"source_id": config["source_id"], "repo": repo, "revision": revision,
                       "path": relative, "role": role, "status": "verified", "bytes": len(body),
                       "sha256": actual_sha256, "git_blob_sha": actual_git_sha,
                       "url": f"https://github.com/{repo}/blob/{revision}/{quote(relative, safe='/')}",
                       "license": dict(license_info)}
            verified.append((receipt, body))
    return verified


def _parse_file(receipt, body, lower, upper):
    """CSV record numbers include the header; physical line spans are separate."""
    try:
        text = body.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Invalid UTF-8 source: " + receipt["path"]) from error
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    records, rejected = [], []
    try:
        header = next(reader, None)
        if (header is None or len(header) != len(set(header))
                or not REQUIRED_COLUMNS.issubset(header)):
            raise ValueError("Missing or duplicate CSV columns: " + receipt["path"])
        record_number = 1
        while True:
            line_start = reader.line_num + 1
            raw_row = next(reader, None)
            if raw_row is None:
                break
            record_number += 1
            source = {key: receipt[key] for key in
                      ("source_id", "repo", "revision", "path", "sha256", "url")}
            source.update(record_number=record_number, line_start=line_start, line_end=reader.line_num)
            reasons = []
            if len(raw_row) != len(header):
                reasons.append("column_count_mismatch")
            else:
                row = dict(zip(header, raw_row))
                if not BVID.fullmatch(row["bv_id"]):
                    reasons.append("invalid_bv_id")
                for name in ("oid", "rpid", "root_rpid", "parent_rpid"):
                    pattern = NONNEGATIVE_ID if name in ("root_rpid", "parent_rpid") else POSITIVE_ID
                    if not pattern.fullmatch(row[name]):
                        reasons.append(("scientific_notation:" if SCIENTIFIC.fullmatch(row[name])
                                        else "invalid_identifier:") + name)
                timestamp = None
                if not NONNEGATIVE_ID.fullmatch(row["ctime"]):
                    reasons.append("invalid_timestamp")
                else:
                    try:
                        timestamp = int(row["ctime"])
                    except ValueError:
                        reasons.append("invalid_timestamp")
                    if timestamp is not None and not lower <= timestamp < upper:
                        reasons.append("timestamp_out_of_range")
                if not row["message"].strip():
                    reasons.append("empty_text")
            if reasons:
                rejected.append({"source": source, "reasons": reasons, "raw_row": raw_row})
                continue
            value = row["message"]
            flags = []
            if "\n" in value or "\r" in value:
                flags.append("multiline_text")
            if value != value.strip():
                flags.append("boundary_whitespace")
            if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
                flags.append("control_characters")
            records.append({"id": f"{receipt['source_id']}:{row['bv_id']}:{row['rpid']}",
                            "text": value, "video_id": row["bv_id"], "comment_id": row["rpid"],
                            "root_id": row["root_rpid"], "parent_id": row["parent_rpid"],
                            "ctime": timestamp,
                            "comment_time": datetime.fromtimestamp(timestamp, CHINA_TIME).isoformat(),
                            "source": source, "quality_flags": flags})
    except csv.Error as error:
        raise ValueError("Unrecoverable CSV syntax in " + receipt["path"]
                         + " near physical line " + str(reader.line_num)) from error
    return records, rejected


def import_comments(source_root, config):
    """Verify all local files before parsing; never write, normalize, or deduplicate.

    Rejected rows retain their source fields, including possible author metadata.
    The caller must keep this audit output outside Git and never log raw rows.
    Source acquisition/licensing declarations are receipts, not verified usage grants.
    """
    _, repo, revision, license_info, lower, upper = _configuration(config)
    root = Path(source_root)
    info = root.lstat()
    if _is_link(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("Source root must be an ordinary directory")
    root = root.resolve()
    # Parse exactly these already-verified bytes, never a second filesystem read.
    verified = _verify_files(root, config, repo, revision, license_info)
    records, rejected = [], []
    for receipt, body in verified:
        if receipt["role"] == "data":
            accepted, invalid = _parse_file(receipt, body, lower, upper)
            records.extend(accepted)
            rejected.extend(invalid)
    return {"records": records, "rejected": rejected,
            "sources": [receipt for receipt, _ in verified]}
