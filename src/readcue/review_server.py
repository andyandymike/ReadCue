"""Loopback-only, versioned saves for an existing prepared review packet."""
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit

from .data_pipeline import REVIEW_KEYS, verify_prepared


MAX_BODY_BYTES = 512 * 1024
MAX_SUGGESTION_BYTES = 8 * 1024 * 1024
MAX_HISTORY_RECEIPT_BYTES = 64 * 1024
TRACKS = {"core_tn", "context_reading", "preservation"}
SUGGESTION_POLICIES = {"bilibili-ai-review-v1", "bilibili-ai-review-v2",
                       "bilibili-ai-review-v3", "bilibili-ai-review-v4", "bilibili-ai-review-v5",
                       "bilibili-ai-review-v6"}
SINGLE_CONFIRM_POLICIES = {"bilibili-ai-review-v5", "bilibili-ai-review-v6"}
SUGGESTION_KEYS = {"id", "text_sha256", "decision", "acceptable_outputs", "track", "family",
                   "context_required", "reason", "flags"}
AUDIT_STATUSES = {"text_only_ok", "needs_reading_annotation", "needs_decision", "exclude_suggested"}
READING_MODES = {"letters_en", "word_en", "mixed", "number_zh", "digits_zh", "omit",
                 "symbol_reading", "pinyin_to_hanzi", "handle", "url", "needs_decision"}
_CHOICE_UNSET = object()


class ReviewError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows):
    lines = []
    for row in rows:
        value = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
        # Keep compatibility with the frozen runner's str.splitlines() reader.
        for char in ("\u0085", "\u2028", "\u2029"):
            value = value.replace(char, f"\\u{ord(char):04x}")
        lines.append(value + "\n")
    return "".join(lines).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json_loads(value):
    def invalid_constant(_):
        raise ValueError("non-finite JSON number")

    return json.loads(value, object_pairs_hook=_unique_object, parse_constant=invalid_constant)


def _decode_rows(body):
    try:
        return [_json_loads(line) for line in body.decode("utf-8").split("\n") if line.strip()]
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ReviewError("审阅文件不是有效的 UTF-8 JSONL，请保留文件并检查格式。") from error


def _validate_review(review, text):
    if not isinstance(review, dict) or set(review) != REVIEW_KEYS:
        raise ReviewError("审阅内容必须完整包含七个规定字段，不能增加其他字段。")
    for key in ("status", "reviewer", "track", "family", "notes"):
        if not isinstance(review[key], str):
            raise ReviewError("状态、审阅人、读法类型、情境分类和备注必须是文本。")
    if not isinstance(review["acceptable_outputs"], list) or any(
            not isinstance(item, str) for item in review["acceptable_outputs"]):
        raise ReviewError("可接受读法必须是文本列表。")
    try:
        for value in [review[key] for key in ("status", "reviewer", "track", "family", "notes")] + review["acceptable_outputs"]:
            value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ReviewError("审阅文本含有无法保存的 Unicode 字符，请重新输入相应内容。") from error
    if review["context_required"] is not None and type(review["context_required"]) is not bool:
        raise ReviewError("是否依赖上下文只能填写是、否或暂未判断。")
    status = review["status"]
    if status not in {"pending", "approved", "excluded", "uncertain"}:
        raise ReviewError("未知的审阅状态。")
    if status == "pending":
        return
    if not review["reviewer"].strip():
        raise ReviewError("请填写审阅人姓名或代号。")
    if status in {"excluded", "uncertain"}:
        if not review["notes"].strip():
            raise ReviewError("待讨论或排除的记录必须说明原因。")
        return
    refs = review["acceptable_outputs"]
    if not refs or any(not item.strip() for item in refs):
        raise ReviewError("通过的记录必须填写至少一种非空的完整读法。")
    if review["context_required"] is not False:
        raise ReviewError("需要外部上下文或尚未判断的记录不能进入当前单句评估，请先标记为待讨论。")
    if review["track"] not in TRACKS:
        raise ReviewError("请选择有效的读法类型。")
    if not review["family"].strip():
        raise ReviewError("通过的记录必须填写情境分类。")
    if review["track"] == "preservation" and any(item != text for item in refs):
        raise ReviewError("原文保留的读法必须与原文完全一致，包括空白和换行。")


def _validate_review_choice(choice, review):
    if choice is _CHOICE_UNSET:
        return
    if not isinstance(choice, str) or choice not in {"", "keep", "edit", "uncertain", "excluded"}:
        raise ReviewError("表单判断必须是有效的选项。")
    if review["status"] != "pending":
        expected = ("keep" if review["track"] == "preservation" else "edit") if review["status"] == "approved" else review["status"]
        if choice != expected:
            raise ReviewError("表单判断与保存的审核状态或读法类型不一致。")


def _plain_path(path, *, kind=None):
    """Check existing components before resolve can hide a symlink/junction."""
    path = Path(path).absolute()
    for item in reversed((path, *path.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ReviewError("审阅及历史记录路径不能经过符号链接或重解析点。")
        if item != path and not stat.S_ISDIR(info.st_mode):
            raise ReviewError("审阅保存路径的上级必须是普通目录。")
        if item == path and kind == "file" and not stat.S_ISREG(info.st_mode):
            raise ReviewError("审阅记录必须是普通文件。")
        if item == path and kind == "directory" and not stat.S_ISDIR(info.st_mode):
            raise ReviewError("历史记录位置必须是普通目录。")
    return path.resolve()


def _git_safe(path, *, directory=False):
    for ancestor in path.parents:
        if (ancestor / ".git").exists():
            relative = path.relative_to(ancestor).as_posix()
            result = subprocess.run(
                ["git", "--literal-pathspecs", "-C", str(ancestor), "ls-files", "--", relative],
                capture_output=True, check=False)
            if result.returncode != 0 or result.stdout.strip():
                raise ReviewError("保存位置包含 Git 跟踪的文件，请使用仓库外的审阅文件。")
            probe = path / "review-audit.json" if directory else path
            ignored = subprocess.run(
                ["git", "--literal-pathspecs", "-C", str(ancestor), "check-ignore", "--quiet", "--", str(probe)],
                capture_output=True, check=False)
            if ignored.returncode != 0:
                raise ReviewError("保存位置位于 Git 仓库内且未被忽略，请使用仓库外的审阅文件。")
            break


def _write_exclusive(path, body):
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_bytes(path, body):
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _ReviewStore:
    def __init__(self, prepared, annotations):
        self.prepared = _plain_path(prepared, kind="directory")
        try:
            manifest = verify_prepared(self.prepared)
            self.manifest_hash = _sha((self.prepared / "manifest.json").read_bytes())
            state = _json_loads((self.prepared / "run.json").read_bytes())
            self.frozen = _decode_rows((self.prepared / "review.jsonl").read_bytes())
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ReviewError("准备包校验失败，请检查冻结文件和来源清单。") from error
        self.fixed = {(self.prepared / name).resolve() for name in manifest["files"]}
        self.fixed.update({self.prepared / "manifest.json", self.prepared / "run.json"})
        if state.get("config_path"):
            self.fixed.add(Path(state["config_path"]).resolve())
        self.raw_roots = {Path(value).resolve() for value in
                          (state.get("source_root"), manifest.get("source_root")) if value}
        self.annotations = _plain_path(annotations or self.prepared / "annotations.jsonl", kind="file")
        self.history = self.annotations.with_name(self.annotations.stem + ".history")
        self.lockfile = self.annotations.with_name("." + self.annotations.name + ".review.lock")
        self.lock = threading.RLock()
        self.templates = {}
        for row in self.frozen:
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                    or not row["id"] or row["id"] in self.templates or not isinstance(row.get("text"), str)):
                raise ReviewError("冻结审阅名单存在空值、重复 ID 或无效记录。")
            self.templates[row["id"]] = row
        if not self.templates:
            raise ReviewError("冻结审阅名单为空。")
        self.suggestion_path = self.prepared / "ai-suggestions.json"
        self.fixed.add(self.suggestion_path)
        self.suggestion_bytes = self.read_suggestion_bytes()
        self.suggestions = self.validate_suggestions(self.suggestion_bytes)
        self.suggestion_hash = _sha(self.suggestion_bytes) if self.suggestion_bytes is not None else None
        self.suggestions_by_id = {item["id"]: item for item in self.suggestions["items"]} if self.suggestions else {}
        self.check_paths()
        if not self.annotations.exists():
            raise ReviewError("找不到审阅文件；请先复制准备包的 annotations.jsonl，再启动审阅。")
        self.validate_rows(self.annotations.read_bytes())

    def read_suggestion_bytes(self):
        path = _plain_path(self.suggestion_path, kind="file")
        try:
            if path.stat().st_size > MAX_SUGGESTION_BYTES:
                raise ReviewError("AI 建议侧车超过大小限制，未应用任何建议。")
            body = path.read_bytes()
        except FileNotFoundError:
            return None
        if len(body) > MAX_SUGGESTION_BYTES:
            raise ReviewError("AI 建议侧车超过大小限制，未应用任何建议。")
        return body

    def validate_suggestions(self, body):
        if body is None:
            return None
        try:
            envelope = _json_loads(body.decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError) as error:
            raise ReviewError("AI 建议侧车格式无效，未应用任何建议。") from error
        keys = {"schema_version", "prepared_manifest_sha256", "generator", "human_reviewed",
                "policy_version", "created_at_utc", "items"}
        if not isinstance(envelope, dict) or set(envelope) != keys:
            raise ReviewError("AI 建议侧车缺少规定字段或包含未知字段。")
        if (type(envelope["schema_version"]) is not int
                or envelope["human_reviewed"] is not False or envelope["generator"] != "Codex assistant"
                or not isinstance(envelope["policy_version"], str)
                or envelope["policy_version"] not in SUGGESTION_POLICIES):
            raise ReviewError("AI 建议版本、生成来源或未人工审核声明不符合要求。")
        has_audit = envelope["policy_version"] in {
            "bilibili-ai-review-v3", "bilibili-ai-review-v4", "bilibili-ai-review-v5",
            "bilibili-ai-review-v6"}
        if envelope["schema_version"] != (2 if has_audit else 1):
            raise ReviewError("AI 建议的数据格式与策略版本不匹配。")
        if envelope["prepared_manifest_sha256"] != self.manifest_hash:
            raise ReviewError("AI 建议绑定的准备包哈希不匹配，未应用任何建议。")
        try:
            stamp = envelope["created_at_utc"]
            if not isinstance(stamp, str):
                raise ValueError("timestamp must be text")
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
                raise ValueError("timestamp must use UTC")
        except (ValueError, TypeError) as error:
            raise ReviewError("AI 建议生成时间必须是有效的 UTC 时间。") from error
        if not isinstance(envelope["items"], list):
            raise ReviewError("AI 建议 items 必须是列表。")
        seen = set()
        for item in envelope["items"]:
            expected_keys = SUGGESTION_KEYS | ({"reading_audit"} if has_audit else set())
            if not isinstance(item, dict) or set(item) != expected_keys:
                raise ReviewError("AI 建议条目缺少规定字段或包含未知字段。")
            ident = item["id"]
            if not isinstance(ident, str) or ident not in self.templates or ident in seen:
                raise ReviewError("AI 建议存在未知或重复样本 ID。")
            seen.add(ident)
            text = self.templates[ident]["text"]
            if item["text_sha256"] != _sha(text.encode("utf-8")):
                raise ReviewError("AI 建议与样本原文哈希不匹配，未应用任何建议。")
            if has_audit:
                self.validate_reading_audit(item["reading_audit"], text)
            if (not isinstance(item["decision"], str) or item["decision"] not in {"keep", "edit", "uncertain", "excluded"}
                    or not isinstance(item["track"], str) or item["track"] not in TRACKS | {""}
                    or not isinstance(item["family"], str) or type(item["context_required"]) is not bool
                    or not isinstance(item["reason"], str) or not item["reason"].strip()
                    or not isinstance(item["flags"], list)
                    or any(not isinstance(flag, str) or not flag.strip() for flag in item["flags"])):
                raise ReviewError("AI 建议的判断、分类、上下文、理由或标记类型无效。")
            decision = item["decision"]
            review = {"status": "approved" if decision in {"keep", "edit"} else decision,
                      "reviewer": "AI suggestion validation only", "acceptable_outputs": item["acceptable_outputs"],
                      "track": item["track"], "family": item["family"],
                      "context_required": item["context_required"], "notes": item["reason"]}
            try:
                _validate_review(review, text)
                for flag in item["flags"]:
                    flag.encode("utf-8")
            except (ReviewError, UnicodeError) as error:
                raise ReviewError("AI 建议读法或字段无效：" + str(error)) from error
            if decision == "keep" and (item["track"] != "preservation" or item["acceptable_outputs"] != [text]):
                raise ReviewError("原样保留的 AI 建议必须仅包含完整原文，并使用原文保留类型。")
            if decision == "edit" and item["track"] == "preservation":
                raise ReviewError("填写读法的 AI 建议不能标记为原文保留。")
        return envelope

    @staticmethod
    def validate_reading_audit(audit, text):
        """Validate draft span suggestions, never human labels or scoring inputs."""
        def invalid():
            raise ReviewError("AI 建议的读法复查字段或原文片段无效。")

        if not isinstance(audit, dict) or set(audit) != {"status", "summary", "spans", "issues"}:
            invalid()
        if (not isinstance(audit["status"], str) or audit["status"] not in AUDIT_STATUSES
                or not isinstance(audit["summary"], str) or not audit["summary"].strip()
                or not isinstance(audit["issues"], list)
                or any(not isinstance(issue, str) or not issue.strip() for issue in audit["issues"])
                or not isinstance(audit["spans"], list)):
            invalid()
        if audit["status"] == "text_only_ok" and (audit["spans"] or audit["issues"]):
            invalid()
        if audit["status"] == "needs_decision" and not audit["issues"]:
            invalid()
        values = [audit["summary"], *audit["issues"]]
        previous_end = 0
        for span in audit["spans"]:
            if not isinstance(span, dict) or set(span) != {
                    "start", "end", "surface", "occurrence", "mode", "reading", "reason"}:
                invalid()
            if (any(type(span[key]) is not int for key in ("start", "end", "occurrence"))
                    or not previous_end <= span["start"] < span["end"] <= len(text)
                    or span["occurrence"] < 1
                    or not isinstance(span["mode"], str) or span["mode"] not in READING_MODES
                    or any(not isinstance(span[key], str) or not span[key].strip()
                           for key in ("surface", "reading", "reason"))):
                invalid()
            start = -1
            # Count literal occurrences by Unicode code point, including overlaps.
            for _ in range(span["occurrence"]):
                start = text.find(span["surface"], start + 1)
                if start < 0:
                    invalid()
            if start != span["start"] or text[span["start"]:span["end"]] != span["surface"]:
                invalid()
            previous_end = span["end"]
            values.extend(span[key] for key in ("surface", "reading", "reason"))
        try:
            for value in values:
                value.encode("utf-8")
        except UnicodeError:
            invalid()

    def check_suggestions(self):
        if self.read_suggestion_bytes() != self.suggestion_bytes:
            raise ReviewError("AI 建议文件已新增、修改或删除；未应用变化，请重新启动审阅服务后载入。", 409)

    def check_paths(self):
        for path, directory in ((self.annotations, False), (self.history, True), (self.lockfile, False)):
            actual = _plain_path(path, kind="directory" if directory else "file")
            if actual != path:
                raise ReviewError("保存位置发生变化，请重新打开审阅服务。")
            if actual in self.fixed or any(actual == root or actual.is_relative_to(root) for root in self.raw_roots):
                raise ReviewError("保存位置不能覆盖冻结材料或原始数据。")
            if directory and any(fixed.is_relative_to(actual) for fixed in self.fixed):
                raise ReviewError("历史目录不能包含冻结材料。")
            _git_safe(actual, directory=directory)

    def check_fixed(self):
        try:
            verify_prepared(self.prepared)
            if _sha((self.prepared / "manifest.json").read_bytes()) != self.manifest_hash:
                raise ValueError("changed manifest")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ReviewError("冻结材料在审阅期间发生变化；已停止保存，请先复核准备包。") from error

    def validate_rows(self, body):
        rows = _decode_rows(body)
        by_id = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ReviewError("审阅文件包含无效记录。")
            ident = row["id"]
            if ident not in self.templates or ident in by_id:
                raise ReviewError("审阅名单有重复或未知 ID，不能保存。")
            original = self.templates[ident]
            immutable = {key: value for key, value in row.items() if key != "review"}
            expected = {key: value for key, value in original.items() if key != "review"}
            if _json_bytes(immutable) != _json_bytes(expected):
                raise ReviewError("原文、来源、上下文或名单已被改动；只能编辑 review 对象。")
            _validate_review(row.get("review"), original["text"])
            by_id[ident] = row
        if set(by_id) != set(self.templates):
            raise ReviewError("审阅文件缺少冻结名单中的记录。")
        return [by_id[row["id"]] for row in self.frozen]

    def archived_suggestions(self, digest):
        """Read a verified old bundle for legacy receipts; never repair or write it."""
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        try:
            directory = _plain_path(self.prepared / "ai-suggestions.history", kind="directory")
            path = _plain_path(directory / (digest + ".json"), kind="file")
            if path.stat().st_size > MAX_SUGGESTION_BYTES:
                return None
            with path.open("rb") as stream:
                body = stream.read(MAX_SUGGESTION_BYTES + 1)
            if len(body) > MAX_SUGGESTION_BYTES or _sha(body) != digest:
                return None
            return self.validate_suggestions(body)
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            # Missing, malformed, rebound, or unsafe archives grant no acknowledgement.
            return None

    def suggestion_receipt_matches(self, available, ident, archives):
        if not isinstance(available, dict):
            return False
        current = self.suggestions_by_id[ident]
        item_hash = _sha(_json_bytes(current))
        if available.get("sha256") == self.suggestion_hash:
            return "item_sha256" not in available or available["item_sha256"] == item_hash
        if (available.get("policy_version") not in SINGLE_CONFIRM_POLICIES
                or self.suggestions["policy_version"] not in SINGLE_CONFIRM_POLICIES):
            return False
        if "item_sha256" in available:
            return available["item_sha256"] == item_hash
        digest = available.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return False
        if digest not in archives:
            archives[digest] = self.archived_suggestions(digest)
        previous = archives[digest]
        if previous is None or previous["policy_version"] != available["policy_version"]:
            return False
        old_item = next((item for item in previous["items"] if item["id"] == ident), None)
        return old_item is not None and _json_bytes(old_item) == _json_bytes(current)

    def suggestion_review_metadata(self, rows):
        """Recognize explicit saves for unchanged suggestions, including amendments.

        Receipts are separate from the seven human review fields. An older match
        must not hide a later committed edit, a restored row, or a changed item.
        A verified archive allows old v5/v6 receipts without item hashes to match.
        Receipts without review hashes never imply this acknowledgement.
        """
        metadata = {"suggestion_review_state": {}, "suggestion_review_choices": {}}
        if not self.suggestions_by_id or not self.history.exists():
            return metadata
        latest = {}
        for event in sorted(self.history.iterdir(), key=lambda path: path.name, reverse=True):
            # Only the server's named transaction directories are receipts.
            if not re.fullmatch(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}", event.name):
                continue
            _plain_path(event, kind="directory")
            receipt = _plain_path(event / "change.json", kind="file")
            try:
                if receipt.stat().st_size > MAX_HISTORY_RECEIPT_BYTES:
                    raise ValueError("oversized history receipt")
                body = receipt.read_bytes()
                if len(body) > MAX_HISTORY_RECEIPT_BYTES:
                    raise ValueError("oversized history receipt")
                audit = _json_loads(body.decode("utf-8"))
            except FileNotFoundError:
                # A transaction directory can exist before its prepared receipt.
                continue
            except (UnicodeError, ValueError, RecursionError) as error:
                raise ReviewError("审阅历史记录格式无效，请保留文件并检查历史目录。", 500) from error
            if not isinstance(audit, dict) or audit.get("status") not in {
                    "prepared", "committed", "failed", "conflict"}:
                raise ReviewError("审阅历史记录状态无效，请保留文件并检查历史目录。", 500)
            if audit["status"] != "committed":
                continue
            ident = audit.get("id")
            if (not isinstance(ident, str) or ident not in self.templates
                    or not isinstance(audit.get("annotation_path"), str)
                    or os.path.normcase(audit["annotation_path"]) != os.path.normcase(str(self.annotations))
                    or any(not isinstance(audit.get(key), str)
                           or not re.fullmatch(r"[0-9a-f]{64}", audit[key])
                           for key in ("old_sha256", "new_sha256"))):
                raise ReviewError("审阅历史记录与当前审阅文件不匹配，请检查历史目录。", 500)
            if ident not in latest:
                latest[ident] = audit
        archives = {}
        for row in rows:
            ident = row["id"]
            audit = latest.get(ident)
            if ident not in self.suggestions_by_id or audit is None:
                continue
            available = audit.get("available_ai_suggestion")
            if (audit.get("saved_review_sha256") == _sha(_json_bytes(row["review"]))
                    and self.suggestion_receipt_matches(available, ident, archives)):
                metadata["suggestion_review_state"][ident] = "draft" if row["review"]["status"] == "pending" else "reviewed"
                if "review_choice" in audit:
                    try:
                        _validate_review_choice(audit["review_choice"], row["review"])
                    except ReviewError as error:
                        raise ReviewError("审阅历史中的表单判断无效，请检查历史目录。", 500) from error
                    metadata["suggestion_review_choices"][ident] = audit["review_choice"]
        return metadata

    def state(self):
        with self.lock:
            self.check_paths()
            self.check_fixed()
            self.check_suggestions()
            body = self.annotations.read_bytes()
            rows = self.validate_rows(body)
            return {"items": rows, "version": _sha(body),
                    "batch": self.prepared.name, "annotation_path": str(self.annotations), "preview": False,
                    "suggestions": self.suggestions, "suggestions_sha256": self.suggestion_hash,
                    **self.suggestion_review_metadata(rows)}

    @contextmanager
    def process_lock(self):
        try:
            descriptor = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise ReviewError("另一个审阅进程正在保存，请稍后重试；若进程已退出，请先检查遗留的保存锁。", 409) from error
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_json_bytes({"pid": os.getpid(), "created_at_utc": _now()}))
                stream.flush()
                os.fsync(stream.fileno())
            yield
        finally:
            self.lockfile.unlink(missing_ok=True)

    def save(self, ident, review, version, review_choice=_CHOICE_UNSET):
        if not isinstance(ident, str) or ident not in self.templates:
            raise ReviewError("没有找到这条冻结样本，不能新增或改写名单。")
        if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version):
            raise ReviewError("保存请求缺少有效版本，请重新载入页面。")
        _validate_review(review, self.templates[ident]["text"])
        _validate_review_choice(review_choice, review)
        with self.lock:
            self.check_paths()
            with self.process_lock():
                self.check_fixed()
                self.check_suggestions()
                before = self.annotations.read_bytes()
                if _sha(before) != version:
                    raise ReviewError("审阅文件已被其他页面或程序更新；本次未保存，请先保留当前编辑并重新载入。", 409)
                rows = self.validate_rows(before)
                self.suggestion_review_metadata(rows)
                item = next(row for row in rows if row["id"] == ident)
                item["review"] = review
                after = _jsonl_bytes(rows)
                new_version = _sha(after)
                self.history.mkdir(exist_ok=True)
                self.check_paths()
                event = self.history / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-") + secrets.token_hex(8))
                event.mkdir()
                audit = {"status": "prepared", "at_utc": _now(), "id": ident,
                         "reviewer": review["reviewer"], "old_sha256": version, "new_sha256": new_version,
                         "annotation_path": str(self.annotations)}
                if review_choice is not _CHOICE_UNSET:
                    audit["review_choice"] = review_choice
                if ident in self.suggestions_by_id:
                    audit["saved_review_sha256"] = _sha(_json_bytes(review))
                    audit["available_ai_suggestion"] = {"sha256": self.suggestion_hash,
                        "item_sha256": _sha(_json_bytes(self.suggestions_by_id[ident])),
                        "generator": self.suggestions["generator"], "policy_version": self.suggestions["policy_version"],
                        "human_reviewed": False, "note": "Suggestion was available; this does not assert it was adopted verbatim."}
                _write_exclusive(event / "before.jsonl", before)
                _write_exclusive(event / "change.json", _json_bytes(audit))
                descriptor, name = tempfile.mkstemp(prefix="." + self.annotations.name + ".", suffix=".tmp", dir=self.annotations.parent)
                temporary = Path(name)
                committed = False
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(after)
                        stream.flush()
                        os.fsync(stream.fileno())
                    # Recheck after history/temp writes; never replace a known newer file.
                    self.check_paths()
                    self.check_fixed()
                    self.check_suggestions()
                    if self.annotations.read_bytes() != before:
                        raise ReviewError("保存期间文件被外部程序更改；本次未覆盖，请保留编辑并重新载入。", 409)
                    os.replace(temporary, self.annotations)
                    committed = True
                    audit.update(status="committed", completed_at_utc=_now())
                    _atomic_bytes(event / "change.json", _json_bytes(audit))
                except Exception as error:
                    if not committed:
                        audit.update(status="conflict" if isinstance(error, ReviewError) and error.status == 409 else "failed",
                                     completed_at_utc=_now(), error_type=type(error).__name__)
                        _atomic_bytes(event / "change.json", _json_bytes(audit))
                        raise
                    raise ReviewError("审阅内容已保存，但历史审计状态写入失败；请重新载入并检查历史目录。", 500) from error
                finally:
                    temporary.unlink(missing_ok=True)
                return {"item": item, "version": new_version,
                        **self.suggestion_review_metadata(rows)}


class _ReviewHandler(BaseHTTPRequestHandler):
    server_version = "ReadCueReview/1"

    def log_message(self, *_):
        # Do not log capability tokens, comment text, or reviewer decisions.
        pass

    def _discard_rejected_body(self):
        """Avoid resetting a Windows connection with a small unread POST body."""
        if self.command != "POST" or getattr(self, "_body_read_started", False):
            return
        self._body_read_started = True
        lengths = self.headers.get_all("Content-Length", [])
        if (self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1
                or len(lengths[0]) > 9 or not re.fullmatch(r"[0-9]+", lengths[0])):
            return
        remaining = int(lengths[0])
        if remaining > MAX_BODY_BYTES:
            return
        deadline = time.monotonic() + 1.0
        previous_timeout = self.connection.gettimeout()
        try:
            while remaining:
                available = deadline - time.monotonic()
                if available <= 0:
                    break
                self.connection.settimeout(available)
                chunk = self.rfile.read1(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            pass
        finally:
            self.connection.settimeout(previous_timeout)
            self.close_connection = True

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        if status >= 400:
            self._discard_rejected_body()
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    def _guard(self):
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0] != self.server.expected_host:
            raise ReviewError("只允许通过当前本机审阅地址访问。", 403)
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1 or origins and origins[0] != self.server.origin:
            raise ReviewError("拒绝跨来源访问，请使用本机审阅页面。", 403)
        if self.headers.get("Sec-Fetch-Site") not in (None, "none", "same-origin"):
            raise ReviewError("拒绝跨站请求，请使用本机审阅页面。", 403)
        if self.client_address[0] != "127.0.0.1":
            raise ReviewError("只允许本机连接。", 403)

    def _dispatch(self, method):
        self._body_read_started = False
        try:
            self.connection.settimeout(5)
            self._guard()
            parsed = urlsplit(self.path)
            if parsed.query or parsed.fragment or parsed.scheme or parsed.netloc:
                raise ReviewError("没有这个审阅地址。", 404)
            prefix = "/" + self.server.token + "/"
            if method == "GET" and parsed.path == prefix:
                self._send(200, files("readcue").joinpath("review.html").read_bytes(), "text/html; charset=utf-8")
            elif method == "GET" and parsed.path == prefix + "api/state":
                state = self.server.store.state()
                state["token"] = self.server.token
                state["default_reviewer"] = self.server.default_reviewer
                self._send(200, _json_bytes(state))
            elif method == "POST" and parsed.path == prefix + "api/save":
                if self.headers.get("Transfer-Encoding") is not None:
                    raise ReviewError("保存请求必须使用明确的正文长度。")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
                    raise ReviewError("保存请求缺少有效的正文长度。", 411)
                if len(lengths[0]) > 9:
                    raise ReviewError("本次审阅内容过大，请缩短备注或读法列表后重试。", 413)
                length = int(lengths[0])
                if length > MAX_BODY_BYTES:
                    raise ReviewError("本次审阅内容过大，请缩短备注或读法列表后重试。", 413)
                if self.headers.get_content_type() != "application/json":
                    raise ReviewError("保存请求必须使用 JSON 格式。", 415)
                self._body_read_started = True
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ReviewError("保存请求没有完整传输，请重试。")
                try:
                    payload = _json_loads(raw.decode("utf-8"))
                except (ValueError, UnicodeError, RecursionError) as error:
                    raise ReviewError("保存内容不是有效的 JSON。") from error
                required_keys = {"id", "review", "version", "token"}
                if (not isinstance(payload, dict) or not required_keys <= set(payload)
                        or set(payload) - required_keys - {"review_choice"}):
                    raise ReviewError("保存请求只能包含样本 ID、review、版本、会话标识和可选表单判断。")
                if (not isinstance(payload["token"], str) or not payload["token"].isascii()
                        or not secrets.compare_digest(payload["token"], self.server.token)):
                    raise ReviewError("审阅会话标识不匹配，请重新打开本机审阅地址。", 403)
                result = self.server.store.save(payload["id"], payload["review"], payload["version"],
                                                payload.get("review_choice", _CHOICE_UNSET))
                self._send(200, _json_bytes(result))
            else:
                raise ReviewError("没有这个审阅地址。", 404)
        except ReviewError as error:
            self._send(error.status, _json_bytes({"error": str(error)}))
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            self._send(500, _json_bytes({"error": "读取或保存审阅记录失败，请检查文件、目录权限及可用磁盘空间。"}))
        except Exception:
            self._send(500, _json_bytes({"error": "本地审阅服务发生异常；请保留当前编辑，检查服务后重试。"}))

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")


def create_server(prepared: Path, annotations: Path | None = None, port: int = 0,
                  reviewer: str | None = None) -> ThreadingHTTPServer:
    """Return a loopback server; caller owns serve_forever/shutdown/server_close.

    Startup validates existing files without writing them. base_url is the private
    browser entry; annotation_path/history_path identify the only save locations.
    Saves coordinate across server processes; other editors trigger version conflicts.
    """
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("端口必须是 0 到 65535 的整数。")
    if reviewer is None:
        try:
            reviewer = getpass.getuser()
        except (OSError, KeyError, ImportError):
            reviewer = ""
    if not isinstance(reviewer, str):
        raise ValueError("默认审阅人必须是文本。")
    store = _ReviewStore(prepared, annotations)
    html = files("readcue").joinpath("review.html").read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", port), _ReviewHandler)
    server.daemon_threads = True
    server.store = store
    server.default_reviewer = reviewer.strip()
    server.token = secrets.token_urlsafe(32)
    server.expected_host = f"127.0.0.1:{server.server_port}"
    server.origin = "http://" + server.expected_host
    server.base_url = server.origin + "/" + server.token + "/"
    server.annotation_path = store.annotations
    server.history_path = store.history
    server.html = html
    return server
