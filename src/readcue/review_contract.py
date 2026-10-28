"""Shared review validation and read-only suggestion-round evidence.

This module has no HTTP or model dependency. Both the local reviewer and offline
export consume the same seven-field review contract and acknowledgement rules.
Reading a round never edits annotations, suggestions, or their history.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat


REVIEW_KEYS = {"status", "reviewer", "acceptable_outputs", "track", "family", "context_required", "notes"}
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


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


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


def validate_review_rows(body, templates):
    """Validate one exact annotation snapshot and return frozen roster order."""
    rows = _decode_rows(body)
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ReviewError("审阅文件包含无效记录。")
        ident = row["id"]
        if ident not in templates or ident in by_id:
            raise ReviewError("审阅名单有重复或未知 ID，不能保存。")
        original = templates[ident]
        immutable = {key: value for key, value in row.items() if key != "review"}
        expected = {key: value for key, value in original.items() if key != "review"}
        if _json_bytes(immutable) != _json_bytes(expected):
            raise ReviewError("原文、来源、上下文或名单已被改动；只能编辑 review 对象。")
        _validate_review(row.get("review"), original["text"])
        by_id[ident] = row
    if set(by_id) != set(templates):
        raise ReviewError("审阅文件缺少冻结名单中的记录。")
    return [by_id[ident] for ident in templates]


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



class ReviewRound:
    """Read a frozen suggestion bundle and its annotation-specific receipts."""

    def __init__(self, prepared, annotations, templates, manifest_hash):
        self.prepared = _plain_path(prepared, kind="directory")
        self.annotations = _plain_path(annotations, kind="file")
        self.history = self.annotations.with_name(self.annotations.stem + ".history")
        self.templates = templates
        self.manifest_hash = manifest_hash
        self.suggestion_path = self.prepared / "ai-suggestions.json"
        self.suggestion_bytes = self.read_suggestion_bytes()
        self.suggestions = self.validate_suggestions(self.suggestion_bytes)
        self.suggestion_hash = _sha(self.suggestion_bytes) if self.suggestion_bytes is not None else None
        self.suggestions_by_id = {item["id"]: item for item in self.suggestions["items"]} if self.suggestions else {}

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


    def suggestion_receipt_match(self, available, ident, archives):
        """Return the evidence basis for a match, or None; never infer adoption."""
        if not isinstance(available, dict):
            return None
        current = self.suggestions_by_id[ident]
        item_hash = _sha(_json_bytes(current))
        if available.get("sha256") == self.suggestion_hash:
            if "item_sha256" not in available or available["item_sha256"] == item_hash:
                return {"kind": "current_bundle_sha256"}
            return None
        if (available.get("policy_version") not in SINGLE_CONFIRM_POLICIES
                or self.suggestions["policy_version"] not in SINGLE_CONFIRM_POLICIES):
            return None
        if "item_sha256" in available:
            return {"kind": "unchanged_item_sha256"} if available["item_sha256"] == item_hash else None
        digest = available.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        if digest not in archives:
            archives[digest] = self.archived_suggestions(digest)
        previous = archives[digest]
        if previous is None or previous["policy_version"] != available["policy_version"]:
            return None
        old_item = next((item for item in previous["items"] if item["id"] == ident), None)
        if old_item is not None and _json_bytes(old_item) == _json_bytes(current):
            return {"kind": "verified_archived_bundle", "archive_sha256": digest}
        return None


    def suggestion_review_metadata(self, rows):
        """Recognize explicit saves for unchanged suggestions, including amendments.

        Receipts are separate from the seven human review fields. An older match
        must not hide a later committed edit, a restored row, or a changed item.
        A verified archive allows old v5/v6 receipts without item hashes to match.
        Receipts without review hashes never imply this acknowledgement.
        """
        metadata = {"suggestion_review_state": {}, "suggestion_review_choices": {},
                    "suggestion_review_provenance": {}}
        if not self.suggestions_by_id:
            return metadata
        latest = {}
        history = _plain_path(self.history, kind="directory")
        events = sorted(history.iterdir(), key=lambda path: path.name, reverse=True) if history.exists() else []
        for event in events:
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
                latest[ident] = (audit, {"path": receipt.relative_to(history).as_posix(),
                                         "sha256": _sha(body)})
        archives = {}
        for row in rows:
            ident = row["id"]
            if ident not in self.suggestions_by_id:
                continue
            provenance = {"state": "unreviewed", "basis": "no_committed_receipt",
                          "saved_review_sha256": _sha(_json_bytes(row["review"])),
                          "suggestion_item_sha256": _sha(_json_bytes(self.suggestions_by_id[ident])),
                          "receipt": None, "suggestion_match": None}
            metadata["suggestion_review_provenance"][ident] = provenance
            if ident not in latest:
                continue
            audit, receipt_identity = latest[ident]
            provenance.update(receipt=receipt_identity, basis="receipt_does_not_match_snapshot_or_suggestion")
            available = audit.get("available_ai_suggestion")
            match = self.suggestion_receipt_match(available, ident, archives)
            if (audit.get("saved_review_sha256") == _sha(_json_bytes(row["review"]))
                    and match is not None):
                state = "draft" if row["review"]["status"] == "pending" else "reviewed"
                metadata["suggestion_review_state"][ident] = state
                provenance.update(state=state, basis="matching_committed_receipt", suggestion_match=match)
                if "review_choice" in audit:
                    try:
                        _validate_review_choice(audit["review_choice"], row["review"])
                    except ReviewError as error:
                        raise ReviewError("审阅历史中的表单判断无效，请检查历史目录。", 500) from error
                    metadata["suggestion_review_choices"][ident] = audit["review_choice"]
        return metadata

    def export_binding(self, rows, annotation_bytes, frozen_review_sha256, *, require_current_round=False):
        """Bind saved labels to observed round evidence without changing either."""
        metadata = self.suggestion_review_metadata(rows)
        states = metadata["suggestion_review_provenance"]
        counts = {state: sum(item["state"] == state for item in states.values())
                  for state in ("reviewed", "draft", "unreviewed")}
        unconfirmed = counts["draft"] + counts["unreviewed"]
        self.check_suggestions()
        if require_current_round and unconfirmed:
            raise ReviewError(f"当前建议轮仍有 {unconfirmed} 条未确认；未导出，请先确认或使用已保存标注快照模式。")
        suggestions = None
        if self.suggestions is not None:
            suggestions = {key: self.suggestions[key] for key in
                           ("generator", "policy_version", "created_at_utc")}
            suggestions.update(path=str(self.suggestion_path), sha256=self.suggestion_hash,
                               items=len(self.suggestions_by_id))
        return {"schema_version": 1, "contract": "saved_annotation_snapshot_v1",
                "require_current_round": require_current_round,
                "annotation_snapshot": {"path": str(self.annotations), "sha256": _sha(annotation_bytes)},
                "prepared_manifest_sha256": self.manifest_hash,
                "frozen_review_sha256": frozen_review_sha256,
                "current_suggestions": suggestions,
                "round": {"history_path": str(self.history), "state_source": "latest_committed_receipt_v1",
                          "counts": counts, "unconfirmed": unconfirmed,
                          "complete": unconfirmed == 0 if suggestions is not None else None,
                          "items": [{"id": ident, **value} for ident, value in states.items()]},
                "note": "Exports the exact saved annotation snapshot. Unconfirmed newer suggestions do not revoke earlier human decisions; round completion is recorded separately."}
