"""Loopback-only, versioned saves for an existing prepared review packet."""
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit

from .data_pipeline import verify_prepared
from .inference_contract import jsonl_bytes
from .review_contract import (
    MAX_SUGGESTION_BYTES, REVIEW_KEYS, ReviewError, ReviewRound, _CHOICE_UNSET,
    _decode_rows, _json_bytes, _json_loads, _plain_path, _sha,
    _validate_review, _validate_review_choice, validate_review_rows,
)


MAX_BODY_BYTES = 512 * 1024


def _now():
    return datetime.now(timezone.utc).isoformat()


def _git_safe(path, *, directory=False):
    for ancestor in path.parents:
        if (ancestor / ".git").exists():
            relative = path.relative_to(ancestor).as_posix()
            result = subprocess.run(
                ["git", "--literal-pathspecs", "-C", str(ancestor), "ls-files", "--", relative],
                capture_output=True, check=False)
            if result.returncode != 0 or result.stdout.strip():
                raise ReviewError("保存位置包含 Git 跟踪的文件，请使用仓库外的审阅文件。")
            if not directory:
                # check-ignore takes concrete filenames, not pathspecs. History
                # and temporary files are checked at their actual write paths.
                ignored = subprocess.run(
                    ["git", "-C", str(ancestor), "check-ignore", "--quiet", "--", str(path)],
                    capture_output=True, check=False)
                if ignored.returncode != 0:
                    raise ReviewError("保存位置位于 Git 仓库内且未被忽略，请使用仓库外的审阅文件。")
            break


def _write_exclusive(path, body):
    _git_safe(path)
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_bytes(path, body):
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _git_safe(path)
            _git_safe(temporary)
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
        self.round = ReviewRound(self.prepared, self.annotations, self.templates, self.manifest_hash)
        self.fixed.add(self.round.suggestion_path)
        self.suggestions = self.round.suggestions
        self.suggestion_hash = self.round.suggestion_hash
        self.suggestions_by_id = self.round.suggestions_by_id
        self.check_paths()
        if not self.annotations.exists():
            raise ReviewError("找不到审阅文件；请先复制准备包的 annotations.jsonl，再启动审阅。")
        self.validate_rows(self.annotations.read_bytes())


    def check_suggestions(self):
        self.round.check_suggestions()

    def suggestion_review_metadata(self, rows):
        return self.round.suggestion_review_metadata(rows)

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
        return validate_review_rows(body, self.templates)


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
                after = jsonl_bytes(rows, sort_keys=True)
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
                _git_safe(event / "before.jsonl")
                _git_safe(event / "change.json")
                _write_exclusive(event / "before.jsonl", before)
                # Exercise the receipt's real atomic-write path before replacing
                # annotations too; extension-only ignore rules must fail early.
                _atomic_bytes(event / "change.json", _json_bytes(audit))
                descriptor, name = tempfile.mkstemp(prefix="." + self.annotations.name + ".", suffix=".tmp", dir=self.annotations.parent)
                temporary = Path(name)
                committed = False
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        _git_safe(temporary)
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
