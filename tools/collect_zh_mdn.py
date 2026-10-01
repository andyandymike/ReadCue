"""Bounded MDN Chinese prose probe; raw third-party material stays outside Git."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO = "mdn/translated-content"
USER_AGENT = "ReadCueSourceProbe/0.1 (+https://github.com/andyandymike/ReadCue)"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/2.5/"
GROUPS = (("web/http", "guides"), ("web/javascript", "guide"), ("web/http/reference", "status"))


def digest(body):
    return hashlib.sha256(body).hexdigest()


def git_digest(body):
    return hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_json(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def api(endpoint, output, label):
    result = subprocess.run(["gh", "api", "repos/" + REPO + "/" + endpoint], capture_output=True)
    body = result.stdout if result.returncode == 0 else result.stderr
    path = output / "metadata" / (label + ".json")
    path.write_bytes(body)
    append_json(output / "requests.jsonl", {"url": "https://api.github.com/repos/" + REPO + "/" + endpoint,
        "at_utc": utc(), "transport": "gh api", "exit_code": result.returncode,
        "saved": path.relative_to(output).as_posix(), "bytes": len(body), "sha256": digest(body)})
    if result.returncode:
        raise RuntimeError("GitHub API failed; see saved response: " + label)
    return json.loads(body)


def download(path, blob_sha, revision, output, name):
    url = "https://raw.githubusercontent.com/" + REPO + "/" + revision + "/" + quote(path, safe="/")
    record = {"url": url, "at_utc": utc(), "source_path": path}
    try:
        with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as response:
            record["http_status"] = response.status
            body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError("File exceeds 1 MiB probe limit")
        if git_digest(body) != blob_sha:
            raise ValueError("Upstream Git blob hash mismatch")
        target = output / "raw" / name
        target.write_bytes(body)
        record.update(status="verified", bytes=len(body), sha256=digest(body), git_blob_sha1=blob_sha,
                      saved=target.relative_to(output).as_posix())
        return body, record
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        if isinstance(error, HTTPError):
            record["http_status"] = error.code
        raise
    finally:
        append_json(output / "requests.jsonl", record)
        time.sleep(0.3)


def prose_blocks(markdown):
    """Keep prose paragraphs, with exact source lines; never expand MDN macros."""
    lines = markdown.splitlines()
    start, heading, fence, front = None, "", None, bool(lines and lines[0] == "---")
    skip_continuation = False
    for index, line in enumerate(lines):
        number = index + 1
        stripped = line.strip()
        if front:
            if index and stripped == "---":
                front = False
            continue
        if not stripped:
            skip_continuation = False
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
        excluded = fence is not None or fence_match is not None
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
        container = re.match(r"^\s*(?:>|\||[-*+]\s|\d+[.)]\s|:\s)", line)
        if container:
            skip_continuation = True
        structural = (not stripped or excluded or skip_continuation or line.startswith((" ", "\t"))
                      or stripped.startswith("#"))
        if structural:
            if start is not None:
                yield heading, start + 1, index, "\n".join(lines[start:index])
                start = None
            if not excluded and stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
            continue
        if start is None:
            start = index
    if start is not None:
        yield heading, start + 1, len(lines), "\n".join(lines[start:])


def display_prose(raw):
    """Only presentation edits; reject syntax that needs a template renderer."""
    if "{{" in raw or "}}" in raw or "![" in raw:
        return None, "macro_or_image"
    if re.search(r"<(?!!--)(?!/?a(?:\s|>))/?[A-Za-z][^>]*>", raw):
        return None, "non_link_html"
    # Formatting syntax inside inline code is literal (e.g. __proto__ and 2**3).
    literals = []

    def protect_code(match):
        literals.append(match.group(1))
        return "\x00CODE" + str(len(literals) - 1) + "\x00"

    text = re.sub(r"`([^`\n]+)`", protect_code, raw)
    if "`" in text:
        return None, "unresolved_inline_code"
    # Escaped punctuation is literal too; do not interpret \*DAV as emphasis.
    text = re.sub(r"\\([\\`*_{}\[\]()#+.!<>~-])", protect_code, text)
    text = re.sub(r"\[([^\]\n]+)\]\((?:[^()\n]|\([^()\n]*\))*\)", r"\1", text)
    text = re.sub(r"</?a(?:\s[^>]*)?>", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    if "*" in text or "_" in text:
        return None, "unresolved_emphasis_or_literal"
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    if any(token in text for token in ("{{", "}}", "`", "](", "][", "<!--")):
        return None, "unresolved_markup"
    for index, literal in enumerate(literals):
        text = text.replace("\x00CODE" + str(index) + "\x00", literal)
    return text, None


def extract_document(body, document, output, seen, rejected):
    rows = []
    for heading, first, last, original in prose_blocks(body.decode("utf-8")):
        rendered, reason = display_prose(original)
        if reason:
            rejected[reason] += 1
            continue
        for match in re.finditer(r"[^。！？]+[。！？][”’\"'）)\]】》」』]*", rendered):
            sentence = match.group().strip()
            if not 12 <= len(sentence) <= 300 or len(re.findall(r"[\u3400-\u9fff]", sentence)) < 6:
                rejected["sentence_length_or_han_count"] += 1
                continue
            key = digest(sentence.encode())
            origin = {"document_id": document["id"], "line_start": first, "line_end": last}
            if key in seen:
                append_json(output / "duplicates.jsonl", {"duplicate_of": seen[key], "origin": origin})
                rejected["exact_duplicate"] += 1
                continue
            sentence_id = "mdn-" + key[:20]
            seen[key] = sentence_id
            rows.append({"id": sentence_id, "text": sentence, **origin,
                "source_url": document["source_url"] + f"#L{first}-L{last}",
                "source_sha256": document["sha256"], "revision": document["revision"],
                "group": document["group"], "heading": heading, "context": rendered,
                "raw_paragraph": original, "display_span": [match.start(), match.end()],
                "transformations": ["markdown_links_to_visible_labels", "remove_inline_code_and_bold_markers",
                                    "remove_anchor_tags", "decode_html_entities", "collapse_whitespace",
                                    "decode_markdown_punctuation_escapes_outside_code", "split_on_chinese_sentence_endings"],
                "license": "CC-BY-SA-2.5", "license_url": LICENSE_URL,
                "attribution": "Mozilla Contributors", "human_reviewed": False,
                "annotation_status": "unlabeled", "provenance": "published_document",
                "has_arabic_digit": bool(re.search(r"[0-9]", sentence)),
                "has_latin": bool(re.search(r"[A-Za-z]", sentence))})
    return rows


def collect(output, revision, per_group):
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "metadata").mkdir()
    state = {"status": "started", "started_at_utc": utc(), "repository": REPO,
             "revision": revision, "documents": 0, "sentences": 0, "training": False,
             "human_reviewed": False, "selection": "sha256(path) order; bounded quota per technical documentation group",
             "per_group": per_group, "collector_sha256": digest(Path(__file__).read_bytes())}
    write_json(output / "summary.json", state)
    (output / "collector-source.py").write_bytes(Path(__file__).read_bytes())
    try:
        license_meta = api("contents/LICENSE.md?ref=" + revision, output, "license")
        download("LICENSE.md", license_meta["sha"], revision, output, "LICENSE.md")
        (output / "ATTRIBUTION.md").write_text(
            "# MDN Chinese prose probe\n\nSource: Mozilla Contributors, mdn/translated-content at " + revision +
            ".\n\nProse is CC-BY-SA 2.5: " + LICENSE_URL +
            "\nSee raw/LICENSE.md and documents.jsonl for file-specific source, title, hash and attribution links.\n\n"
            "Raw Markdown bytes are preserved. Derived sentences remove presentation markup, decode entities, "
            "collapse whitespace and split paragraphs at Chinese sentence endings; each record retains its original "
            "paragraph and line range. Macros, images, non-link HTML, code blocks and structural blocks are excluded.\n\n"
            "This is unlabeled published prose, not user logs or confirmed TTS errors. No human review, model "
            "evaluation or training has taken place. Do not relabel this material as ReadCue Apache-2.0 data.\n",
            encoding="utf-8")
        selected = []
        for index, (parent, child) in enumerate(GROUPS):
            directory = api(f"contents/files/zh-cn/{parent}?ref={revision}", output, f"directory-{index}")
            tree_sha = next(item["sha"] for item in directory if item["name"] == child and item["type"] == "dir")
            tree = api(f"git/trees/{tree_sha}?recursive=1", output, f"tree-{index}")
            if tree.get("truncated"):
                raise ValueError("Source tree was truncated")
            prefix = f"files/zh-cn/{parent}/{child}/"
            paths = [{"path": prefix + item["path"], "sha": item["sha"], "group": parent + "/" + child}
                     for item in tree["tree"] if item["type"] == "blob" and item["path"].endswith("index.md")]
            selected.extend(sorted(paths, key=lambda item: digest(item["path"].encode()))[:per_group])
        write_json(output / "selection.json", selected)
        rejected, seen, rows, documents = Counter(), {}, [], []
        for item in selected:
            identifier = digest(item["path"].encode())[:16]
            body, receipt = download(item["path"], item["sha"], revision, output, identifier + ".md")
            title = re.search(r"(?m)^title:\s*(.+)$", body.decode("utf-8"))
            document = {"id": identifier, "path": item["path"], "revision": revision,
                "group": item["group"], "title": title.group(1).strip() if title else None,
                "source_url": f"https://github.com/{REPO}/blob/{revision}/{item['path']}",
                "attribution_url": f"https://github.com/{REPO}/blame/{revision}/{item['path']}",
                "attribution": "Mozilla Contributors", "license": "CC-BY-SA-2.5",
                "license_url": LICENSE_URL, **receipt}
            append_json(output / "documents.jsonl", document)
            extracted = extract_document(body, document, output, seen, rejected)
            for row in extracted:
                append_json(output / "sentences.jsonl", row)
            rows.extend(extracted)
            documents.append(document)
            state.update(documents=len(documents), sentences=len(rows))
            write_json(output / "summary.json", state)
            print(json.dumps({"documents": len(documents), "sentences": len(rows)}), flush=True)
        digits = [row for row in rows if row["has_arabic_digit"]]
        sample = sorted(digits, key=lambda row: row["id"])[:50] + sorted(
            [row for row in rows if not row["has_arabic_digit"]], key=lambda row: row["id"])[:50]
        for row in sample:
            append_json(output / "inspection-sample.jsonl", row)
        state.update(status="complete", digit_sentences=len(digits), latin_sentences=sum(row["has_latin"] for row in rows),
            by_group=dict(Counter(row["group"] for row in rows)), excluded=dict(rejected),
            raw_document_bytes=sum(document["bytes"] for document in documents),
            evidence_scope="unlabeled technical prose; not user logs, error cases, or a representative benchmark")
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state["finished_at_utc"] = utc()
        write_json(output / "summary.json", state)


def reextract(source, output, revision):
    """Reprocess a completed capture offline, preserving the previous extraction."""
    source = source.resolve()
    previous = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if previous["status"] != "complete" or previous["revision"] != revision:
        raise ValueError("Offline source must be complete and match the selected revision")
    documents = [json.loads(line) for line in (source / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    bodies = []
    for document in documents:
        path = (source / document["saved"]).resolve()
        if source not in path.parents:
            raise ValueError("Raw source path escapes capture")
        body = path.read_bytes()
        if digest(body) != document["sha256"] or git_digest(body) != document["git_blob_sha1"]:
            raise ValueError("Raw source hash changed: " + document["id"])
        bodies.append(body)
    receipts = [json.loads(line) for line in (source / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    evidence = []
    for receipt in receipts:
        path = (source / receipt["saved"]).resolve()
        if source not in path.parents:
            raise ValueError("Request evidence path escapes capture")
        body = path.read_bytes()
        if digest(body) != receipt["sha256"] or len(body) != receipt["bytes"]:
            raise ValueError("Request evidence hash changed: " + receipt["saved"])
        evidence.append((receipt["saved"], body))
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    for name in ("documents.jsonl", "requests.jsonl", "selection.json", "ATTRIBUTION.md", "raw/LICENSE.md"):
        (output / name).write_bytes((source / name).read_bytes())
    for relative, body in evidence:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    (output / "collector-source.py").write_bytes(Path(__file__).read_bytes())
    state = {**previous, "status": "started", "started_at_utc": utc(), "source_capture": str(source),
             "network_requests_this_extraction": 0, "collector_sha256": digest(Path(__file__).read_bytes())}
    write_json(output / "summary.json", state)
    rejected, seen, rows = Counter(), {}, []
    try:
        for document, body in zip(documents, bodies):
            (output / document["saved"]).write_bytes(body)
            extracted = extract_document(body, document, output, seen, rejected)
            rows.extend(extracted)
            for row in extracted:
                append_json(output / "sentences.jsonl", row)
        digits = [row for row in rows if row["has_arabic_digit"]]
        sample = sorted(digits, key=lambda row: row["id"])[:50] + sorted(
            [row for row in rows if not row["has_arabic_digit"]], key=lambda row: row["id"])[:50]
        for row in sample:
            append_json(output / "inspection-sample.jsonl", row)
        state.update(status="complete", sentences=len(rows), digit_sentences=len(digits),
                     latin_sentences=sum(row["has_latin"] for row in rows),
                     by_group=dict(Counter(row["group"] for row in rows)), excluded=dict(rejected))
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state["finished_at_utc"] = utc()
        write_json(output / "summary.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory outside the checkout")
    parser.add_argument("--revision", required=True, help="Exact 40-character upstream commit SHA")
    parser.add_argument("--per-group", type=int, default=20, help="1-30 documents per predefined group")
    parser.add_argument("--offline-from", type=Path, help="Re-extract a completed local capture without networking")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision) or not 1 <= args.per_group <= 30:
        parser.error("Use an exact revision and 1-30 documents per group")
    output = args.output.resolve()
    checkout = Path(__file__).resolve().parents[1]
    if output == checkout or checkout in output.parents:
        parser.error("Raw third-party material must be stored outside the checkout")
    if args.offline_from:
        reextract(args.offline_from, output, args.revision)
    else:
        collect(output, args.revision, args.per_group)


if __name__ == "__main__":
    main()
