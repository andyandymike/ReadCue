"""Offline comment preparation and reviewed, label-isolated evaluation export.

Heuristics select things to inspect, not confirmed reading errors. No inference,
network access, annotation generation, or train/test split happens here.
"""
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata

from .artifacts import hash_file, project_path, read_json, read_jsonl, write_json
from .comment_sources import import_comments


VERSION = "comment-preparation-v1"
REVIEW_KEYS = {"status", "reviewer", "acceptable_outputs", "track", "family", "context_required", "notes"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
            # Preserve string values while remaining readable by the frozen v1
            # runner, whose JSONL reader uses str.splitlines().
            for separator in ("\u0085", "\u2028", "\u2029"):
                encoded = encoded.replace(separator, f"\\u{ord(separator):04x}")
            stream.write(encoded + "\n")


def _new_output(output, protected=()):
    output = Path(output).resolve()
    for other in protected:
        other = Path(other).resolve()
        if output == other or output.is_relative_to(other) or other.is_relative_to(output):
            raise ValueError("Output must be separate from the source/prepared capture")
    # Third-party text must not become an accidentally committable research file.
    for ancestor in output.parents:
        if (ancestor / ".git").exists():
            checked = subprocess.run(
                ["git", "-C", str(ancestor), "check-ignore", "--quiet", "--",
                 str(output / "readcue-data.jsonl")], capture_output=True, check=False)
            if checked.returncode != 0:
                raise ValueError("Data output inside a Git checkout must be Git-ignored")
            break
    output.mkdir(parents=True, exist_ok=False)
    return output


def _failure(output, state, error):
    state.update(status="failed", completed_at_utc=_now(),
                 error={"type": type(error).__name__, "message": str(error)})
    write_json(output / "run.json", state)


def _manifest(output, stage, filenames, **metadata):
    result = {"schema_version": 1, "pipeline_version": VERSION, "stage": stage,
              **metadata, "files": {name: {"sha256": hash_file(output / name),
                  "bytes": (output / name).stat().st_size} for name in sorted(filenames)}}
    write_json(output / "manifest.json", result)
    return result


def verify_prepared(prepared):
    prepared = Path(prepared).resolve()
    state, manifest = read_json(prepared / "run.json"), read_json(prepared / "manifest.json")
    if state.get("status") != "complete" or manifest.get("stage") != "prepare":
        raise ValueError("Prepared capture is incomplete or has the wrong stage")
    if state.get("manifest_sha256") != hash_file(prepared / "manifest.json"):
        raise ValueError("Prepared manifest changed")
    required = {"review.jsonl", "records.jsonl", "config.json", "selection.json", "sources.json"}
    if not required.issubset(manifest.get("files", {})):
        raise ValueError("Prepared manifest lacks required files")
    for name, expected in manifest["files"].items():
        path = project_path(prepared, name)
        if not path.is_file() or hash_file(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Prepared file changed: " + name)
    return manifest


def text_signals(text):
    """Cheap, explicit selection cues; these never become gold labels."""
    signals = []
    if re.search(r"[0-9０-９]", text):
        signals.append("digits")
    if re.search(r"[A-Za-z]", text):
        signals.append("latin_and_han" if re.search(r"[\u3400-\u9fff]", text) else "latin")
    if re.search(r"([^\s\d])\1{2,}", text):
        signals.append("repeated_character")
    if re.search(r"\[[^\[\]\r\n]{1,24}\]", text):
        signals.append("bracket_expression")
    if any(unicodedata.category(char) in {"So", "Sk", "Sm"} for char in text):
        signals.append("symbol_or_emoji")
    return signals


def _canonicalize(imported):
    buckets, records, duplicates = defaultdict(list), [], []
    rejected = list(imported["rejected"])
    for row in imported["records"]:
        buckets[row["id"]].append(row)
    fields = ("text", "video_id", "comment_id", "root_id", "parent_id", "ctime")
    for ident, members in sorted(buckets.items()):
        if len({tuple(row[key] for key in fields) for row in members}) != 1:
            for row in members:
                rejected.append({"source": row["source"], "reasons": ["conflicting_comment_id"], "record": row})
            continue
        row = dict(members[0])
        row["duplicate_sources"] = [item["source"] for item in members[1:]]
        for item in members[1:]:
            duplicates.append({"id": ident, "kind": "same_id_same_payload",
                               "source": item["source"], "canonical_source": row["source"]})
        records.append(row)
    by_text = defaultdict(list)
    by_comment = {(row["video_id"], row["comment_id"]): row for row in records}
    for row in records:
        text = row["text"]
        row["signals"] = text_signals(text)
        flags = set(row["quality_flags"])
        if not text.strip():
            flags.add("empty_text")
        if "\ufffd" in text:
            flags.add("replacement_character")
        if any(unicodedata.category(c) == "Cc" and c not in "\r\n\t" for c in text):
            flags.add("control_character")
        if len(text) <= 4:
            flags.add("short_text")
        if len(text) > 400:
            flags.add("long_text")
        if re.search(r"https?://", text):
            flags.add("contains_url")
        if "@" in text:
            flags.add("contains_at_sign")
        for key in ("root_id", "parent_id"):
            if row[key] != "0" and (row["video_id"], row[key]) not in by_comment:
                flags.add("unresolved_" + key)
        if row["root_id"] == "0" and row["parent_id"] != "0":
            flags.add("inconsistent_root_parent")
        row["quality_flags"] = sorted(flags)
        if text:
            by_text[text].append(row)
    for members in by_text.values():
        if len(members) > 1:
            ids = sorted(row["id"] for row in members)
            for row in members:
                row["exact_text_group"] = ids[0]
                row["quality_flags"] = sorted(set(row["quality_flags"]) | {"duplicate_text"})
            duplicates.append({"kind": "exact_text", "ids": ids,
                               "text_sha256": hashlib.sha256(members[0]["text"].encode("utf-8")).hexdigest()})
    return records, rejected, duplicates


def _groups(records):
    """Keep a video together, also linking videos sharing an exact nonempty text."""
    parent = {row["video_id"]: row["video_id"] for row in records}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    first_text = {}
    for row in records:
        if row["text"]:
            first = first_text.setdefault(row["text"], row["video_id"])
            a, b = find(first), find(row["video_id"])
            parent[max(a, b)] = min(a, b)
    grouped = defaultdict(list)
    for row in records:
        group_id = "video-component:" + find(row["video_id"])
        row["group_id"] = group_id
        grouped[group_id].append(row)
    return [{"group_id": key, "video_ids": sorted({row["video_id"] for row in rows}),
             "ids": sorted(row["id"] for row in rows), "rows": len(rows)}
            for key, rows in sorted(grouped.items())]


def _context(row, by_comment, parent_semantics):
    if parent_semantics == "root_only":
        if row["root_id"] in {"0", row["comment_id"]}:
            return [], None
        root = by_comment.get((row["video_id"], row["root_id"]))
        if root is None:
            return [], "incomplete_context"
        return [{"id": root["id"], "text": root["text"], "comment_time": root["comment_time"]}], None
    chain, seen = [], {row["comment_id"]}
    parent = row["parent_id"]
    while parent != "0":
        if parent in seen:
            return list(reversed(chain)), "context_cycle"
        seen.add(parent)
        item = by_comment.get((row["video_id"], parent))
        if item is None:
            return list(reversed(chain)), "incomplete_context"
        chain.append({"id": item["id"], "text": item["text"], "comment_time": item["comment_time"]})
        parent = item["parent_id"]
    # A chain ending at another root is structurally inconsistent, even if all IDs exist.
    if row["root_id"] != "0" and chain and chain[-1]["id"] != by_comment.get(
            (row["video_id"], row["root_id"]), {}).get("id"):
        return list(reversed(chain)), "context_root_mismatch"
    return list(reversed(chain)), None


def _rank(seed, pool, key):
    return hashlib.sha256(f"{seed}\0{pool}\0{key}".encode("utf-8")).hexdigest()


def _balanced_sample(rows, size, seed, pool):
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    for members in by_video.values():
        members.sort(key=lambda row: (_rank(seed, pool, row["id"]), row["id"]))
    videos = sorted(by_video, key=lambda key: (_rank(seed, pool, key), key))
    selected, cursor = [], 0
    while len(selected) < size:
        added = False
        for video in videos:
            if cursor < len(by_video[video]):
                selected.append(by_video[video][cursor])
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
        cursor += 1
    return selected


def _sampling(config):
    sample = config["sampling"]
    since, until = date.fromisoformat(sample["since"]), date.fromisoformat(sample["until"])
    if since >= until:
        raise ValueError("Sampling since must precede exclusive until")
    for key in ("seed", "candidates", "controls"):
        if type(sample[key]) is not int or (key != "seed" and sample[key] < 0):
            raise ValueError("Invalid integer sampling field: " + key)
    if sample["candidates"] + sample["controls"] == 0:
        raise ValueError("Choose a nonzero review sample")
    return sample, since, until


def _review_markdown(reviews, summary):
    lines = ["# 近期评论待审阅清单", "", "这是规则抽样，尚无人类审核，也不是已确认的朗读错误或盲测。",
             "填写 annotations.jsonl 的 review 对象；review.jsonl、原文及来源字段保持不变。",
             "批准需填写审核者、可接受读法、track、family，并确认仅凭本条文本可确定读法。",
             "无法确定填 uncertain；排除填 excluded；两者填写审核者与原因。", "",
             f"选中 {len(reviews)} 条，覆盖 {summary['review_videos']} 个视频。上下文仅供审阅，不进入模型输入。",
             "控制组表示未命中这些启发式规则，不代表一定容易或无需处理。",
             "回复关系说明：" + summary["context_policy"]["note"], ""]
    for index, row in enumerate(reviews, 1):
        lines.extend([f"## {index}. {row['id']}", "", f"抽样组：{row['selection_pool']}；线索：{', '.join(row['signals']) or '无'}",
                      f"质量标记：{', '.join(row['quality_flags']) or '无'}", "", "原文（逐行缩进显示）：", ""])
        lines.extend("    " + line for line in row["text"].split("\n"))
        if row["context"]:
            label = ("已保留的根评论（真实直接回复对象未保留）：" if row["context_kind"] == "root_only"
                     else "按上游字段回溯的评论（真实回复关系未独立核实）：")
            lines.extend(["", label, ""])
            for item in row["context"]:
                lines.append("    " + item["id"])
                lines.extend("    " + line for line in item["text"].split("\n"))
        source = row["source"]
        lines.extend(["", f"来源文件：{source['path']}；CSV记录：{source['record_number']}",
                      f"[上游文件]({source['url']})", ""])
    return "\n".join(lines)


def prepare_data(source_root, config_path, output):
    source_root, config_path = Path(source_root).resolve(), Path(config_path).resolve()
    output = _new_output(output, (source_root,))
    state = {"status": "started", "stage": "prepare", "pipeline_version": VERSION,
             "started_at_utc": _now(), "source_root": str(source_root), "config_path": str(config_path)}
    write_json(output / "run.json", state)
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if config.get("schema_version") != 1:
            raise ValueError("Unsupported data configuration schema_version")
        sample, since, until = _sampling(config)
        context_policy = config.get("context", {"parent_semantics": "unverified",
            "note": "上游 parent 字段含义尚未独立核实；不可据此认定完整对话关系。"})
        if (context_policy.get("parent_semantics") not in {"root_only", "unverified"}
                or not isinstance(context_policy.get("note"), str) or not context_policy["note"].strip()):
            raise ValueError("Context policy needs parent_semantics and an explicit note")
        imported = import_comments(source_root, config)
        records, rejected, duplicates = _canonicalize(imported)
        groups = _groups(records)
        by_comment = {(row["video_id"], row["comment_id"]): row for row in records}
        contexts = {}
        for row in records:
            context, issue = _context(row, by_comment, context_policy["parent_semantics"])
            contexts[row["id"]] = context
            row["context_kind"] = context_policy["parent_semantics"]
            if context_policy["parent_semantics"] == "root_only" and row["parent_id"] != "0":
                row["quality_flags"] = sorted(set(row["quality_flags"]) | {"direct_parent_not_preserved"})
            if issue:
                row["quality_flags"] = sorted(set(row["quality_flags"]) | {issue})
        bad_text = {"empty_text", "replacement_character", "control_character"}
        eligible = [row for row in records if since <= date.fromisoformat(row["comment_time"][:10]) < until
                    and not bad_text.intersection(row["quality_flags"])]
        pools = {"candidate": [row for row in eligible if row["signals"]],
                 "control": [row for row in eligible if not row["signals"]]}
        reviews, selections = [], []
        for pool, requested in (("candidate", sample["candidates"]), ("control", sample["controls"])):
            selected = _balanced_sample(pools[pool], requested, sample["seed"], pool)
            for row in selected:
                reviews.append({"id": row["id"], "text": row["text"], "source": row["source"],
                    "context": contexts[row["id"]], "signals": row["signals"],
                    "quality_flags": row["quality_flags"], "group_id": row["group_id"],
                    "context_kind": row["context_kind"],
                    "selection_pool": pool, "review": {"status": "pending", "reviewer": "",
                        "acceptable_outputs": [], "track": "", "family": "", "context_required": None, "notes": ""}})
                selections.append({"id": row["id"], "pool": pool, "video_id": row["video_id"]})
        if not reviews:
            raise ValueError("No reviewable comments in configured range")
        summary = {"pipeline_version": VERSION, "source_records": len(imported["records"]) + len(imported["rejected"]),
            "valid_records": len(records), "rejected_records": len(rejected),
            "duplicate_id_occurrences": sum(item["kind"] == "same_id_same_payload" for item in duplicates),
            "unique_exact_texts": len({row["text"] for row in records}),
            "videos": len({row["video_id"] for row in records}),
            "rows_by_year": dict(sorted(Counter(row["comment_time"][:4] for row in records).items())),
            "eligible_records": len(eligible), "pool_sizes": {key: len(rows) for key, rows in pools.items()},
            "review_records": len(reviews), "review_videos": len({item["video_id"] for item in selections}),
            "selected_by_pool": dict(Counter(row["selection_pool"] for row in reviews)),
            "selected_by_video": dict(sorted(Counter(item["video_id"] for item in selections).items())),
            "signal_counts": dict(sorted(Counter(s for row in eligible for s in row["signals"]).items())),
            "quality_counts": dict(sorted(Counter(s for row in records for s in row["quality_flags"]).items())),
            "leakage_components": len(groups), "split": "exploratory_only_no_train_test_split",
            "annotation_status": "pending_human_review", "training": False, "model_inference": False,
            "license": config["license"], "context_policy": context_policy, "text_transformations": [],
            "notes": ["Heuristic flags are not confirmed reading errors.",
                      "Round-robin video sampling is exploratory, not a population error-rate estimate.",
                      "Exact-text duplicates and same-video records remain linked for future split decisions.",
                      "Author metadata is omitted; mentions and other identifiers inside text are preserved.",
                      "Shortfalls in a pool are reported, never filled from the other pool."]}
        (output / "config.json").write_bytes(config_bytes)
        for name, rows in (("records.jsonl", records), ("rejected.jsonl", rejected),
                           ("duplicates.jsonl", duplicates), ("groups.jsonl", groups), ("review.jsonl", reviews)):
            _jsonl(output / name, rows)
        shutil.copyfile(output / "review.jsonl", output / "annotations.jsonl")
        write_json(output / "sources.json", imported["sources"])
        write_json(output / "selection.json", {"method": "sha256_rank_then_video_round_robin_v1", "sampling": sample,
            "date_timezone": "UTC+08:00", "until_exclusive": True,
            "quality_exclusions": sorted(bad_text), "selected": selections})
        write_json(output / "summary.json", summary)
        (output / "review.md").write_text(_review_markdown(reviews, summary), encoding="utf-8", newline="\n")
        filenames = ["config.json", "records.jsonl", "rejected.jsonl", "duplicates.jsonl", "groups.jsonl",
                     "review.jsonl", "sources.json", "selection.json", "summary.json", "review.md"]
        (output / "code").mkdir()
        for name in ("data_pipeline.py", "comment_sources.py", "artifacts.py"):
            shutil.copyfile(Path(__file__).with_name(name), output / "code" / name)
            filenames.append("code/" + name)
        _manifest(output, "prepare", filenames, editable_files=["annotations.jsonl"],
                  config_sha256=hashlib.sha256(config_bytes).hexdigest(), source_root=str(source_root))
        state.update(status="complete", completed_at_utc=_now(), manifest_sha256=hash_file(output / "manifest.json"))
        write_json(output / "run.json", state)
        return summary
    except Exception as error:
        _failure(output, state, error)
        raise


def export_review(prepared, review_path, output):
    prepared, review_path = Path(prepared).resolve(), Path(review_path).resolve()
    output = _new_output(output, (prepared,))
    state = {"status": "started", "stage": "review_export", "pipeline_version": VERSION,
             "started_at_utc": _now(), "prepared": str(prepared), "review_path": str(review_path)}
    write_json(output / "run.json", state)
    try:
        verify_prepared(prepared)
        frozen = read_jsonl(prepared / "review.jsonl")
        review_bytes = review_path.read_bytes()
        reviewed = [json.loads(line) for line in review_bytes.decode("utf-8").split("\n") if line.strip()]
        state.update(review_sha256=hashlib.sha256(review_bytes).hexdigest(),
                     frozen_review_sha256=hash_file(prepared / "review.jsonl"),
                     prepared_manifest_sha256=hash_file(prepared / "manifest.json"))
        expected = {row["id"]: row for row in frozen}
        if len(reviewed) != len(frozen) or {row["id"] for row in reviewed} != set(expected):
            raise ValueError("Review must contain every frozen ID exactly once")
        inputs, cases, provenance, ordered = [], [], [], []
        by_id = {row["id"]: row for row in reviewed}
        for template in frozen:
            row = by_id[template["id"]]
            immutable = {key: value for key, value in row.items() if key != "review"}
            if immutable != {key: value for key, value in template.items() if key != "review"}:
                raise ValueError("Only review fields may change: " + row["id"])
            review = row["review"]
            if set(review) != REVIEW_KEYS:
                raise ValueError("Unexpected or missing review fields: " + row["id"])
            status = review["status"]
            if status not in {"approved", "excluded", "uncertain"}:
                raise ValueError("All review decisions must be resolved; pending is not approved")
            if not isinstance(review["reviewer"], str) or not review["reviewer"].strip():
                raise ValueError("Each decision must identify its reviewer")
            if not isinstance(review["notes"], str):
                raise ValueError("Review notes must be text")
            ordered.append(row)
            if status != "approved":
                if not review["notes"].strip():
                    raise ValueError("Excluded/uncertain decisions need a reason")
                continue
            refs = review["acceptable_outputs"]
            if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
                raise ValueError("Approved rows need nonempty acceptable_outputs")
            if review["context_required"] is not False:
                raise ValueError("Context-dependent/unknown cases cannot be scored as text-only inputs")
            if review["track"] not in {"core_tn", "context_reading", "preservation"}:
                raise ValueError("Unknown review track")
            if not isinstance(review["family"], str) or not review["family"].strip():
                raise ValueError("Approved rows need a family")
            if review["track"] == "preservation" and any(ref != row["text"] for ref in refs):
                raise ValueError("Preservation references must retain the exact source text")
            inputs.append({"id": row["id"], "text": row["text"]})
            cases.append({"id": row["id"], "text": row["text"], "acceptable_outputs": refs,
                          "track": review["track"], "family": review["family"]})
            provenance.append({"id": row["id"], "source": row["source"], "group_id": row["group_id"],
                               "context": row["context"], "selection_pool": row["selection_pool"],
                               "context_kind": row["context_kind"],
                               "reviewer": review["reviewer"], "notes": review["notes"]})
        if not inputs:
            raise ValueError("No approved cases to export")
        _jsonl(output / "inputs.jsonl", inputs)
        _jsonl(output / "cases.jsonl", cases)
        _jsonl(output / "provenance.jsonl", provenance)
        _jsonl(output / "review_decisions.jsonl", ordered)
        # Preserve the submitted bytes as well as the canonical decision ordering.
        (output / "submitted_annotations.jsonl").write_bytes(review_bytes)
        from .baselines import validate_inputs
        validate_inputs(output / "inputs.jsonl")
        summary = {"reviewed": len(ordered), "approved": len(inputs),
                   "decisions": dict(Counter(row["review"]["status"] for row in ordered)),
                   "input_fields": ["id", "text"], "split": "exploratory_only_no_train_test_split",
                   "license": read_json(prepared / "config.json")["license"],
                   "training": False, "model_inference": False,
                   "note": "Recorded reviewer attestations; software cannot prove human review or TTS correctness."}
        write_json(output / "summary.json", summary)
        _manifest(output, "review_export", ["inputs.jsonl", "cases.jsonl", "provenance.jsonl",
                  "review_decisions.jsonl", "submitted_annotations.jsonl", "summary.json"],
                  review_sha256=state["review_sha256"], frozen_review_sha256=state["frozen_review_sha256"],
                  prepared_manifest_sha256=state["prepared_manifest_sha256"])
        state.update(status="complete", completed_at_utc=_now(), manifest_sha256=hash_file(output / "manifest.json"))
        write_json(output / "run.json", state)
        return summary
    except Exception as error:
        _failure(output, state, error)
        raise
