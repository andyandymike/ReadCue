"""Gold-free input validation and JSONL transport; no experiment/model imports."""
import json
from pathlib import Path


def escape_jsonl_separators(text):
    """Preserve JSON string values for readers which split on Unicode lines."""
    for separator in ("\u0085", "\u2028", "\u2029"):
        text = text.replace(separator, f"\\u{ord(separator):04x}")
    return text


def serialize_jsonl(rows, *, sort_keys=False):
    return "".join(escape_jsonl_separators(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys, allow_nan=False)) + "\n"
                   for row in rows)


def jsonl_bytes(rows, *, sort_keys=False):
    return serialize_jsonl(rows, sort_keys=sort_keys).encode("utf-8")


def write_jsonl(path, rows, *, sort_keys=False):
    Path(path).write_bytes(jsonl_bytes(rows, sort_keys=sort_keys))


def parse_inputs(text):
    """Validate decoded JSONL without treating Unicode text as record breaks."""
    rows = [json.loads(line) for line in text.split("\n") if line.strip()]
    seen = set()
    if not rows:
        raise ValueError("No inference inputs")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "text"}:
            raise ValueError("Inference input must contain only id and text")
        if not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in seen:
            raise ValueError("Missing or duplicate inference id")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Inference text must be a nonempty string")
        seen.add(row["id"])
    return rows


def validate_inputs(path):
    return parse_inputs(Path(path).read_text(encoding="utf-8"))
