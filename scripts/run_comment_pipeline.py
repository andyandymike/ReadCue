"""Apply a frozen comment policy, then the unchanged local TN component."""
import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from readcue.artifacts import hash_file, read_json, read_jsonl, verify_pilot, write_json
from readcue.baselines import validate_inputs


def now():
    return datetime.now(timezone.utc).isoformat()


def load_policy():
    return importlib.import_module("readcue.comment_policy")


def validate_resources(resources):
    if not isinstance(resources, dict) or not isinstance(resources.get("metadata"), dict):
        raise ValueError("Resources require a metadata object")
    for field in ("platform_emotes", "unicode_emotes"):
        values = resources.get(field)
        if not isinstance(values, list) or any(not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError(field + " must be a list of nonempty strings")
    if any("[" in x or "]" in x for x in resources["platform_emotes"]):
        raise ValueError("platform_emotes must contain names without brackets")
    return resources


def validate_kernel_settings(path):
    spec = importlib.util.spec_from_file_location("comment_pipeline_tn_component", PROJECT / "scripts/run_zh_reviewed_baseline.py")
    component = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(component)
    return component.validate_settings(read_json(path))


def guarded_tn_edits(text, raw, prediction, protected):
    """Replay sequential anchors and reject the complete edit set on any risk."""
    edits, reasons, parts, cursor = [], [], [], 0
    spans = protected(text)
    for span in spans:
        if len(span) != 2 or not all(type(x) is int for x in span) or not 0 <= span[0] < span[1] <= len(text):
            raise ValueError("Policy returned an invalid protected span")
    if not isinstance(raw, str) or not isinstance(prediction, str):
        return [], [{"code": "invalid_tn_output_type"}]
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        split = line.rfind("->")
        if split <= 0:
            reasons.append({"code": "malformed_edit", "line": line_number})
            break
        anchor, reading = line[:split], line[split + 2:]
        start = text.find(anchor, cursor)
        if start < 0 or not reading.strip():
            reasons.append({"code": "anchor_not_found" if start < 0 else "empty_reading", "line": line_number})
            break
        end = start + len(anchor)
        edit = dict(start=start, end=end, before=anchor, after=reading, line=line_number)
        edits.append(edit)
        if not any(character.isdecimal() for character in anchor):
            reasons.append({"code": "anchor_without_decimal_digit", "line": line_number, "start": start, "end": end})
        original_han = [character for character in anchor
                        if unicodedata.name(character, "").startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))]
        remaining = iter(reading)
        if not all(any(character == candidate for candidate in remaining) for character in original_han):
            reasons.append({"code": "original_han_not_preserved_in_order", "line": line_number,
                            "start": start, "end": end})
        overlaps = [[left, right] for left, right in spans if start < right and end > left]
        if overlaps:
            reasons.append({"code": "protected_span_overlap", "line": line_number,
                            "start": start, "end": end, "protected_spans": overlaps})
        parts.extend([text[cursor:start], reading])
        cursor = end
    if "".join(parts) + text[cursor:] != prediction:
        reasons.append({"code": "reconstructed_prediction_mismatch"})
    return edits, reasons


def _write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()


def _validate_component(output, transformed, transformed_hash, settings_hash, args, sources):
    child = output / "tn"
    state, meta = read_json(child / "run.json"), read_json(child / "predictions.meta.json")
    if state.get("status") != "complete" or meta.get("baseline") != "tn":
        raise ValueError("TN component did not complete with the requested identity")
    if meta.get("experiment") != "bilibili-reviewed-v1":
        raise ValueError("Unexpected historical TN component protocol")
    if meta.get("input_sha256") != transformed_hash or meta.get("settings_sha256") != settings_hash:
        raise ValueError("TN component input or settings provenance mismatch")
    if meta.get("revision") != args.revision or meta.get("model_revision") != args.revision:
        raise ValueError("TN component model revision mismatch")
    if meta.get("training") is not False or meta.get("external_api_cost_usd") != 0:
        raise ValueError("TN component must remain free local inference")
    for relative, digest in meta.get("source_sha256", {}).items():
        if sources.get(relative) != digest:
            raise ValueError("TN component source mismatch: " + relative)
    if meta.get("runner_sha256") != sources["scripts/run_zh_reviewed_baseline.py"]:
        raise ValueError("TN component runner hash mismatch")
    path = child / "predictions.jsonl"
    digest = hash_file(path)
    if meta.get("predictions_sha256") != digest or state.get("predictions_sha256") != digest:
        raise ValueError("TN component prediction hash mismatch")
    predictions = read_jsonl(path)
    if [row.get("id") for row in predictions] != [row["id"] for row in transformed]:
        raise ValueError("TN component must return every transformed input once, in order")
    if meta.get("case_count") != len(transformed):
        raise ValueError("TN component count mismatch")
    return predictions, meta


def run(args, *, executor=subprocess.run):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(schema_version=1, experiment="comment-policy-tn-v1", baseline="comment-policy-tn",
                 status="started", phase="preflight", started_at_utc=now(),
                 training=False, external_api_cost_usd=0)
    write_json(output / "run.json", state)
    try:
        verify_pilot(PROJECT)
        inputs_path, resources_path, settings_path = (Path(value).resolve()
                                                     for value in (args.inputs, args.resources, args.settings))
        inputs = validate_inputs(inputs_path)
        resources = validate_resources(read_json(resources_path))
        settings = validate_kernel_settings(settings_path)
        if not args.model_dir or not args.revision:
            raise ValueError("The pipeline requires an explicit local model directory and revision")
        policy = load_policy()
        source_paths = ("scripts/run_comment_pipeline.py", "src/readcue/comment_policy.py",
                        "scripts/run_zh_reviewed_baseline.py", "scripts/run_zh_baseline.py",
                        "src/readcue/artifacts.py", "src/readcue/baselines.py")
        sources = {relative: hash_file(PROJECT / relative) for relative in source_paths}
        watched = {str(path): hash_file(path) for path in (inputs_path, resources_path, settings_path)}
        meta = dict(schema_version=1, experiment=state["experiment"], baseline=state["baseline"],
                    started_at_utc=state["started_at_utc"], training=False, external_api_cost_usd=0,
                    input_sha256=watched[str(inputs_path)], resources_sha256=watched[str(resources_path)],
                    settings_sha256=watched[str(settings_path)], settings=settings,
                    resources_metadata=resources["metadata"], source_sha256=sources,
                    runner_sha256=sources[source_paths[0]], policy_source_sha256=sources[source_paths[1]],
                    case_count=len(inputs), inference_fields=["id", "text"],
                    pronunciation_dictionary_used=False, emote_resource_lists_used=True,
                    guard="sequential anchors; decimal digit required; original Han characters retained in order; protected spans preserved",
                    component_experiment_semantics="The child bilibili-reviewed-v1 label identifies the unchanged TN computation component, not this parent experiment or dataset.")
        state.update(phase="preprocessing", **meta)
        write_json(output / "run.json", state)
        rules_output = output / "rules_only"
        rules_output.mkdir()
        rule_rows, transformed = [], []
        for row in inputs:
            started = time.perf_counter()
            cleaned = policy.preprocess(row["text"], platform_emotes=set(resources["platform_emotes"]),
                                        unicode_emotes=set(resources["unicode_emotes"]))
            if not isinstance(cleaned, dict) or not isinstance(cleaned.get("text"), str):
                raise ValueError("Policy must return a text string")
            if not isinstance(cleaned.get("edits"), list) or not isinstance(cleaned.get("warnings"), list):
                raise ValueError("Policy must return edits and warnings lists")
            if not isinstance(cleaned.get("policy_version"), str) or not cleaned["policy_version"]:
                raise ValueError("Policy must identify its version")
            elapsed = (time.perf_counter() - started) * 1000
            result = dict(id=row["id"], prediction=cleaned["text"], raw_output=cleaned["text"],
                          status="ok" if cleaned["text"].strip() else "empty_after_policy",
                          latency_ms=elapsed, generated_tokens=None, policy_edits=cleaned["edits"],
                          policy_warnings=cleaned["warnings"], policy_version=cleaned["policy_version"])
            rule_rows.append(result)
            if result["status"] == "ok":
                transformed.append({"id": row["id"], "text": cleaned["text"]})
        versions = {row["policy_version"] for row in rule_rows}
        if len(versions) != 1:
            raise ValueError("Policy version changed during preprocessing")
        meta["policy_version"] = next(iter(versions))
        rules_path, transformed_path = rules_output / "predictions.jsonl", output / "transformed_inputs.jsonl"
        _write_jsonl(rules_path, rule_rows)
        _write_jsonl(transformed_path, transformed)
        meta.update(rules_predictions_sha256=hash_file(rules_path),
                    transformed_inputs_sha256=hash_file(transformed_path), transformed_case_count=len(transformed),
                    empty_after_policy=sum(row["status"] == "empty_after_policy" for row in rule_rows),
                    rules_inference_seconds=sum(row["latency_ms"] for row in rule_rows) / 1000)
        rules_meta = {**meta, "baseline": "comment-policy", "predictions_sha256": meta["rules_predictions_sha256"],
                      "failed_cases": meta["empty_after_policy"], "load_seconds": 0, "warmup_seconds": 0}
        write_json(rules_output / "predictions.meta.json", rules_meta)
        write_json(rules_output / "run.json", {**rules_meta, "status": "complete", "phase": "complete"})
        state.update(phase="tn_component", **meta)
        write_json(output / "run.json", state)
        predictions, child_meta = [], None
        if transformed:
            command = [sys.executable, str(PROJECT / "scripts/run_zh_reviewed_baseline.py"),
                       "--inputs", str(transformed_path), "--baseline", "tn", "--output", str(output / "tn"),
                       "--settings", str(settings_path), "--model-dir", str(Path(args.model_dir).resolve()),
                       "--revision", args.revision, "--device", args.device]
            state["command"] = command
            write_json(output / "run.json", state)
            with (output / "tn.stdout.log").open("w", encoding="utf-8") as stdout, (output / "tn.stderr.log").open("w", encoding="utf-8") as stderr:
                process = executor(command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=False)
            state["tn_returncode"] = process.returncode
            if process.returncode:
                raise RuntimeError("TN component failed; see tn.stderr.log and tn/run.json")
            predictions, child_meta = _validate_component(output, transformed, meta["transformed_inputs_sha256"],
                                                         meta["settings_sha256"], args, sources)
        else:
            (output / "tn").mkdir()
            write_json(output / "tn/run.json", dict(status="skipped_empty_input", case_count=0,
                                                     training=False, external_api_cost_usd=0))
        state.update(phase="guard_and_merge")
        write_json(output / "run.json", state)
        by_id, cleaned_by_id = {row["id"]: row for row in predictions}, {row["id"]: row["text"] for row in transformed}
        merged = []
        for rule in rule_rows:
            record = {**rule, "policy_latency_ms": rule["latency_ms"], "tn_status": "not_run_empty_after_policy",
                      "tn_prediction": None, "tn_edits": [], "tn_guard_reasons": [],
                      "tn_latency_ms": 0, "guard_latency_ms": 0, "input_tokens": None}
            if rule["id"] in by_id:
                prediction = by_id[rule["id"]]
                record.update(prediction=prediction.get("prediction", ""), raw_output=prediction.get("raw_output", ""),
                              status=prediction.get("status", "missing"), tn_status=prediction.get("status", "missing"),
                              tn_prediction=prediction.get("prediction", ""),
                              generated_tokens=prediction.get("generated_tokens"), input_tokens=prediction.get("input_tokens"),
                              tn_latency_ms=prediction.get("latency_ms", 0))
                if record["status"] == "ok":
                    started = time.perf_counter()
                    edits, reasons = guarded_tn_edits(cleaned_by_id[rule["id"]], record["raw_output"],
                                                      record["prediction"], policy.protected_spans)
                    record.update(tn_edits=edits, tn_guard_reasons=reasons,
                                  guard_latency_ms=(time.perf_counter() - started) * 1000)
                    if reasons:
                        record.update(status="tn_guard_rejected", prediction=cleaned_by_id[rule["id"]])
                record["latency_ms"] = record["policy_latency_ms"] + record["tn_latency_ms"] + record["guard_latency_ms"]
            merged.append(record)
        final_path = output / "predictions.jsonl"
        _write_jsonl(final_path, merged)
        meta.update(predictions_sha256=hash_file(final_path),
                    failed_cases=sum(row["status"] != "ok" for row in merged),
                    guard_rejected_cases=sum(row["status"] == "tn_guard_rejected" for row in merged),
                    truncated_cases=sum(row["tn_status"] == "truncated" for row in merged),
                    tn_component=child_meta, tn_skipped_no_nonempty_inputs=not transformed,
                    load_seconds=(child_meta or {}).get("load_seconds", 0),
                    warmup_seconds=(child_meta or {}).get("warmup_seconds", 0),
                    inference_seconds=sum(row["latency_ms"] for row in merged) / 1000,
                    finished_at_utc=now())
        for name, digest in watched.items():
            if hash_file(name) != digest:
                raise ValueError("Inputs, resources or settings changed during execution: " + name)
        if hash_file(transformed_path) != meta["transformed_inputs_sha256"]:
            raise ValueError("Transformed inputs changed during execution")
        if any(hash_file(PROJECT / name) != digest for name, digest in sources.items()):
            raise ValueError("Pipeline or dependency source changed during execution")
        write_json(output / "predictions.meta.json", meta)
        state.update(status="complete", phase="complete", **meta)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state["finished_at_utc"] = now()
        write_json(output / "run.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
