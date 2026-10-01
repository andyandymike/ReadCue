"""Versioned comment, lexical/context reading and unchanged offline TN composition."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from readcue.artifacts import hash_file, read_json, write_json, verify_pilot
from readcue.baselines import validate_inputs
from readcue import comment_policy, reading_policy, context_reading_policy
import run_comment_pipeline as component


def now():
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()


def checked_policy_result(source, result, name):
    """Check stage coordinates and replayability before forwarding its text."""
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise ValueError(name + " must return text")
    if not isinstance(result.get("edits"), list) or not isinstance(result.get("warnings"), list):
        raise ValueError(name + " must return edit and warning lists")
    if not isinstance(result.get("policy_version"), str) or not result["policy_version"]:
        raise ValueError(name + " must identify its version")
    cursor, parts = 0, []
    for edit in result["edits"]:
        start, end = edit.get("start"), edit.get("end")
        if type(start) is not int or type(end) is not int or not cursor <= start < end <= len(source):
            raise ValueError(name + " returned invalid or overlapping edit coordinates")
        if edit.get("original") != source[start:end] or not isinstance(edit.get("replacement"), str):
            raise ValueError(name + " edit text does not match its source coordinates")
        parts.extend([source[cursor:start], edit["replacement"]])
        cursor = end
    if "".join(parts) + source[cursor:] != result["text"]:
        raise ValueError(name + " edits do not reconstruct its output")
    return result


def context_output_protection(result):
    """Retain declared protection and every generated context-reading span."""
    text = result["text"]
    declared = result.get("protected_spans")
    if not isinstance(declared, list):
        raise ValueError("Context policy must provide output-coordinate protected_spans")
    spans = list(reading_policy.protected_spans(text))
    for span in declared:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError("Invalid context protected span")
        left, right = span
        if type(left) is not int or type(right) is not int or not 0 <= left < right <= len(text):
            raise ValueError("Context protected span lies outside its output")
        spans.append((left, right))
    shift = 0
    for edit in result["edits"]:
        left = edit["start"] + shift
        right = left + len(edit["replacement"])
        if right > left:
            spans.append((left, right))
        shift += len(edit["replacement"]) - (edit["end"] - edit["start"])
    merged = []
    for left, right in sorted(spans):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def run(args, *, executor=subprocess.run):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(schema_version=1, experiment="context-reading-v3", baseline="comment-reading-context-tn",
                 status="started", phase="preflight", started_at_utc=now(), training=False, external_api_cost_usd=0)
    write_json(output / "run.json", state)
    try:
        verify_pilot(PROJECT)
        paths = {name: Path(getattr(args, name)).resolve()
                 for name in ("inputs", "resources", "lexicon", "context_resources", "settings")}
        watched = {name: hash_file(path) for name, path in paths.items()}
        inputs = validate_inputs(paths["inputs"])
        resources = component.validate_resources(read_json(paths["resources"]))
        lexicon, context_resources = read_json(paths["lexicon"]), read_json(paths["context_resources"])
        reading_policy.validate_lexicon(lexicon)
        context_reading_policy.validate_resources(context_resources)
        settings = component.validate_kernel_settings(paths["settings"])
        if not args.model_dir or not args.revision:
            raise ValueError("Explicit local model directory and revision required")
        names = ("scripts/run_context_reading_pipeline.py", "src/readcue/context_reading_policy.py",
                 "src/readcue/reading_policy.py", "src/readcue/comment_policy.py", "scripts/run_comment_pipeline.py",
                 "scripts/run_zh_reviewed_baseline.py", "scripts/run_zh_baseline.py",
                 "src/readcue/artifacts.py", "src/readcue/baselines.py")
        sources = {name: hash_file(PROJECT / name) for name in names}
        meta = dict(schema_version=1, experiment=state["experiment"], baseline=state["baseline"],
                    started_at_utc=state["started_at_utc"], training=False, external_api_cost_usd=0,
                    input_sha256=watched["inputs"], resources_sha256=watched["resources"],
                    lexicon_sha256=watched["lexicon"], context_resources_sha256=watched["context_resources"],
                    settings_sha256=watched["settings"], settings=settings, source_sha256=sources,
                    runner_sha256=sources[names[0]], case_count=len(inputs), inference_fields=["id", "text"],
                    pronunciation_dictionary_used=True, emote_resource_lists_used=True,
                    lexicon_entry_count=len(lexicon["entries"]), lexicon_provenance=lexicon.get("provenance"),
                    lexicon_policy_version=lexicon["policy_version"],
                    context_resources_policy_version=context_resources["policy_version"],
                    context_resource_provenance=context_resources.get("provenance", context_resources.get("metadata")),
                    guard="frozen sequential/digit/Han guard plus reading literals, context-declared protection and all generated context spans",
                    edit_coordinates="comment_edits index original text; reading_edits index comment_text; context_edits index reading_text; context_protected_spans and tn_edits index rules_prediction",
                    warning_coordinates="comment_warnings index original text; reading_warnings index comment_text; context_warnings index reading_text",
                    component_experiment_semantics="bilibili-reviewed-v1 identifies the unchanged TN computation component, not this parent experiment or dataset")
        state.update(phase="preprocessing", **meta)
        write_json(output / "run.json", state)
        rows, transformed = [], []
        for item in inputs:
            started = time.perf_counter()
            comment = checked_policy_result(item["text"], comment_policy.preprocess(item["text"],
                platform_emotes=set(resources["platform_emotes"]), unicode_emotes=set(resources["unicode_emotes"])), "comment")
            reading = checked_policy_result(comment["text"], reading_policy.preprocess(comment["text"], lexicon=lexicon), "reading")
            context = checked_policy_result(reading["text"], context_reading_policy.preprocess(reading["text"], resources=context_resources), "context")
            protection = context_output_protection(context)
            record = dict(id=item["id"], prediction=context["text"], raw_output=context["text"],
                          status="ok" if context["text"].strip() else "empty_after_policy",
                          comment_text=comment["text"], reading_text=reading["text"],
                          comment_edits=comment["edits"], reading_edits=reading["edits"], context_edits=context["edits"],
                          comment_warnings=comment["warnings"], reading_warnings=reading["warnings"], context_warnings=context["warnings"],
                          comment_policy_version=comment["policy_version"], reading_policy_version=reading["policy_version"],
                          context_policy_version=context["policy_version"], policy_version=context["policy_version"],
                          context_protected_spans=protection, latency_ms=(time.perf_counter() - started) * 1000,
                          generated_tokens=None, input_tokens=None)
            rows.append(record)
            if record["status"] == "ok":
                transformed.append(dict(id=item["id"], text=record["prediction"]))
        for stage in ("comment", "reading", "context"):
            versions = {row[stage + "_policy_version"] for row in rows}
            if len(versions) != 1:
                raise ValueError(stage + " policy version changed during preprocessing")
            meta[stage + "_policy_version"] = next(iter(versions))
        rules = output / "rules_only"
        rules.mkdir()
        write_jsonl(rules / "predictions.jsonl", rows)
        transformed_path = output / "transformed_inputs.jsonl"
        write_jsonl(transformed_path, transformed)
        meta.update(rules_predictions_sha256=hash_file(rules / "predictions.jsonl"),
                    transformed_inputs_sha256=hash_file(transformed_path), transformed_case_count=len(transformed),
                    empty_after_policy=len(rows) - len(transformed),
                    rules_inference_seconds=sum(row["latency_ms"] for row in rows) / 1000)
        rules_meta = {**meta, "baseline": "comment-reading-context-rules",
                      "predictions_sha256": meta["rules_predictions_sha256"], "failed_cases": meta["empty_after_policy"],
                      "load_seconds": 0, "warmup_seconds": 0}
        write_json(rules / "predictions.meta.json", rules_meta)
        write_json(rules / "run.json", {**rules_meta, "status": "complete", "phase": "complete"})
        state.update(phase="tn_component", **meta)
        write_json(output / "run.json", state)
        predictions, child_meta = [], None
        if transformed:
            command = [sys.executable, str(PROJECT / "scripts/run_zh_reviewed_baseline.py"),
                       "--inputs", str(transformed_path), "--baseline", "tn", "--output", str(output / "tn"),
                       "--settings", str(paths["settings"]), "--model-dir", str(Path(args.model_dir).resolve()),
                       "--revision", args.revision, "--device", args.device]
            state["command"] = command
            write_json(output / "run.json", state)
            with (output / "tn.stdout.log").open("x", encoding="utf-8") as stdout, (output / "tn.stderr.log").open("x", encoding="utf-8") as stderr:
                process = executor(command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=False)
            state["tn_returncode"] = process.returncode
            if process.returncode:
                raise RuntimeError("TN component failed; see logs and tn/run.json")
            predictions, child_meta = component._validate_component(output, transformed,
                meta["transformed_inputs_sha256"], meta["settings_sha256"], args, sources)
        else:
            (output / "tn").mkdir()
            write_json(output / "tn/run.json", dict(status="skipped_empty_input", case_count=0,
                                                     training=False, external_api_cost_usd=0))
        state["phase"] = "guard_and_merge"
        write_json(output / "run.json", state)
        by_id, merged = {row["id"]: row for row in predictions}, []
        for row in rows:
            record = {**row, "rules_prediction": row["prediction"], "policy_latency_ms": row["latency_ms"],
                      "tn_status": "not_run_empty_after_policy", "tn_prediction": None,
                      "tn_edits": [], "tn_guard_reasons": [], "tn_latency_ms": 0, "guard_latency_ms": 0}
            if row["id"] in by_id:
                pred = by_id[row["id"]]
                record.update(prediction=pred.get("prediction", ""), raw_output=pred.get("raw_output", ""),
                              status=pred.get("status", "missing"), tn_status=pred.get("status", "missing"),
                              tn_prediction=pred.get("prediction", ""), generated_tokens=pred.get("generated_tokens"),
                              input_tokens=pred.get("input_tokens"), tn_latency_ms=pred.get("latency_ms", 0))
                if record["status"] == "ok":
                    started = time.perf_counter()
                    edits, reasons = component.guarded_tn_edits(row["prediction"], record["raw_output"],
                        record["prediction"], lambda text: row["context_protected_spans"])
                    record.update(tn_edits=edits, tn_guard_reasons=reasons,
                                  guard_latency_ms=(time.perf_counter() - started) * 1000)
                    if reasons:
                        record.update(status="tn_guard_rejected", prediction=row["prediction"])
                record["latency_ms"] = record["policy_latency_ms"] + record["tn_latency_ms"] + record["guard_latency_ms"]
            merged.append(record)
        write_jsonl(output / "predictions.jsonl", merged)
        meta.update(predictions_sha256=hash_file(output / "predictions.jsonl"),
                    failed_cases=sum(row["status"] != "ok" for row in merged),
                    guard_rejected_cases=sum(row["status"] == "tn_guard_rejected" for row in merged),
                    truncated_cases=sum(row["tn_status"] == "truncated" for row in merged),
                    tn_component=child_meta, tn_skipped_no_nonempty_inputs=not transformed,
                    load_seconds=(child_meta or {}).get("load_seconds", 0),
                    warmup_seconds=(child_meta or {}).get("warmup_seconds", 0),
                    inference_seconds=sum(row["latency_ms"] for row in merged) / 1000)
        for name, path in paths.items():
            if hash_file(path) != watched[name]:
                raise ValueError("Frozen input/resource/settings changed: " + name)
        if hash_file(transformed_path) != meta["transformed_inputs_sha256"]:
            raise ValueError("Transformed inputs changed")
        if any(hash_file(PROJECT / name) != digest for name, digest in sources.items()):
            raise ValueError("Inference source changed")
        meta["finished_at_utc"] = now()
        write_json(output / "predictions.meta.json", meta)
        state.update(status="complete", phase="complete", **meta)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state["finished_at_utc"] = now()
        write_json(output / "run.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "resources", "lexicon", "context-resources", "settings", "model-dir", "revision", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run(parser.parse_args())
