"""Score and archive the existing pilot without changing its frozen semantics."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import statistics

from .artifacts import BASELINE_REPOS, hash_file, project_path, read_json, read_jsonl, verify_pilot, write_json

BASELINES = ("identity", "wetext", "tn", "qwen")


def _run_files(run, name):
    legacy = run / f"{name}.jsonl"
    wrapped = run / name / "predictions.jsonl"
    if legacy.exists() and wrapped.exists():
        raise ValueError("Ambiguous legacy and wrapped results: " + name)
    if wrapped.exists():
        state = read_json(run / name / "run.json")
        if state.get("status") != "complete" or state.get("baseline") != name:
            raise ValueError("Wrapped run is incomplete or mislabeled: " + name)
        if state.get("predictions_sha256") != hash_file(wrapped):
            raise ValueError("Wrapped prediction hash mismatch: " + name)
        return wrapped, run / name / "predictions.meta.json"
    return legacy, run / f"{name}.meta.json"


def frozen_scorer(project):
    freeze = verify_pilot(project)
    path = project_path(project, "scripts/score_zh_pilot.py")
    spec = importlib.util.spec_from_file_location("readcue_pilot_v1_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score, freeze


def score_files(project, cases_path, predictions_path):
    scorer, _ = frozen_scorer(project)
    result = scorer(read_jsonl(cases_path), read_jsonl(predictions_path))
    result["cases_sha256"] = hash_file(cases_path)
    result["predictions_sha256"] = hash_file(predictions_path)
    return result


def validate_run_metadata(name, predictions, metadata, freeze):
    """A filename is not evidence of which baseline or revision produced it."""
    if metadata.get("baseline") != name:
        raise ValueError("Baseline label mismatch: " + name)
    if metadata.get("input_sha256") != freeze["files_sha256"]["eval/zh_reading_pilot_v1/inputs.jsonl"]:
        raise ValueError("Input hash does not match the frozen pilot: " + name)
    runner_hash = freeze["files_sha256"]["scripts/run_zh_baseline.py"]
    allowed = {runner_hash}
    if name == "identity":
        allowed.add(freeze["pre_rule_inference_config_review"]["previous_runner_sha256"])
    elif name == "wetext":
        allowed.add(freeze["neural_loading_compatibility_revision"]["previous_runner_sha256"])
    if metadata.get("runner_sha256") not in allowed:
        raise ValueError("Unrecorded runner revision for baseline: " + name)
    if name in BASELINE_REPOS:
        expected = next(asset for asset in freeze["baseline_assets"] if asset["repo"] == BASELINE_REPOS[name])
        if metadata.get("revision") != expected["revision"]:
            raise ValueError("Model revision mismatch: " + name)
    elif metadata.get("revision") is not None:
        raise ValueError("Unexpected model revision for non-model baseline: " + name)
    if metadata.get("external_api_cost_usd") != 0 or metadata.get("training") is not False:
        raise ValueError("Only this original-data, free inference pilot can be archived here")
    if hash_file(predictions) != metadata.get("predictions_sha256"):
        raise ValueError("Prediction hash mismatch: " + name)


def archive_pilot(project, run, output):
    """Validate all four identities and results before creating a new archive."""
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new archive directory; preserve prior reports")
    scorer, freeze = frozen_scorer(project)
    case_path = project_path(project, "eval/zh_reading_pilot_v1/cases.jsonl")
    cases, prepared = read_jsonl(case_path), {}
    for name in BASELINES:
        path, metadata_path = _run_files(run, name)
        predictions = read_jsonl(path)
        metadata = read_json(metadata_path)
        validate_run_metadata(name, path, metadata, freeze)
        if len(predictions) != len(cases) or metadata.get("case_count") != len(cases):
            raise ValueError("Incomplete prediction count: " + name)
        result = scorer(cases, predictions)
        result.update(cases_sha256=hash_file(case_path), predictions_sha256=hash_file(path))
        prepared[name] = predictions, metadata, result, (path, metadata_path)
    output.mkdir(parents=True)
    archive_state = {"status": "writing", "experiment": "zh_reading_pilot_v1"}
    write_json(output / "archive.json", archive_state)
    try:
        summary = {}
        for name, (predictions, metadata, result, paths) in prepared.items():
            write_json(output / f"{name}.score.json", result)
            for source, suffix in zip(paths, ("jsonl", "meta.json")):
                shutil.copyfile(source, output / f"{name}.{suffix}")
            timings = [record["latency_ms"] for record in predictions]
            summary[name] = dict(overall=result["overall"], by_track=result["by_track"], families=result["families"],
                latency_ms_median=statistics.median(timings), latency_ms_max=max(timings),
                total_inference_seconds=sum(sorted(timings)) / 1000, load_seconds=metadata["load_seconds"])
            lines = [f"# {name}: unmatched cases", "", "Exact-match failures are not automatically semantic/pronunciation errors.", ""]
            for row in result["cases"]:
                if not row["surface_correct"]:
                    lines.extend([f"## {row['id']} ({row['track']}, {row['status']})", "", f"Input: {row['text']}", "",
                        f"Output: {row['prediction']}", "", "Accepted references: " + json.dumps(row["acceptable_outputs"], ensure_ascii=False), ""])
            (output / f"{name}.unmatched.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
        write_json(output / "summary.json", summary)
        with (output / "comparison.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["id", "track", "text", "acceptable_outputs", *BASELINES])
            by_id = {name: {row["id"]: row["prediction"] for row in prepared[name][0]} for name in BASELINES}
            for case in cases:
                writer.writerow([case["id"], case["track"], case["text"], json.dumps(case["acceptable_outputs"], ensure_ascii=False),
                                 *[by_id[name][case["id"]] for name in BASELINES]])
        archive_state["status"] = "complete"
    except BaseException as error:
        archive_state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / "archive.json", archive_state)
    return summary
