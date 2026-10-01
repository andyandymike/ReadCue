"""Rules-only audit correction; optionally validate already-generated TN edits.

Never runs a model, downloads assets, scores references or overwrites a run.
"""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from readcue import audit_policy_adapter as policy
from readcue.artifacts import hash_file, read_json, read_jsonl, write_json
from readcue.baselines import validate_inputs
from run_comment_pipeline import guarded_tn_edits, validate_resources


def run(args):
    paths = {key: Path(getattr(args, key)).resolve() for key in
             ("inputs", "resources", "semantic_resources", "lexicon", "context_resources")}
    watched = {key: hash_file(path) for key, path in paths.items()}
    inputs = validate_inputs(paths["inputs"])
    resources = {key: read_json(paths[key]) for key in paths if key != "inputs"}
    validate_resources(resources["resources"])
    candidates = None
    if args.tn_candidates:
        paths["tn_candidates"] = Path(args.tn_candidates).resolve()
        watched["tn_candidates"] = hash_file(paths["tn_candidates"])
        candidates = read_jsonl(paths["tn_candidates"])
        if [row.get("id") for row in candidates] != [row["id"] for row in inputs]:
            raise ValueError("TN candidates must match input ids once and in order")
        if any(not all(key in row for key in ("raw_output", "prediction", "status")) for row in candidates):
            raise ValueError("TN candidates require raw_output, prediction and status")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = ["src/readcue/" + name + ".py" for name in (
        "audit_policy_adapter", "comment_policy", "comment_policy_v2", "reading_policy",
        "context_reading_policy_v3", "semantic_emoji_policy", "artifacts", "baselines")]
    source_files += ["scripts/run_audited_rules.py", "scripts/run_comment_pipeline.py"]
    hashes = {name: hash_file(PROJECT / name) for name in source_files}
    state = dict(schema_version=1, policy_version=policy.POLICY_VERSION,
        historical_base=policy.HISTORICAL_BASE, historical_results_replaced=False,
        model_executed=False, model_benefit_measured=False, training=False, external_api_cost_usd=0,
        status="started", inputs_sha256=watched, source_sha256=hashes,
        candidate_audit_only=candidates is not None, case_count=len(inputs))
    write_json(output / "run.json", state)
    try:
        with (output / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for index, item in enumerate(inputs):
                result = policy.preprocess(item["text"], **resources)
                record = dict(id=item["id"], prediction=result["text"], rules_prediction=result["text"],
                              status="rules_only", **{k: v for k, v in result.items() if k != "text"})
                if candidates is not None:
                    candidate = candidates[index]
                    record["tn_candidate"] = candidate
                    if candidate["status"] == "ok":
                        edits, reasons = guarded_tn_edits(result["text"], candidate["raw_output"],
                            candidate["prediction"], lambda _: result["protected_spans"])
                        record.update(tn_edits=edits, tn_guard_reasons=reasons,
                                      status="candidate_rejected" if reasons else "candidate_guard_accepted")
                        if not reasons:
                            record["prediction"] = candidate["prediction"]
                    else:
                        record.update(status="candidate_not_ok", tn_guard_reasons=[{"code": "candidate_status_not_ok"}])
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if any(hash_file(path) != watched[key] for key, path in paths.items()):
            raise ValueError("An audit input changed during processing")
        if any(hash_file(PROJECT / name) != digest for name, digest in hashes.items()):
            raise ValueError("An audit source changed during processing")
        state.update(status="complete", predictions_sha256=hash_file(output / "predictions.jsonl"))
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / "run.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "resources", "semantic-resources", "lexicon", "context-resources", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tn-candidates", help="Existing id/raw_output/prediction/status JSONL; no model is executed")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))
