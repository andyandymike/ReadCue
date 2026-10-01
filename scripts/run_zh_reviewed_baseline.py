"""Run reviewed-comment baselines without changing the frozen 72-case pilot."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from readcue.artifacts import (BASELINE_REPOS, hash_file, read_json, verify_model,
                              verify_pilot, write_json)
from readcue.baselines import validate_inputs


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_settings(settings):
    keys = {"schema_version", "experiment", "system_prompt", "max_new_tokens",
            "seed", "temperature", "top_p", "top_k"}
    if not isinstance(settings, dict) or set(settings) != keys:
        raise ValueError("Settings must contain exactly the reviewed-protocol fields")
    if settings["schema_version"] != 1 or settings["experiment"] != "bilibili-reviewed-v1":
        raise ValueError("Unknown reviewed experiment or settings schema")
    if not isinstance(settings["system_prompt"], str) or not settings["system_prompt"].strip():
        raise ValueError("A frozen public system prompt is required")
    for field in ("max_new_tokens", "seed", "top_k"):
        if type(settings[field]) is not int or settings[field] < 1:
            raise ValueError(field + " must be a positive integer")
    for field in ("temperature", "top_p"):
        value = settings[field]
        if type(value) not in (int, float) or not 0 < value <= 1:
            raise ValueError(field + " must be in (0, 1]")
    return settings


def load_legacy_helpers():
    path = PROJECT / "scripts/run_zh_baseline.py"
    spec = importlib.util.spec_from_file_location("readcue_frozen_baseline_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(schema_version=1, status="started", phase="preflight",
                 experiment="bilibili-reviewed-v1", baseline=args.baseline,
                 started_at_utc=now(), training=False, external_api_cost_usd=0)
    write_json(output / "run.json", state)
    try:
        freeze = verify_pilot(PROJECT)
        inputs_path, settings_path = Path(args.inputs).resolve(), Path(args.settings).resolve()
        inputs = validate_inputs(inputs_path)
        settings = validate_settings(read_json(settings_path))
        source_paths = ("scripts/run_zh_reviewed_baseline.py", "scripts/run_zh_baseline.py",
                        "src/readcue/artifacts.py", "src/readcue/baselines.py")
        sources = {name: hash_file(PROJECT / name) for name in source_paths}
        meta = dict(schema_version=1, experiment=settings["experiment"], baseline=args.baseline,
                    model_dir=str(Path(args.model_dir).resolve()) if args.model_dir else None,
                    revision=args.revision, input_sha256=hash_file(inputs_path),
                    settings_sha256=hash_file(settings_path), settings=settings,
                    runner_sha256=sources[source_paths[0]], legacy_runner_sha256=sources[source_paths[1]],
                    artifacts_sha256=sources[source_paths[2]], baselines_sha256=sources[source_paths[3]],
                    source_sha256=sources, started_at_utc=state["started_at_utc"],
                    platform=platform.platform(), python=platform.python_version(),
                    external_api_cost_usd=0, training=False, case_count=len(inputs),
                    inference_fields=["id", "text"], input_transform="none",
                    batch_size=1, max_new_tokens=settings["max_new_tokens"], seed=settings["seed"])
        if args.baseline in BASELINE_REPOS:
            if not args.model_dir or not args.revision:
                raise ValueError("Neural baselines require a local model directory and resolved revision")
            trusted = next((asset for asset in freeze["baseline_assets"]
                            if asset["repo"] == BASELINE_REPOS[args.baseline]
                            and asset["revision"] == args.revision), None)
            if trusted is None:
                raise ValueError("Model revision was not audited in the original local pilot")
            manifest = verify_model(Path(args.model_dir).resolve(), BASELINE_REPOS[args.baseline],
                                    args.revision, trusted=trusted)
            meta.update(model_repo=manifest["repo"], model_revision=manifest["revision"],
                        model_manifest_sha256=hash_file(Path(args.model_dir) / "readcue_manifest.json"))
        elif args.model_dir is not None or args.revision is not None:
            raise ValueError("Non-model baselines must not carry model or revision labels")
        state.update(phase="loading", **meta)
        write_json(output / "run.json", state)
        load_start = time.perf_counter()
        if args.baseline == "wetext":
            from tn.chinese.normalizer import Normalizer
            normalizer = Normalizer(remove_erhua=False, remove_interjections=False,
                                    traditional_to_simple=False, overwrite_cache=True,
                                    cache_dir=str(output / "wetext-cache"))
            meta.update(wetext_version=importlib.metadata.version("WeTextProcessing"),
                        remove_erhua=False, remove_interjections=False, traditional_to_simple=False)
            if meta["wetext_version"] != "1.2.0":
                raise ValueError("The reviewed experiment requires WeTextProcessing 1.2.0")
        elif args.baseline in BASELINE_REPOS:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from transformers import AutoModelForCausalLM, set_seed
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device is unavailable")
            legacy = load_legacy_helpers()
            set_seed(settings["seed"])
            tokenizer, config, compatibility = legacy.load_local_tokenizer_and_config(args.model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir, local_files_only=True, config=config, trust_remote_code=False,
                use_safetensors=True, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
                attn_implementation="sdpa").to(args.device).eval()
            meta.update(torch=torch.__version__, transformers=importlib.metadata.version("transformers"),
                        device=args.device, gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                        dtype=str(model.dtype), loading_compatibility=compatibility,
                        enable_thinking=False, local_files_only=True, use_safetensors=True,
                        trust_remote_code=False, use_cache=True,
                        decoding="greedy" if args.baseline == "tn" else "sample",
                        prompt=settings["system_prompt"] if args.baseline == "qwen"
                        else "<|fim_prefix|>{text}<|fim_suffix|>")
            if args.baseline == "qwen":
                meta.update({field: settings[field] for field in ("temperature", "top_p", "top_k")})
        meta["load_seconds"] = time.perf_counter() - load_start

        def infer(text, index):
            if args.baseline == "identity":
                return text, text, "ok", None, None
            if args.baseline == "wetext":
                result = normalizer.normalize(text)
                return result, result, "ok" if result.strip() else "empty_output", None, None
            if args.baseline == "tn":
                prompt = "<|fim_prefix|>" + text + "<|fim_suffix|>"
            else:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": settings["system_prompt"]},
                     {"role": "user", "content": text}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
            set_seed(settings["seed"] + index)
            generation = dict(max_new_tokens=settings["max_new_tokens"],
                              pad_token_id=tokenizer.eos_token_id, use_cache=True)
            if args.baseline == "tn":
                generation.update(do_sample=False, eos_token_id=151643)
            else:
                generation.update(do_sample=True, **{field: settings[field]
                                  for field in ("temperature", "top_p", "top_k")})
            with torch.inference_mode():
                tokens = model.generate(**encoded, **generation)[0, encoded.input_ids.shape[1]:]
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            eos = generation.get("eos_token_id", model.generation_config.eos_token_id)
            eos = [eos] if isinstance(eos, int) else eos
            ended = bool(len(tokens)) and int(tokens[-1]) in (eos or [])
            counts = len(tokens), int(encoded.input_ids.shape[1])
            if not ended:
                return "", raw, "truncated", *counts
            if args.baseline == "tn":
                try:
                    result = legacy.parse_edits(text, raw)
                except ValueError as error:
                    return text, raw, "parse_error: " + str(error), *counts
            else:
                result = raw.strip()
            return result, raw, "ok" if result.strip() else "empty_output", *counts

        state.update(phase="warmup", **meta)
        write_json(output / "run.json", state)
        warmup_start = time.perf_counter()
        warmup = infer("今天温度为25℃。", -1)
        meta.update(warmup_seconds=time.perf_counter() - warmup_start,
                    warmup_status=warmup[2], warmup_text="今天温度为25℃。")
        state.update(phase="inference", **meta)
        write_json(output / "run.json", state)
        failed, truncated, total_tokens = 0, 0, 0
        inference_start = time.perf_counter()
        predictions_path = output / "predictions.jsonl"
        with predictions_path.open("x", encoding="utf-8", newline="\n") as stream:
            for index, row in enumerate(inputs):
                started = time.perf_counter()
                try:
                    prediction, raw, status, token_count, input_tokens = infer(row["text"], index)
                except Exception as error:
                    prediction, raw, status = "", "", type(error).__name__ + ": " + str(error)
                    token_count, input_tokens = None, None
                elapsed = (time.perf_counter() - started) * 1000
                failed += status != "ok"
                truncated += status == "truncated"
                total_tokens += token_count or 0
                record = dict(id=row["id"], prediction=prediction, raw_output=raw, status=status,
                              latency_ms=elapsed, generated_tokens=token_count, input_tokens=input_tokens)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{args.baseline}: {index + 1}/{len(inputs)} {status} {elapsed:.0f}ms", flush=True)
        meta.update(finished_at_utc=now(), inference_seconds=time.perf_counter() - inference_start,
                    predictions_sha256=hash_file(predictions_path), failed_cases=failed,
                    truncated_cases=truncated, generated_tokens_total=total_tokens)
        # A change during execution invalidates the run instead of relabeling its provenance.
        if hash_file(inputs_path) != meta["input_sha256"] or hash_file(settings_path) != meta["settings_sha256"]:
            raise ValueError("Inference inputs or settings changed during execution")
        if any(hash_file(PROJECT / name) != digest for name, digest in sources.items()):
            raise ValueError("Runner or helper source changed during execution")
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
    parser.add_argument("--baseline", choices=["identity", "wetext", "tn", "qwen"], required=True)
    parser.add_argument("--output", required=True, help="A new run directory")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
