"""Build a fresh, offline v2 adapter candidate; never publish or modify a run.

Only explicitly allowlisted files are copied. Inference helpers are extracted
verbatim from the source snapshot bound to the selected training run.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat


BASE_REPO = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
BASE_LICENSE_SHA256 = "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
ADAPTER_SHA256 = "27ab01f82e6811cd48ebb846cdf31c1d9adad746aa0370ec6ae8dd77f50b0f75"
SOURCE_PATHS = ("scripts/run_zh_baseline.py", "src/readcue/model_experiment.py",
                "src/readcue/inference_contract.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_relative(root, relative):
    """Reject escape paths and links before opening an enumerated artifact."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe relative artifact path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(p in ("", ".", "..") for p in relative.split("/")):
        raise ValueError("Unsafe relative artifact path")
    root = Path(root).absolute()
    current = root
    for part in (None, *parts.parts):
        if part is not None:
            current = current / part
        if current.is_symlink() or (current.exists() and getattr(current.lstat(), "st_file_attributes", 0)
                                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError("Artifact paths cannot contain links or reparse points")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escaped root")
    return current


def extract_functions(path, names):
    source = Path(path).read_text(encoding="utf-8")
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    if set(names) - set(functions):
        raise ValueError("Missing frozen function: " + ", ".join(sorted(set(names) - set(functions))))
    return "\n\n\n".join(functions[name] for name in names) + "\n"


def verify_bundle(bundle, base_dir=None):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("base") != {
            "repo": BASE_REPO, "revision": BASE_REVISION}:
        raise ValueError("Unexpected bundle/base identity")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Empty bundle inventory")
    for relative, expected in entries.items():
        path = checked_relative(bundle, relative)
        if (not path.is_file() or path.stat().st_size != expected["bytes"]
                or sha256(path) != expected["sha256"]):
            raise ValueError("Bundle file hash/size mismatch: " + relative)
    actual = set()
    for path in bundle.rglob("*"):
        relative = path.relative_to(bundle).as_posix()
        if relative == "checks" or relative.startswith("checks/"):
            continue
        checked_relative(bundle, relative)
        if path.is_file() and relative != "bundle-manifest.json":
            actual.add(relative)
    if actual != set(entries):
        raise ValueError("Unexpected or missing bundle payload files")
    if base_dir is not None:
        base_manifest = json.loads((bundle / "provenance/base-model-manifest.json").read_text(encoding="utf-8"))
        verify_base(base_dir, base_manifest)
    return manifest


def verify_base(base_dir, manifest):
    if manifest.get("repo") != BASE_REPO or manifest.get("revision") != BASE_REVISION:
        raise ValueError("Base model revision mismatch")
    base_dir = Path(base_dir)
    files = manifest["files"]
    names = {item["name"] for item in files}
    if len(names) != len(files) or not {"LICENSE", "model.safetensors", "tokenizer.json",
                                      "config.json", "generation_config.json"}.issubset(names):
        raise ValueError("Invalid base inventory")
    for entry in files:
        path = checked_relative(base_dir, entry["name"])
        if (not path.is_file() or path.stat().st_size != entry["bytes"]
                or sha256(path) != entry["sha256"]):
            raise ValueError("Base file hash/size mismatch: " + entry["name"])
    actual = {path.name for path in base_dir.iterdir()}
    if actual - names - {"readcue_manifest.json"}:
        raise ValueError("Unexpected base directory assets")
    for name in actual:
        checked_relative(base_dir, name)
    if sha256(base_dir / "LICENSE") != BASE_LICENSE_SHA256:
        raise ValueError("Base license identity mismatch")


PORTABLE_GLUE = r'''
def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_model(base_dir, bundle, config, device):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), config=config,
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(model, str(bundle / "adapter"), is_trainable=False,
                                      local_files_only=True, use_safetensors=True)
    return model.eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline local ReadCue v2 research candidate")
    parser.add_argument("command", choices=("check", "predict"))
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--load-model", action="store_true", help="check only: load weights on CPU, no generation")
    args = parser.parse_args(argv)
    bundle = Path(__file__).resolve().parent
    if args.command == "check":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.load_model:
        parser.error("--load-model only applies to check")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError("Result directory must be new: " + str(output))
    for ancestor in (output, *output.parents):
        if ancestor.is_symlink() or (ancestor.exists() and getattr(ancestor.lstat(), "st_file_attributes", 0)
                                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError("Result path cannot contain links or reparse points")
    output = output.resolve()
    if output.is_relative_to(bundle) and not output.is_relative_to(bundle / "checks"):
        raise ValueError("Results inside the bundle must be under checks/")
    base_dir = args.base_dir.resolve()
    if output.resolve().is_relative_to(base_dir) or base_dir.is_relative_to(output.resolve()):
        raise ValueError("Results must be separate from base assets")
    manifest = verify_bundle(bundle, base_dir)
    task = strict_json((bundle / "configs/model_task_v1.json").read_bytes())
    input_path = args.inputs or bundle / "example-inputs.jsonl"
    rows = parse_inputs(input_path.read_text(encoding="utf-8"))
    dependencies = verify_runtime_dependencies(bundle / "requirements-model-v1.lock")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"schema_version": 1, "command": args.command, "status": "running",
        "bundle_manifest_sha256": sha256(bundle / "bundle-manifest.json"),
        "inputs_sha256": sha256(input_path), "input_count": len(rows),
        "base": manifest["base"], "task_config_sha256": sha256(bundle / "configs/model_task_v1.json"),
        "adapter_sha256": sha256(bundle / "adapter/adapter_model.safetensors"),
        "runtime_dependencies": dependencies, "network": "offline_local_files_only",
        "external_api_cost_usd": 0, "audio_evaluation": False}
    try:
        import torch
        from transformers import set_seed
        from safetensors import safe_open
        set_seed(task["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        tokenizer, config, compatibility = load_local_tokenizer_and_config(base_dir)
        receipt["loading_compatibility"] = compatibility
        receipt["python"] = platform.python_version()
        receipt["torch_cuda_version"] = torch.version.cuda
        receipt["prompt_token_counts"] = [len(prompt_tokens(tokenizer, task, row["text"])[1]) for row in rows]
        with safe_open(bundle / "adapter/adapter_model.safetensors", framework="pt", device="cpu") as tensors:
            keys = list(tensors.keys())
            if not keys or any("lora_" not in key for key in keys):
                raise ValueError("Unexpected adapter tensor keys")
            receipt["adapter_tensor_count"] = len(keys)
            receipt["adapter_parameter_count"] = sum(tensors.get_tensor(key).numel() for key in keys)
            if not all(torch.isfinite(tensors.get_tensor(key)).all().item() for key in keys):
                raise ValueError("Non-finite adapter tensors")
        if args.command == "check":
            receipt.update(device="cpu", generation_executed=False, loaded_model=False)
            if args.load_model:
                model = load_model(base_dir, bundle, config, "cpu")
                receipt.update(loaded_model=True, base_dtype=str(model.dtype),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    loaded_adapter_parameters=sum(p.numel() for n, p in model.named_parameters() if "lora_" in n),
                    adapter_dtypes=sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n}))
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("Frozen prediction semantics require CUDA FP16; check is CPU-only")
            model = load_model(base_dir, bundle, config, "cuda")
            receipt.update(device=torch.cuda.get_device_name(0), base_dtype=str(model.dtype), generation_executed=True)
            _predict(torch, model, tokenizer, task, rows, output, receipt, lambda: None)
        receipt["status"] = "complete"
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def portable_source(code_root):
    code_root = Path(code_root)
    own = Path(__file__)
    prefix = ('"""Offline portable inference; frozen helpers copied verbatim, no gold or training.\n'
              'CPU check never generates; predict retains original CUDA FP16 semantics.\n"""\n'
              'import argparse\nimport hashlib\nimport importlib.metadata\nimport json\nimport os\n'
              'from pathlib import Path, PurePosixPath\nimport platform\nimport re\nimport stat\nimport time\n\n'
              f'BASE_REPO = {BASE_REPO!r}\nBASE_REVISION = {BASE_REVISION!r}\n'
              f'BASE_LICENSE_SHA256 = {BASE_LICENSE_SHA256!r}\n'
              'CONFIG = {"generation_max_new_tokens": 1024}\n\n')
    return (prefix + extract_functions(own, ("sha256", "checked_relative", "verify_base", "verify_bundle"))
            + '\n\nhash_file = sha256\n\n'
            + extract_functions(code_root / SOURCE_PATHS[0], ("load_local_tokenizer_and_config",))
            + extract_functions(code_root / SOURCE_PATHS[1],
                                ("strict_json", "verify_runtime_dependencies", "prompt_tokens", "_predict"))
            + extract_functions(code_root / SOURCE_PATHS[2],
                                ("escape_jsonl_separators", "serialize_jsonl", "parse_inputs"))
            + PORTABLE_GLUE)


README = '''# ReadCue v2 local research candidate

Status: local, unpublished candidate. No upload or release has been performed.
The adapter's publication license is **pending a separate maintainer decision**;
neither the source-code license nor the base-model license assigns one to it.
See `RELEASE_TERMS.md`, `docs/MODEL_CARD_V2.md`, and `provenance/source-summary.json`.

This is the selected v2 LoRA adapter for Qwen/Qwen3-0.6B, revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Obtain that base revision separately
and supply its complete local directory via `--base-dir`. This bundle contains
no base weights, tokenizer copy, raw datasets, gold labels, predictions, secrets,
or training caches. The original adapter directory, including its autogenerated
README, is byte-identical. That README is historical PEFT metadata; this outer
README and the actual model card describe the candidate.

## Offline use

Use the recorded Linux Python 3.12 environment and exact
`requirements-model-v1.lock`. It includes CUDA and inherited experiment packages;
it is an environment snapshot, not a cross-platform CPU installation guarantee.
`requirements-zh-pilot.lock` is retained as historical supporting provenance,
not an alternative runtime lock. No command here installs or downloads anything.

From any directory, with a new output directory each time:

```sh
python /path/to/candidate/inference.py check --load-model \\
  --base-dir /path/to/fixed-qwen3-0.6b --output /path/to/new-cpu-check
python /path/to/candidate/inference.py predict \\
  --base-dir /path/to/fixed-qwen3-0.6b \\
  --inputs /path/to/candidate/example-inputs.jsonl --output /path/to/new-gpu-smoke
```

Every input line must contain only `id` and `text`; ID is bookkeeping only.
The included assistant-original one-line example has no expected output and is
only a loading/inference demonstration, not a new evaluation set. Prediction
uses the frozen system prompt, non-thinking template, greedy decoding, 1024 new
tokens, CUDA FP16, SDPA, original EOS/padding handling and unchanged decoding.
There is no answer extraction, trimming, normalization, or rule fallback.
Failures and truncations remain in predictions. Inputs are never truncated.

The CPU check verifies all bundle and base bytes, dependency versions, complete
token vocabulary/IDs, added-token flags, FIM markers, default RoPE mapping,
adapter tensor finiteness and (with `--load-model`) binding to the base on CPU.
It performs no generation, provides no inference-quality or audio evidence,
and does not certify CUDA output equivalence. CUDA smoke is a separate step.

`bundle-manifest.json` inventories immutable payload hashes. `checks/` is reserved
for separately generated local validation receipts and excluded from the payload
inventory. Do not treat these hashes as a cryptographic signature from a publisher.
The copied model card is an exact repository snapshot; links to the original
run report refer to the source checkout and are not bundled raw evidence.

## Research limits

The frozen 45 synthetic assistant-authored cases measured v2 strict sentence EM
32/45, v1 28/45 and fixed rules 25/45. These are measured text results, not paper
results, pronunciation accuracy, human acceptance or proof of broad superiority.
One training seed, one assistant review per candidate, known evaluation-isolation
limitations, semantic errors and unresolved readings remain. No TTS/audio claim.
No publication terms have been finalized and no commercial guarantee is made.
'''


TERMS = '''# Release terms status

This directory is a local, unpublished candidate; no model license is assigned.
Actual publication requires a separate decision on adapter terms and final
review of attribution and distribution contents. Preparation is not upload
authorization. There is no fabricated model URL, release tag or DOI.

- Original ReadCue code/documentation: project Apache-2.0 text is copied as
  `licenses/READCUE_CODE_LICENSE.txt`; it does not automatically license weights.
- Base: Qwen/Qwen3-0.6B at the fixed revision in the manifest. Its exact LICENSE
  is copied as `licenses/QWEN_BASE_LICENSE.txt`; the base itself is not redistributed.
- Training material: project assistant-original, explicitly versioned research
  training/validation derivatives (166/53); their Apache-2.0 scope and purpose
  records do not automatically determine the adapter's publication terms.
  Assistant review is not user approval. No third-party comments or specialist
  TN predictions supplied supervised targets. No dataset/gold is included here.
- The original PEFT adapter README remains unchanged for byte-level provenance.
  Its placeholder fields and local paths are not a completed publication card.
'''


def build_bundle(project, run_root, base_dir, output):
    project, run_root, base_dir, output = map(lambda p: Path(p).absolute(),
                                             (project, run_root, base_dir, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError("Candidate output must be new")
    for protected in (project, run_root, base_dir):
        if output.resolve().is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(output.resolve()):
            raise ValueError("Candidate must be separate from project, run and base")
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise ValueError("Weights must stay outside Git checkouts")
    train = run_root / "train"
    run = json.loads((train / "run.json").read_text(encoding="utf-8"))
    if (run.get("status") != "complete" or run.get("experiment") != "model-experiment-v2"
            or run.get("command") != "train" or not run.get("independent_reload_verified")):
        raise ValueError("A completed, independently reloaded v2 training run is required")
    selected = json.loads((train / "selected_adapter.json").read_text(encoding="utf-8"))
    adapter = checked_relative(train, selected["relative_directory"])
    metadata_path = checked_relative(adapter, "readcue_adapter.json")
    if sha256(metadata_path) != selected["metadata_sha256"]:
        raise ValueError("Selected adapter metadata changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("base_repo") != BASE_REPO or metadata.get("base_revision") != BASE_REVISION
            or metadata.get("dataset_version") != "model-data-v2" or metadata.get("experiment_phase") != "train"):
        raise ValueError("Unexpected selected adapter identity")
    names = set(metadata["files_sha256"])
    if names != {"adapter_config.json", "adapter_model.safetensors", "README.md"} or {
            p.name for p in adapter.iterdir()} != names | {"readcue_adapter.json"}:
        raise ValueError("Unexpected adapter assets")
    copies = {}
    for name, digest in metadata["files_sha256"].items():
        path = checked_relative(adapter, name)
        if sha256(path) != digest:
            raise ValueError("Adapter file changed: " + name)
        copies["adapter/" + name] = path
    if sha256(adapter / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise ValueError("This builder only packages the original selected v2 adapter")
    copies["adapter/readcue_adapter.json"] = metadata_path
    snapshots = {"task-config.json": ("configs/model_task_v1.json", "task_config_sha256"),
                 "annotation-policy.md": ("docs/MODEL_TASK_V1.md", "annotation_policy_sha256"),
                 "training-config.json": ("provenance/training-config.json", "training_config_sha256"),
                 "requirements-model-v1.lock": ("requirements-model-v1.lock", "runtime_lock_sha256"),
                 "base-model-manifest.json": ("provenance/base-model-manifest.json", "base_model_manifest_sha256")}
    for source_name, (destination, hash_key) in snapshots.items():
        source = checked_relative(train, source_name)
        if sha256(source) != run[hash_key]:
            raise ValueError("Frozen training snapshot changed: " + source_name)
        if hash_key in metadata and metadata[hash_key] != run[hash_key]:
            raise ValueError("Adapter and training snapshot disagree: " + hash_key)
        copies[destination] = source
    for source_name, destination in (("task-config.json", "configs/model_task_v1.json"),
                                     ("annotation-policy.md", "docs/MODEL_TASK_V1.md"),
                                     ("requirements-model-v1.lock", "requirements-model-v1.lock")):
        if sha256(project / destination) != sha256(train / source_name):
            raise ValueError("Checkout differs from frozen task/policy/lock: " + destination)
    for relative in SOURCE_PATHS:
        source = checked_relative(train / "code", relative)
        if sha256(source) != run["source_sha256"][relative]:
            raise ValueError("Frozen execution source changed: " + relative)
        copies["provenance/code/" + relative] = source
    base_manifest = json.loads((train / "base-model-manifest.json").read_text(encoding="utf-8"))
    verify_base(base_dir, base_manifest)
    copies.update({"licenses/QWEN_BASE_LICENSE.txt": base_dir / "LICENSE",
                   "licenses/READCUE_CODE_LICENSE.txt": project / "LICENSE",
                   "docs/MODEL_CARD_V2.md": project / "docs/MODEL_CARD_V2.md",
                   "requirements-zh-pilot.lock": project / "requirements-zh-pilot.lock",
                   "provenance/selected_adapter.json": train / "selected_adapter.json",
                   "provenance/build_model_research_bundle.py": Path(__file__)})
    source_hashes = {relative: sha256(source) for relative, source in copies.items()}
    portable = portable_source(train / "code")
    compile(portable, "inference.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    for relative, source in sorted(copies.items()):
        destination = checked_relative(output, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(destination) != source_hashes[relative] or sha256(source) != source_hashes[relative]:
            raise ValueError("Source changed during byte-identical copy: " + relative)
    generated = {"inference.py": portable, "README.md": README, "RELEASE_TERMS.md": TERMS,
                 "example-inputs.jsonl": json.dumps({"id": "local-smoke-1", "text": "仓库今天收到了37箱白色信封。"}, ensure_ascii=False) + "\n"}
    if sha256(train / "environment.json") != run["environment_sha256"]:
        raise ValueError("Frozen training environment snapshot changed")
    summary = {"schema_version": 1, "status": "local_unpublished_candidate",
        "adapter_publication_license": "pending_separate_confirmation", "source_run": run_root.name,
        "source_run_json_sha256": sha256(train / "run.json"),
        "source_run_fields": {key: run[key] for key in ("experiment", "base_repo", "base_revision",
            "dataset_version", "dataset_manifest_sha256", "data_counts", "data_files_sha256",
            "optimizer_updates", "completed_epochs", "trainable_parameters", "independent_reload_verified",
            "independent_reload_validation_loss", "elapsed_seconds", "external_api_cost_usd", "source_sha256")},
        "source_environment_sha256": sha256(train / "environment.json"),
        "environment": {key: value for key, value in json.loads((train / "environment.json").read_text(encoding="utf-8")).items()
                        if key != "runtime_paths"},
        "dataset_scope": "assistant-original versioned research training/validation; no third-party comments or TN targets",
        "copied_source_sha256": source_hashes,
        "portable_frozen_functions": {SOURCE_PATHS[0]: ["load_local_tokenizer_and_config"],
            SOURCE_PATHS[1]: ["strict_json", "verify_runtime_dependencies", "prompt_tokens", "_predict"],
            SOURCE_PATHS[2]: ["escape_jsonl_separators", "serialize_jsonl", "parse_inputs"]},
        "example_scope": "assistant-original id/text-only smoke input; no reference; not an evaluation set",
        "excluded": ["base_weights", "tokenizer_copy", "raw_datasets", "gold_labels", "predictions", "secrets", "runtime_caches"]}
    generated["provenance/source-summary.json"] = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    for relative, body in generated.items():
        target = checked_relative(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(body)
    manifest = {"schema_version": 1, "base": {"repo": BASE_REPO, "revision": BASE_REVISION},
        "status": "local_unpublished_candidate", "adapter_publication_license": "pending_separate_confirmation",
        "files": {path.relative_to(output).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
                  for path in sorted(output.rglob("*")) if path.is_file()}}
    with (output / "bundle-manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    verify_bundle(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="Verify an existing payload; optional --base-dir also hashes base")
    args = parser.parse_args(argv)
    if args.verify:
        result = verify_bundle(args.verify, args.base_dir)
    else:
        if not all((args.project, args.run_root, args.base_dir, args.output)):
            parser.error("Build requires --project, --run-root, --base-dir and --output")
        result = build_bundle(args.project, args.run_root, args.base_dir, args.output)
    print(json.dumps({"status": "verified", "payload_files": len(result["files"]),
                      "adapter_publication_license": result.get("adapter_publication_license")}, indent=2))


if __name__ == "__main__":
    main()
