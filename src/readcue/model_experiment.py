"""One bounded, offline Qwen3 LoRA experiment; ML imports occur only on execution.

Training reads only explicitly assigned train/validation files. Prediction accepts
only id/text and never opens a dataset or a held-out reference file.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import stat
import sys
import tempfile
import time
import traceback
import unicodedata

from . import artifacts, inference_contract
from .artifacts import hash_file, project_path, verify_model, verify_pilot, write_json
from .inference_contract import parse_inputs, serialize_jsonl


BASE_REPO = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DATASET_USAGES = {
    "model-data-v1": "first_bounded_model_research_training_validation",
    "model-data-v2": "second_bounded_model_research_training_validation",
}
CONFIG = {"version": "model-experiment-v1", "seed": 20260928, "rank": 8, "alpha": 16,
          "dropout": 0.05, "target_modules": ["q_proj", "v_proj"], "bias": "none",
          "base_dtype": "float16", "trainable_dtype": "float32", "device": "cuda",
          "micro_batch_size": 1, "gradient_accumulation": 8, "learning_rate": 1e-4,
          "weight_decay": 0.0, "warmup_ratio": 0.03, "lr_schedule": "linear_warmup_then_constant",
          "max_grad_norm": 1.0, "gradient_checkpointing": True, "max_sequence_tokens": 1024,
          "max_epochs": 3, "max_optimizer_updates": 100, "max_training_seconds": 7200,
          "max_preflight_seconds": 900, "max_output_bytes": 5 * 1024 ** 3,
          "max_checkpoints": 2, "checkpoint_selection": "validation_target_token_mean_cross_entropy_earlier_tie",
          "generation_max_new_tokens": 1024, "external_api_cost_usd": 0}
SOURCE_FILES = ("scripts/run_model_experiment.py", "src/readcue/__init__.py", "src/readcue/model_experiment.py",
                "src/readcue/artifacts.py", "src/readcue/inference_contract.py",
                "scripts/run_zh_baseline.py")
RUNTIME_LOCK = "requirements-model-v1.lock"
ML_MODULES = ("torch", "transformers", "peft", "accelerate", "triton", "huggingface_hub")


def execution_sources(project, entrypoint=None):
    """Bind checkout snapshots to the files supplying the running Python code."""
    project = Path(project).resolve()
    loaded = {"src/readcue/__init__.py": sys.modules[__package__].__file__,
              "src/readcue/model_experiment.py": __file__,
              "src/readcue/artifacts.py": artifacts.__file__,
              "src/readcue/inference_contract.py": inference_contract.__file__}
    snapshots, modules = {}, {}
    for relative in SOURCE_FILES:
        expected = project_path(project, relative).read_bytes()
        if relative in loaded:
            actual_path = Path(loaded[relative]).resolve()
            actual = actual_path.read_bytes()
            if actual != expected:
                raise ValueError("Loaded module differs from --project: " + relative)
            modules[relative] = {"path": str(actual_path), "sha256": hashlib.sha256(actual).hexdigest()}
            snapshots[relative] = actual
        else:
            snapshots[relative] = expected
    entrypoint = Path(entrypoint or __file__).resolve()
    if entrypoint == Path(__file__).resolve():
        kind, relative = "module", "src/readcue/model_experiment.py"
    else:
        kind, relative = "script", "scripts/run_model_experiment.py"
        if entrypoint != project_path(project, relative):
            raise ValueError("Script entrypoint must belong to --project; do not mix checkouts")
    if entrypoint.read_bytes() != snapshots[relative]:
        raise ValueError("Entrypoint changed during source verification")
    identity = {"project": str(project), "loaded_modules": modules,
                "entrypoint": {"kind": kind, "path": str(entrypoint),
                               "sha256": hashlib.sha256(snapshots[relative]).hexdigest()}}
    return snapshots, identity


def verify_runtime_dependencies(lock_path):
    """Inspect installed metadata before importing any optional model library."""
    body = Path(lock_path).read_bytes()
    expected, installed, problems = {}, {}, []
    for line in body.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)", line)
        if not match:
            raise ValueError("Runtime lock requires exact package==version entries: " + line)
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in expected:
            raise ValueError("Duplicate runtime lock package: " + name)
        expected[name] = match[2]
    if not expected:
        raise ValueError("Runtime dependency lock is empty")
    for name, version in expected.items():
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
        if installed[name] != version:
            problems.append(f"{name}: expected {version}, installed {installed[name] or 'missing'}")
    if problems:
        raise RuntimeError("Locked runtime dependency mismatch: " + "; ".join(problems))
    return {"lock_sha256": hashlib.sha256(body).hexdigest(), "installed_versions": installed,
            "unlisted_packages": "not_checked"}


def now():
    return datetime.now(timezone.utc).isoformat()


def strict_json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Non-finite JSON value: " + value)

    return json.loads(body, object_pairs_hook=pairs, parse_constant=invalid)


def load_task(project, path):
    body = Path(path).read_bytes()
    task = strict_json(body)
    expected = {"schema_version": 1, "task_version": "zh-reading-model-v1",
                "input_fields": ["id", "text"], "model_input_fields": ["text"],
                "output_format": "complete_sentence_text", "enable_thinking": False,
                "assistant_label_masking": "prompt_and_input_ignored_target_and_eos_supervised",
                "generation": {"do_sample": False, "max_new_tokens": 1024}, "seed": 20260928}
    for key, value in expected.items():
        if task.get(key) != value or type(task.get(key)) is not type(value):
            raise ValueError("Unexpected fixed task setting: " + key)
    if not isinstance(task.get("system_prompt"), str) or not task["system_prompt"].strip():
        raise ValueError("The fixed task prompt is missing")
    policy = project_path(project, task["annotation_policy"])
    return task, body, policy


def normalized_text(text):
    return "".join(unicodedata.normalize("NFKC", text).split()).casefold()


def load_training_data(directory, task_hash, policy_hash, *, include_validation=True):
    """Never traverse manifest files: the only opened labels are these two roles."""
    directory = Path(directory)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = strict_json(manifest_bytes)
    dataset_version = manifest.get("dataset_version")
    if (manifest.get("schema_version") != 1 or dataset_version not in DATASET_USAGES
            or manifest.get("roles") != {"train": "train.jsonl", "validation": "validation.jsonl"}
            or manifest.get("task_config_sha256") != task_hash
            or manifest.get("annotation_policy_sha256") != policy_hash):
        raise ValueError("Dataset role or task/policy binding does not match this experiment")
    if dataset_version == "model-data-v2" and not re.fullmatch(
            r"[0-9a-f]{64}", str(manifest.get("source_v1_manifest_sha256", ""))):
        raise ValueError("The v2 dataset must identify its unchanged v1 source manifest")
    result, snapshots = {}, {"manifest.json": manifest_bytes}
    for role in (("train", "validation") if include_validation else ("train",)):
        filename = role + ".jsonl"
        body = (directory / filename).read_bytes()
        if hashlib.sha256(body).hexdigest() != manifest.get("files_sha256", {}).get(filename):
            raise ValueError("Dataset file hash changed: " + filename)
        rows = [strict_json(line) for line in body.decode("utf-8").split("\n") if line.strip()]
        if not 0 < len(rows) <= (256 if role == "train" else 64):
            raise ValueError("Dataset count exceeds the fixed role limit: " + role)
        ids, texts = set(), set()
        for row in rows:
            required = {"id", "text", "target", "acceptable_outputs", "track", "family", "group_id", "provenance"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError("Missing supervised-data fields")
            if any(not isinstance(row[key], str) or not row[key].strip()
                   for key in ("id", "text", "target", "track", "family", "group_id")):
                raise ValueError("Invalid supervised-data text or identity")
            references = row["acceptable_outputs"]
            if (not isinstance(references, list) or not references
                    or any(not isinstance(value, str) or not value.strip() for value in references)
                    or row["target"] not in references):
                raise ValueError("The registered training target must be an acceptable reference")
            provenance = row["provenance"]
            if (not isinstance(provenance, dict) or provenance.get("origin") != "assistant_original"
                    or provenance.get("license") != "Apache-2.0"
                    or provenance.get("assigned_role") != role
                    or provenance.get("reference_status") != "assistant_policy_reviewed"
                    or provenance.get("usage") != DATASET_USAGES[dataset_version]
                    or provenance.get("training_ready") is not True
                    or provenance.get("task_config_sha256") != task_hash
                    or provenance.get("annotation_policy_sha256") != policy_hash):
                raise ValueError("Unregistered or unapproved supervised-data provenance: " + row["id"])
            canonical = normalized_text(row["text"])
            if row["id"] in ids or canonical in texts:
                raise ValueError("Duplicate id or normalized source within " + role)
            ids.add(row["id"])
            texts.add(canonical)
        result[role] = rows
        snapshots[filename] = body
    if include_validation:
        for label, key in (("id", lambda row: row["id"]), ("group", lambda row: row["group_id"]),
                           ("normalized source", lambda row: normalized_text(row["text"]))):
            if {key(row) for row in result["train"]} & {key(row) for row in result["validation"]}:
                raise ValueError("Train/validation overlap: " + label)
    return result, manifest, snapshots


def load_heldout_proof(path, task_hash):
    """Read a metadata proof only; never open any file named inside it."""
    body = Path(path).read_bytes()
    proof = strict_json(body)
    files = proof.get("files_sha256", {})
    if (proof.get("schema_version") != 1 or proof.get("status") != "frozen"
            or proof.get("task_config_sha256") != task_hash
            or not isinstance(files, dict)
            or not {"inputs.jsonl", "cases.jsonl", "manifest.json"}.issubset(files)
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value) for value in files.values())):
        raise ValueError("A frozen held-out metadata proof is required before training")
    return proof, body


def prompt_tokens(tokenizer, task, text):
    messages = [{"role": "system", "content": task["system_prompt"]}, {"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    return prompt, list(tokenizer.encode(prompt, add_special_tokens=False))


def encode_supervised(tokenizer, task, row, max_tokens=1024):
    """The assistant non-thinking prefix is input, not a supervised target."""
    prompt, prefix = prompt_tokens(tokenizer, task, row["text"])
    target = list(tokenizer.encode(row["target"], add_special_tokens=False))
    combined = list(tokenizer.encode(prompt + row["target"], add_special_tokens=False))
    if combined != prefix + target:
        raise ValueError("Prompt/target token boundary changed: " + row["id"])
    eos = tokenizer.eos_token_id
    if type(eos) is not int or not target or eos in target:
        raise ValueError("Invalid target or EOS token: " + row["id"])
    tokens = combined + [eos]
    if len(tokens) > max_tokens:
        raise ValueError(f"Sequence exceeds {max_tokens} tokens without truncation: {row['id']} ({len(tokens)})")
    return {"id": row["id"], "input_ids": tokens,
            "labels": [-100] * len(prefix) + target + [eos],
            "target_tokens": len(target) + 1, "prompt_tokens": len(prefix)}


def accumulation_windows(items, size=8):
    if size < 1:
        raise ValueError("Accumulation size must be positive")
    return [items[offset:offset + size] for offset in range(0, len(items), size)]


def target_weight(example, window):
    return example["target_tokens"] / sum(item["target_tokens"] for item in window)


def checkpoint_is_better(loss, best_loss):
    if not math.isfinite(loss):
        raise ValueError("Non-finite validation loss")
    return best_loss is None or loss < best_loss


def learning_rate_for_update(step, total):
    warmup = max(1, math.ceil(total * CONFIG["warmup_ratio"]))
    return CONFIG["learning_rate"] * min(step / warmup, 1.0)


def directory_bytes(root):
    def fail(error):
        raise error

    total = 0
    _reject_link(root)
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        for name in dirs + files:
            path = Path(directory) / name
            _reject_link(path)
        total += sum((Path(directory) / name).stat().st_size for name in files)
    return total


def _reject_link(path):
    info = Path(path).lstat()
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
        raise ValueError("Experiment budget directory must not contain links or reparse points")


def prepare_runtime_paths(budget_root, output):
    """Share owned caches between stages; keep temporary files inside each new run."""
    budget_root, output = Path(budget_root).resolve(), Path(output).resolve()
    if output == budget_root or not output.is_relative_to(budget_root):
        raise ValueError("Each stage must have its own directory inside --budget-root")
    if output.is_relative_to(budget_root / "runtime-cache-v1"):
        raise ValueError("Stage outputs must not overlap the shared runtime cache")
    directory_bytes(budget_root)
    runtime = budget_root / "runtime-cache-v1"
    marker = runtime / "readcue-runtime.json"
    ownership = {"schema_version": 1, "purpose": "readcue-model-experiment-runtime-cache",
                 "budget_root": str(budget_root)}
    if runtime.exists():
        if not marker.is_file() or strict_json(marker.read_bytes()) != ownership:
            raise ValueError("Refuse to reuse an unowned runtime cache directory")
    else:
        runtime.mkdir()
        write_json(marker, ownership)
    cache, temporary = runtime / "cache", output / "tmp"
    paths = {"HF_HOME": cache / "huggingface", "HF_HUB_CACHE": cache / "huggingface/hub",
             "HUGGINGFACE_HUB_CACHE": cache / "huggingface/hub",
             "HF_ASSETS_CACHE": cache / "huggingface/assets", "HF_XET_CACHE": cache / "huggingface/xet",
             "HF_DATASETS_CACHE": cache / "huggingface/datasets", "TRANSFORMERS_CACHE": cache / "transformers",
             "TORCH_HOME": cache / "torch", "TORCH_EXTENSIONS_DIR": cache / "torch-extensions",
             "XDG_CACHE_HOME": cache, "TRITON_CACHE_DIR": cache / "triton",
             "TRITON_DUMP_DIR": cache / "triton-dump", "TRITON_OVERRIDE_DIR": cache / "triton-override",
             "CUDA_CACHE_PATH": cache / "cuda", "TEMP": temporary, "TMP": temporary, "TMPDIR": temporary}
    for path in sorted(set(paths.values())):
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path) for name, path in paths.items()}


def activate_runtime_paths(paths):
    """Also reset Python's cached temp path; inherited library paths must not win."""
    loaded = [name for name in ML_MODULES if name in sys.modules]
    if loaded:
        raise RuntimeError("Start a fresh process before configuring model caches: " + ", ".join(loaded))
    updates = {**paths, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"}
    previous = ({name: os.environ.get(name) for name in updates}, tempfile.tempdir, sys.dont_write_bytecode)
    os.environ.update(updates)
    tempfile.tempdir = paths["TMPDIR"]
    sys.dont_write_bytecode = True
    return previous


def restore_runtime_paths(previous):
    environment, tempfile.tempdir, sys.dont_write_bytecode = previous
    for name, value in environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def check_resources(budget_root, started, seconds):
    elapsed = time.monotonic() - started
    if elapsed >= seconds:
        raise RuntimeError("Fixed wall-clock resource limit reached")
    used = directory_bytes(budget_root)
    if used > CONFIG["max_output_bytes"]:
        raise RuntimeError("Fixed 5 GiB experiment-output/cache limit reached")
    return {"elapsed_seconds": elapsed, "experiment_bytes": used}


def _new_output(output, project, protected, budget_root):
    output, project = Path(output).resolve(), Path(project).resolve()
    if output.is_relative_to(project) or any((ancestor / ".git").exists() for ancestor in output.parents):
        raise ValueError("Model experiment outputs must be outside Git checkouts")
    for source in protected:
        source = Path(source).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Experiment output must not overlap source assets or data")
    if not budget_root:
        raise ValueError("An explicit shared --budget-root is required for all experiment stages")
    budget_root = Path(budget_root).resolve()
    if output == budget_root or not output.is_relative_to(budget_root):
        raise ValueError("Each stage must have its own directory inside --budget-root")
    if output.is_relative_to(budget_root / "runtime-cache-v1"):
        raise ValueError("Stage outputs must not overlap the shared runtime cache")
    if budget_root.is_relative_to(project) or any((ancestor / ".git").exists() for ancestor in (budget_root, *budget_root.parents)):
        raise ValueError("The experiment budget root must be outside Git checkouts")
    output.mkdir(parents=True, exist_ok=False)
    return output, budget_root


def _legacy_helpers(project):
    path = project_path(project, "scripts/run_zh_baseline.py")
    spec = importlib.util.spec_from_file_location("readcue_verified_pilot_loading", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_base(project, model_dir):
    freeze = verify_pilot(project)
    trusted = next(item for item in freeze["baseline_assets"] if item["repo"] == BASE_REPO)
    return verify_model(model_dir, BASE_REPO, BASE_REVISION, trusted=trusted)


def _runtime(seed, lock_path):
    dependencies = verify_runtime_dependencies(lock_path)
    import torch
    from transformers import set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("The fixed experiment requires the existing CUDA device")
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("torch", "transformers", "peft", "accelerate", "safetensors")},
                   "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                   "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "network": "offline_local_files_only", "external_api_cost_usd": 0,
                   "runtime_dependencies": dependencies}
    return torch, environment


def _load_base(project, model_dir):
    import torch
    from transformers import AutoModelForCausalLM
    _verify_base(project, model_dir)
    tokenizer, config, compatibility = _legacy_helpers(project).load_local_tokenizer_and_config(model_dir)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), config=config, local_files_only=True,
             trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float16,
             attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda")
    return model, tokenizer, compatibility


def _add_lora(model):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=CONFIG["rank"], lora_alpha=CONFIG["alpha"],
                        lora_dropout=CONFIG["dropout"], target_modules=CONFIG["target_modules"], bias="none")
    model = get_peft_model(model, config, autocast_adapter_dtype=True)
    model.peft_config["default"].base_model_name_or_path = BASE_REPO
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError("Unexpected trainable base parameter: " + name)
            parameter.data = parameter.data.to(torch.float32)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
        raise ValueError("LoRA trainable parameters must be FP32 for GradScaler")
    return model, trainable


def _batch(torch, example):
    inputs = torch.tensor([example["input_ids"]], dtype=torch.long, device="cuda")
    labels = torch.tensor([example["labels"]], dtype=torch.long, device="cuda")
    return {"input_ids": inputs, "attention_mask": torch.ones_like(inputs), "labels": labels, "use_cache": False}


def _validate(torch, model, examples, guard):
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for example in examples:
            guard()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**_batch(torch, example)).loss
            value = float(loss.detach().float().item())
            if not math.isfinite(value):
                raise RuntimeError("Non-finite validation loss")
            total_loss += value * example["target_tokens"]
            total_tokens += example["target_tokens"]
    guard()
    return total_loss / total_tokens


def _train_window(torch, model, optimizer, scaler, trainable, window, guard):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_tokens = 0.0, 0
    for example in window:
        guard()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**_batch(torch, example)).loss
        value = float(loss.detach().float().item())
        if not math.isfinite(value):
            raise RuntimeError("Non-finite training loss")
        scaler.scale(loss * target_weight(example, window)).backward()
        total_loss += value * example["target_tokens"]
        total_tokens += example["target_tokens"]
        guard()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(trainable, CONFIG["max_grad_norm"], error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    guard()
    return {"loss": total_loss / total_tokens, "target_tokens": total_tokens,
            "micro_batches": len(window), "gradient_norm": float(norm.item())}


def _save_adapter(model, directory, state, step, epoch, validation_loss):
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(directory), safe_serialization=True, save_embedding_layers=False)
    names = {path.name for path in directory.iterdir() if path.is_file()}
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(names):
        raise RuntimeError("Adapter serialization did not produce local safetensors")
    metadata = {"schema_version": 1, "base_repo": BASE_REPO, "base_revision": BASE_REVISION,
                "task_config_sha256": state["task_config_sha256"],
                "annotation_policy_sha256": state["annotation_policy_sha256"],
                "training_config_sha256": state["training_config_sha256"],
                "dataset_manifest_sha256": state["dataset_manifest_sha256"],
                "dataset_version": state.get("dataset_version", "model-data-v1"),
                "experiment_phase": state["command"], "optimizer_updates": step, "epoch": epoch,
                "validation_target_token_loss": validation_loss,
                "files_sha256": {name: hash_file(directory / name) for name in sorted(names)}}
    write_json(directory / "readcue_adapter.json", metadata)
    return {"directory": str(directory), "optimizer_updates": step, "epoch": epoch,
            "validation_loss": validation_loss, "metadata_sha256": hash_file(directory / "readcue_adapter.json")}


def verify_adapter(path, task_hash, *, allow_preflight=False):
    path = Path(path).resolve()
    if (path / "selected_adapter.json").is_file():
        selected = strict_json((path / "selected_adapter.json").read_bytes())
        path = project_path(path, selected["relative_directory"])
        if hash_file(path / "readcue_adapter.json") != selected["metadata_sha256"]:
            raise ValueError("Selected adapter metadata changed")
    metadata = strict_json((path / "readcue_adapter.json").read_bytes())
    if (metadata.get("base_repo") != BASE_REPO or metadata.get("base_revision") != BASE_REVISION
            or metadata.get("task_config_sha256") != task_hash
            or metadata.get("dataset_version", "model-data-v1") not in DATASET_USAGES
            or metadata.get("experiment_phase") not in ({"train", "preflight"} if allow_preflight else {"train"})):
        raise ValueError("Adapter identity does not match the fixed experiment")
    names = set(metadata.get("files_sha256", {}))
    if (not {"adapter_config.json", "adapter_model.safetensors"}.issubset(names)
            or names - {"adapter_config.json", "adapter_model.safetensors", "README.md"}
            or {item.name for item in path.iterdir()} != names | {"readcue_adapter.json"}):
        raise ValueError("Unexpected or missing adapter assets")
    for name, expected in metadata["files_sha256"].items():
        item = path / name
        if item.is_symlink() or not item.is_file() or hash_file(item) != expected:
            raise ValueError("Adapter asset changed: " + name)
    return path, metadata


def _reload_adapter(project, model_dir, adapter, task_hash, *, allow_preflight=False):
    from peft import PeftModel
    adapter, metadata = verify_adapter(adapter, task_hash, allow_preflight=allow_preflight)
    model, tokenizer, compatibility = _load_base(project, model_dir)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False,
                                     local_files_only=True, use_safetensors=True)
    return model.eval(), tokenizer, compatibility, metadata


def _remove_old_checkpoint(path, output):
    original = Path(path)
    path, root = original.resolve(), (output / "checkpoints").resolve()
    if path.parent != root or not path.name.startswith("step-") or original.is_symlink():
        raise ValueError("Refuse to remove a checkpoint outside this new experiment")
    shutil.rmtree(path)


def execute(args):
    project, model_dir, task_path = Path(args.project).resolve(), Path(args.model_dir).resolve(), Path(args.task_config).resolve()
    protected = [project, model_dir, task_path]
    protected.extend(Path(value).resolve() for value in (args.dataset, args.inputs, args.adapter, args.heldout_proof) if value)
    output, budget_root = _new_output(args.output, project, protected, args.budget_root)
    started = time.monotonic()
    limit = CONFIG["max_preflight_seconds"] if args.command == "preflight" else CONFIG["max_training_seconds"]
    guard = lambda: check_resources(budget_root, started, limit)
    state = {"schema_version": 1, "experiment": CONFIG["version"], "command": args.command,
             "status": "started", "phase": "identity_checks", "started_at_utc": now(),
             "training": args.command in {"train", "preflight"}, "external_api_cost_usd": 0,
             "base_repo": BASE_REPO, "base_revision": BASE_REVISION, "budget_root": str(budget_root),
             "optimizer_updates": 0, "checkpoints": []}
    write_json(output / "run.json", state)
    torch = None
    runtime_previous = None
    try:
        guard()
        sources, identity = execution_sources(project, getattr(args, "entrypoint", None))
        state["execution_identity"] = identity
        task, task_bytes, policy_path = load_task(project, task_path)
        manifest = _verify_base(project, model_dir)
        (output / "task-config.json").write_bytes(task_bytes)
        shutil.copyfile(policy_path, output / "annotation-policy.md")
        write_json(output / "training-config.json", CONFIG)
        write_json(output / "base-model-manifest.json", manifest)
        code = output / "code"
        code.mkdir()
        hashes = {}
        for relative, body in sources.items():
            target = code / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            hashes[relative] = hashlib.sha256(body).hexdigest()
        lock_path = output / RUNTIME_LOCK
        lock_path.write_bytes(project_path(project, RUNTIME_LOCK).read_bytes())
        state.update(task_config_sha256=hash_file(output / "task-config.json"),
                     annotation_policy_sha256=hash_file(output / "annotation-policy.md"),
                     training_config_sha256=hash_file(output / "training-config.json"),
                     base_model_manifest_sha256=hash_file(output / "base-model-manifest.json"),
                     base_model_manifest_source_sha256=hash_file(model_dir / "readcue_manifest.json"), source_sha256=hashes,
                     runtime_lock_sha256=hash_file(lock_path))
        if args.command == "predict":
            if not args.inputs or args.dataset or args.heldout_proof:
                raise ValueError("Prediction requires only id/text inputs; dataset/reference arguments are forbidden")
            original = Path(args.inputs).read_bytes()
            rows = parse_inputs(original.decode("utf-8"))
            (output / "inputs.original.jsonl").write_bytes(original)
            state.update(input_sha256=hash_file(output / "inputs.original.jsonl"), case_count=len(rows), inference_fields=["text"])
        else:
            if not args.dataset or args.inputs or args.adapter:
                raise ValueError("Training/preflight require an assigned dataset and a fresh base, never an input/adapter override")
            data, dataset_manifest, snapshots = load_training_data(args.dataset, state["task_config_sha256"],
                state["annotation_policy_sha256"], include_validation=args.command == "train")
            data_directory = output / "dataset-snapshot"
            data_directory.mkdir()
            for name, body in snapshots.items():
                (data_directory / name).write_bytes(body)
            state.update(dataset_version=dataset_manifest["dataset_version"],
                         experiment=dataset_manifest["dataset_version"].replace("model-data-", "model-experiment-"),
                         initialization="fresh_fixed_base",
                         dataset_manifest_sha256=hash_file(data_directory / "manifest.json"),
                         data_files_sha256={name: hash_file(data_directory / name) for name in snapshots},
                         data_counts={role: len(values) for role, values in data.items()}, heldout_references_read=False)
            if args.command == "train" and not args.heldout_proof:
                raise ValueError("Formal training requires --heldout-proof before the first update")
            if args.heldout_proof:
                proof, proof_bytes = load_heldout_proof(args.heldout_proof, state["task_config_sha256"])
                if (state["dataset_version"] == "model-data-v2"
                        and proof.get("training_dataset_manifest_sha256") != state["dataset_manifest_sha256"]):
                    raise ValueError("The v2 heldout proof must bind the actual training dataset manifest")
                (output / "heldout-proof.json").write_bytes(proof_bytes)
                state["heldout_proof_sha256"] = hash_file(output / "heldout-proof.json")
        state["phase"] = "loading"
        state["runtime_paths"] = prepare_runtime_paths(budget_root, output)
        write_json(output / "run.json", state)
        runtime_previous = activate_runtime_paths(state["runtime_paths"])
        guard()
        torch, environment = _runtime(CONFIG["seed"], lock_path)
        environment["runtime_paths"] = state["runtime_paths"]
        write_json(output / "environment.json", environment)
        state["environment_sha256"] = hash_file(output / "environment.json")
        guard()
        if args.command == "predict" and args.adapter:
            model, tokenizer, compatibility, adapter_metadata = _reload_adapter(
                project, model_dir, args.adapter, state["task_config_sha256"])
            state["adapter"] = adapter_metadata
            state["adapter_dataset_version"] = adapter_metadata.get("dataset_version", "model-data-v1")
            state["experiment"] = state["adapter_dataset_version"].replace("model-data-", "model-experiment-")
        else:
            model, tokenizer, compatibility = _load_base(project, model_dir)
        state.update(loading_compatibility=compatibility, phase="running")
        write_json(output / "run.json", state)
        guard()
        if args.command == "predict":
            _predict(torch, model, tokenizer, task, rows, output, state, guard)
        else:
            model, trainable = _add_lora(model)
            state["trainable_parameters"] = sum(item.numel() for item in trainable)
            state["phase"] = "tokenization"
            write_json(output / "run.json", state)
            examples = {}
            lengths = []
            for role, values in data.items():
                examples[role] = []
                for row in values[:1] if args.command == "preflight" else values:
                    try:
                        example = encode_supervised(tokenizer, task, row, CONFIG["max_sequence_tokens"])
                    except ValueError as error:
                        lengths.append({"id": row["id"], "role": role, "status": "rejected", "error": str(error)})
                        write_json(output / "token-lengths.json", lengths)
                        raise
                    examples[role].append(example)
                    lengths.append({"id": row["id"], "role": role, "tokens": len(example["input_ids"]),
                                    "target_tokens": example["target_tokens"], "status": "accepted"})
            write_json(output / "token-lengths.json", lengths)
            state["phase"] = "training"
            write_json(output / "run.json", state)
            optimizer = torch.optim.AdamW(trainable, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
            scaler = torch.amp.GradScaler("cuda")
            if args.command == "preflight":
                metrics = _train_window(torch, model, optimizer, scaler, trainable, examples["train"], guard)
                state["optimizer_updates"] = 1
                checkpoint = _save_adapter(model, output / "checkpoints/step-0001-preflight", state, 1, 0, None)
                state["checkpoints"] = [checkpoint]
                before_reload = _validate(torch, model, examples["train"], guard)
                del model, optimizer, scaler, trainable
                gc.collect()
                torch.cuda.empty_cache()
                model, _, _, reloaded_metadata = _reload_adapter(project, model_dir, checkpoint["directory"],
                                                                  state["task_config_sha256"], allow_preflight=True)
                after_reload = _validate(torch, model, examples["train"], guard)
                if not math.isclose(before_reload, after_reload, rel_tol=1e-4, abs_tol=1e-4):
                    raise RuntimeError("Preflight adapter reload changed target-token loss")
                state.update(preflight_metrics=metrics, preflight_reload_loss=after_reload,
                             preflight_loss_before_reload=before_reload, independent_reload_verified=True,
                             preflight_adapter_manifest=reloaded_metadata)
                # The smoke adapter is not a training candidate. Retain its
                # hashes, then free its slot so the entire experiment keeps
                # at most the two formal checkpoints.
                write_json(output / "run.json", state)
                _remove_old_checkpoint(checkpoint["directory"], output)
                checkpoint["removed_after_verified_roundtrip"] = True
            else:
                _train(torch, model, optimizer, scaler, trainable, examples, output, state, guard)
                selected = state["selected_adapter"]
                del model, optimizer, scaler, trainable
                gc.collect()
                torch.cuda.empty_cache()
                model, _, _, _ = _reload_adapter(project, model_dir, selected["directory"], state["task_config_sha256"])
                reloaded_loss = _validate(torch, model, examples["validation"], guard)
                if not math.isclose(selected["validation_loss"], reloaded_loss, rel_tol=1e-4, abs_tol=1e-4):
                    raise RuntimeError("Independent adapter reload changed selected validation loss")
                state.update(independent_reload_verified=True, independent_reload_validation_loss=reloaded_loss)
                write_json(output / "selected_adapter.json", {"relative_directory": str(Path(selected["directory"]).relative_to(output)),
                          "metadata_sha256": selected["metadata_sha256"], "validation_loss": selected["validation_loss"]})
        state.update(status="complete", phase="complete", **guard())
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        try:
            state.update(finished_at_utc=now(), elapsed_seconds=time.monotonic() - started)
            if torch is not None:
                state.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(output / "run.json", state)
        finally:
            if runtime_previous is not None:
                restore_runtime_paths(runtime_previous)
    return state


def _train(torch, model, optimizer, scaler, trainable, examples, output, state, guard):
    planned = min(CONFIG["max_optimizer_updates"], math.ceil(len(examples["train"]) / CONFIG["gradient_accumulation"]) * CONFIG["max_epochs"])
    best, retained = None, []
    with (output / "loss.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for epoch in range(1, CONFIG["max_epochs"] + 1):
            order = list(examples["train"])
            random.Random(CONFIG["seed"] + epoch - 1).shuffle(order)
            for window in accumulation_windows(order, CONFIG["gradient_accumulation"]):
                step = state["optimizer_updates"] + 1
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate_for_update(step, planned)
                metrics = _train_window(torch, model, optimizer, scaler, trainable, window, guard)
                state["optimizer_updates"] = step
                metrics.update(epoch=epoch, optimizer_update=step, learning_rate=optimizer.param_groups[0]["lr"])
                log.write(serialize_jsonl([metrics]))
                log.flush()
                write_json(output / "run.json", state)
                if step >= CONFIG["max_optimizer_updates"]:
                    break
            validation_loss = _validate(torch, model, examples["validation"], guard)
            while len(retained) >= CONFIG["max_checkpoints"]:
                removed = next(item for item in retained if best is None or item["directory"] != best["directory"])
                _remove_old_checkpoint(removed["directory"], output)
                retained.remove(removed)
            checkpoint = _save_adapter(model, output / f"checkpoints/step-{state['optimizer_updates']:04d}-epoch-{epoch}",
                                        state, state["optimizer_updates"], epoch, validation_loss)
            retained.append(checkpoint)
            if checkpoint_is_better(validation_loss, None if best is None else best["validation_loss"]):
                best = checkpoint
            state.update(checkpoints=list(retained), selected_adapter=best, completed_epochs=epoch)
            log.write(serialize_jsonl([{"epoch": epoch, "validation_loss": validation_loss,
                                       "optimizer_update": state["optimizer_updates"], "selected": checkpoint is best}]))
            log.flush()
            write_json(output / "run.json", state)
            guard()
            if state["optimizer_updates"] >= CONFIG["max_optimizer_updates"]:
                break


def _predict(torch, model, tokenizer, task, rows, output, state, guard):
    model.eval()
    model.config.use_cache = True
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else eos
    if not eos_ids:
        raise ValueError("Generation requires a recorded EOS token identity")
    failures = 0
    with (output / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            guard()
            started = time.monotonic()
            raw = ""
            prediction = ""
            tokens = []
            error = None
            fatal = None
            try:
                _, prefix = prompt_tokens(tokenizer, task, row["text"])
                if len(prefix) + CONFIG["generation_max_new_tokens"] > model.config.max_position_embeddings:
                    raise ValueError("Input plus generation budget exceeds context; input is not truncated")
                encoded = torch.tensor([prefix], dtype=torch.long, device="cuda")
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = model.generate(input_ids=encoded, attention_mask=torch.ones_like(encoded),
                        do_sample=False, max_new_tokens=CONFIG["generation_max_new_tokens"], use_cache=True,
                        pad_token_id=tokenizer.eos_token_id)
                tokens = result[0, len(prefix):].tolist()
                raw = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                prediction = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                status = "ok" if tokens and tokens[-1] in eos_ids else "truncated"
                if status == "ok" and not prediction.strip():
                    status = "empty"
            except Exception as failure:
                if isinstance(failure, torch.cuda.OutOfMemoryError):
                    fatal = failure
                status = "error"
                error = {"type": type(failure).__name__, "message": str(failure)}
            failures += status != "ok"
            stream.write(serialize_jsonl([{"id": row["id"], "prediction": prediction, "raw_output": raw,
                "status": status, "generated_tokens": len(tokens), "generated_token_ids": tokens,
                "latency_ms": (time.monotonic() - started) * 1000, "error": error}]))
            stream.flush()
            if fatal is not None:
                raise fatal
            guard()
    state.update(failed_cases=failures, predictions_sha256=hash_file(output / "predictions.jsonl"),
                 generation=task["generation"], output_transform="none")


def main(argv=None, *, entrypoint=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train", "predict"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--heldout-proof", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--budget-root", type=Path, required=True,
                        help="Shared external directory accounting for all stages, caches and temporary files")
    args = parser.parse_args(argv)
    args.entrypoint = entrypoint or __file__
    try:
        result = execute(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"model-experiment: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(entrypoint=__file__))
