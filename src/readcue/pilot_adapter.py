"""Explicit v1 execution identity, separate from generic inference inputs."""
import hashlib

BASELINES = ("identity", "wetext", "tn", "qwen")

# UTF-8 SHA-256 of SYSTEM_PROMPT in the frozen scripts/run_zh_baseline.py.
# This is an identity check, not another editable copy of the experiment prompt.
QWEN_PROMPT_SHA256 = "804fc10d7ef73e3bc9764c9681043faaf3e665585e7511da0adb10299ea2fdd1"


def validate_neural_configuration(name, metadata):
    expected = {"device": "cuda", "dtype": "torch.float16", "max_new_tokens": 160,
                "seed": 20260928, "enable_thinking": False,
                "decoding": "greedy" if name == "tn" else "sample_temperature0.7_top_p0.8_top_k20"}
    for field, value in expected.items():
        actual = metadata.get(field)
        if type(actual) is not type(value) or actual != value:
            raise ValueError("Neural configuration does not match the frozen pilot: " + field)
    prompt = metadata.get("prompt")
    if name == "tn":
        valid_prompt = prompt == "<|fim_prefix|>{text}<|fim_suffix|>"
    else:
        valid_prompt = (isinstance(prompt, str)
                        and hashlib.sha256(prompt.encode("utf-8")).hexdigest() == QWEN_PROMPT_SHA256)
    if not valid_prompt:
        raise ValueError("Neural configuration does not match the frozen pilot: prompt")
    compatibility = metadata.get("loading_compatibility")
    if not isinstance(compatibility, dict):
        raise ValueError("Missing neural loading compatibility record")
    for field in ("full_vocabulary_ids_verified", "added_token_flags_and_fim_markers_verified", "inference_use_cache"):
        if compatibility.get(field) is not True:
            raise ValueError("Neural loading configuration does not match the frozen pilot: " + field)
    if compatibility.get("serialized_use_cache") is not (name == "qwen"):
        raise ValueError("Neural loading configuration does not match the frozen pilot: serialized_use_cache")
    if type(compatibility.get("effective_rope_theta")) not in (int, float) or compatibility["effective_rope_theta"] != 1000000:
        raise ValueError("Neural loading configuration does not match the frozen pilot: effective_rope_theta")
