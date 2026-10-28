# ReadCue

Small models for context-aware reading of informal text.

ReadCue aims to train and publish useful, reproducible small-model weights on Hugging Face. **The second local ReadCue LoRA adapter has been trained, independently reloaded and evaluated, including independent blinded assistant review. Weights have not been publicly uploaded.** On 45 assistant-authored synthetic cases, v2 matched 32 references versus 28 for v1 and 25 for fixed rules. One definite semantic error and one unresolved semantic case remain in v2; these text results do not establish speech-quality improvement. See the [v2 experiment record](docs/MODEL_EXPERIMENT_RUN_V2.md) and [local model card](docs/MODEL_CARD_V2.md). The earlier 55-case experiment and its original results remain in the [v1 record](docs/MODEL_EXPERIMENT_RUN_V1.md).

## Lightweight development

A [local v2 research candidate](docs/MODEL_RELEASE_CANDIDATE_V2.md) now provides the unchanged selected adapter, a standalone inference entry point, pinned base identity and integrity checks. CPU loading and a three-input GPU comparison against the original runner passed. The separate [development diagnosis](docs/MODEL_DEVELOPMENT_DIAGNOSTIC_V2.md) records 133/166 training and 29/53 validation reference matches; these are fitted and checkpoint-selection data, not an independent test.

Use Python 3.10 or newer; Python 3.12 is recommended for the recorded experiments. The core package has **no mandatory runtime dependencies**; inspecting evidence and running tests does not require CUDA or model downloads.

```sh
git clone https://github.com/andyandymike/ReadCue.git
cd ReadCue
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then:

```sh
python -m pip install "setuptools>=77" wheel
python -m pip install --no-deps --no-build-isolation -e .
python -m readcue verify
python -m unittest discover -s tests -v
```

Run these commands from the checkout. From elsewhere, use `readcue --project /absolute/path/to/ReadCue ...` and absolute paths for inputs, predictions and outputs: `--project` locates experiment assets but does not change how other relative paths are resolved. The installed wheel contains the core tools; historical experiment assets still require a checkout. `fetch` explicitly downloads a baseline; model inference requires a separately prepared environment.

Try the unchanged-text baseline without a GPU or downloads:

```sh
python -m readcue run --inputs eval/zh_reading_pilot_v1/inputs.jsonl --baseline identity --device cpu --output runs/identity-demo
python -m readcue score --cases eval/zh_reading_pilot_v1/cases.jsonl --predictions runs/identity-demo/predictions.jsonl --output runs/identity-demo/score.json
```

Expected normalized sentence matches: **32/72** (`core_tn` 0/24, `context_reading` 8/24, `preservation` 24/24). This baseline simply returns its input; it is not a trained ReadCue model. The score file contains the total, and the command prints per-track results. Choose a new output directory when repeating the demo; existing results are preserved.

## Current work

The v2 local run is complete: 166 training / 53 validation examples, three epochs, 63 updates, and an independently reloaded adapter selected by validation loss. On the same 45 newly frozen synthetic cases, v2 matched 32 references, the original v1 adapter matched 28, and fixed rules matched 25. Relative to v1, v2 gained four exact matches with no regressions; preservation errors and reading omissions remain. The earlier WSL access blocker was resolved in the continuation session without changing system permissions or training settings. See the [v2 protocol](docs/MODEL_EXPERIMENT_V2.md) and [run record, independent review and limitations](docs/MODEL_EXPERIMENT_RUN_V2.md). These are text results, not evidence of improved speech quality; weights remain local and external cost is zero.

The current milestone is a bounded Chinese small-model research experiment: learn context-dependent reading transformations while preserving the rest of the input. The task remains full-sentence text normalization. The first local run used 118 training and 29 validation examples, completed three epochs and 45 optimizer updates, and independently reloaded the selected adapter. Comparison and analysis are recorded separately from this training result.

The recent rules/TTS exploration has been reviewed and withdrawn as the ongoing workstream. Its frozen versions remain reproducible comparisons. Demonstrating an overall TTS improvement is not a prerequisite for discussing a first training experiment; data suitability, isolated evaluation and a bounded run must be established first. Negative results are valid research outcomes.

- [Current project plan](docs/PROJECT_PLAN.md) and [first model experiment](docs/MODEL_EXPERIMENT_V1.md): task contract, data roles, comparisons and completion conditions.
- [First local model run](docs/MODEL_EXPERIMENT_RUN_V1.md): fixed base, data scope, environment, adapter selection and current comparison status. `scripts/run_model_experiment.py` provides `preflight`, `train` and `predict`; `tools/score_model_experiment.py` scores saved outputs offline after selection. These optional model commands do not add mandatory ML dependencies to the core package.
- [Review and corrective actions](docs/WORK_REVIEW_2026-10-01.md): what was withdrawn, corrected, retained and still needs evidence.
- [Data and licensing](docs/DATA_AND_LICENSES.md): human review does not resolve source rights or approve model publication.
- [Data preparation and review](docs/DATA_PIPELINE.md), [annotation policy](docs/AI_PREANNOTATION.md) and [local workbench](docs/REVIEW_WORKBENCH.md): user-approved functionality remains available.

Historical evidence, with original scopes and limitations:

- [72-case pilot](reports/zh-pilot-v1/REPORT.md), [reproduction](docs/REPRODUCING_ZH_PILOT.md) and [frozen protocol](docs/EXPERIMENT_PROTOCOL.md): assistant-authored development scenarios, not a blind benchmark.
- [138 reviewed comments](reports/bilibili-data-v1/BASELINE_REPORT.md): exploratory comparisons of existing systems under the selected reading policy.
- [Comment rules](reports/bilibili-data-v1/PIPELINE_REPORT.md), [reading rules/TTS diagnostics](reports/bilibili-data-v1/READING_REPORT.md), [context rules](reports/bilibili-data-v1/CONTEXT_REPORT.md), [semantic rules](reports/bilibili-data-v1/SEMANTIC_REPORT.md): development improvements and known limitations of fixed baselines, not trained ReadCue results.

Historical recommendations to keep extending rules or to require stable TTS gains before a training experiment are superseded by the current plan. Original data, scores and frozen protocols are retained.

## Research background

The project began with [UGTPhon](https://github.com/naver-ai/UGTPHON), covering English, Vietnamese and Korean. Its [paper, data and dictionary audits](research/ugtphon-2026-09-28/README.md) remain available. The original [Vietnamese candidate plan](docs/archive/VI_PROJECT_PLAN_2026-09-28.md) is historical.

Paper-reported results, dataset audits and measured ReadCue results remain separate. Historical implementations remain baselines; successful training alone does not prove that the new adapter is useful.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`src/readcue/`](src/readcue/) | Core CLI, asset verification and experiment orchestration |
| [`tests/`](tests/) | Lightweight checks using local fixtures |
| [`configs/`](configs/) | Portable source hashes and sampling settings; no raw datasets |
| [`scripts/`](scripts/) | Frozen pilot, reviewed-comment runners and the separate bounded model experiment |
| [`tools/`](tools/) | Distribution and storage maintenance checks |
| [`eval/`](eval/) | Original pilot cases, provenance and frozen hashes |
| [`reports/`](reports/) | Recorded baseline outputs and analysis |
| [`research/`](research/) | Dated paper and dataset audits |
| [`docs/`](docs/README.md) | Current plan, protocol, reproduction and release guidance |

Start with the [contribution guide](CONTRIBUTING.md) for changes, [architecture](docs/ARCHITECTURE.md) for code boundaries, or [release guidance](docs/RELEASING.md) for source and future model releases. The [storage guide](docs/STORAGE.md) documents model/runtime locations, shared experiment caches and retention rules; machine-specific maintenance records are separate from the portable setup instructions.

## License

Original code and documentation are Apache-2.0. Original pilot scenarios have an explicit Apache-2.0 statement in their README. Third-party data and base models retain their own terms; the [data and licensing record](docs/DATA_AND_LICENSES.md) separates the local adapter's training provenance from the conditions for a public release. The repository license does not override upstream terms.

ReadCue is an independent project, not an official release from the cited authors.
