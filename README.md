# ReadCue

Small models for context-aware reading of informal text.

ReadCue aims to train and publish useful, reproducible small-model weights on Hugging Face. **No ReadCue checkpoint has been trained or published.**

## Lightweight development

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

The current milestone is a bounded Chinese small-model research experiment: learn context-dependent reading transformations while preserving the rest of the input. The task remains full-sentence text normalization. There is no ReadCue checkpoint yet.

The recent rules/TTS exploration has been reviewed and withdrawn as the ongoing workstream. Its frozen versions remain reproducible comparisons. Demonstrating an overall TTS improvement is not a prerequisite for discussing a first training experiment; data suitability, isolated evaluation and a bounded run must be established first. Negative results are valid research outcomes.

- [Current project plan](docs/PROJECT_PLAN.md) and [first model experiment](docs/MODEL_EXPERIMENT_V1.md): task contract, data roles, comparisons and completion conditions.
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

Paper-reported results, dataset audits and actual measurements remain separate. Existing implementations are baselines, not proof that a new model is valuable.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`src/readcue/`](src/readcue/) | Core CLI, asset verification and experiment orchestration |
| [`tests/`](tests/) | Lightweight checks using local fixtures |
| [`configs/`](configs/) | Portable source hashes and sampling settings; no raw datasets |
| [`scripts/`](scripts/) | Frozen pilot and separately versioned reviewed-comment experiment runners |
| [`tools/`](tools/) | Distribution and storage maintenance checks |
| [`eval/`](eval/) | Original pilot cases, provenance and frozen hashes |
| [`reports/`](reports/) | Recorded baseline outputs and analysis |
| [`research/`](research/) | Dated paper and dataset audits |
| [`docs/`](docs/README.md) | Current plan, protocol, reproduction and release guidance |

Start with the [contribution guide](CONTRIBUTING.md) for changes, [architecture](docs/ARCHITECTURE.md) for code boundaries, or [release guidance](docs/RELEASING.md) for source and future model releases. Local disk maintenance records are separate from the portable setup instructions.

## License

Original code and documentation are Apache-2.0. Original pilot scenarios have an explicit Apache-2.0 statement in their README. Third-party data, baseline models and future trained weights have separate terms; the repository license does not override them.

ReadCue is an independent project, not an official release from the cited authors.
