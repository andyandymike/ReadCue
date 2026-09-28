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

We are evaluating Chinese text normalization before committing to a training direction: turn clear written quantities into spoken forms while preserving meaning, tone, names and literal identifiers.

The first pilot compares rules, a published Chinese TN model, a general small language model and unchanged text. Its 72 scenarios are **assistant-authored, not human-reviewed, field-collected or a blind benchmark**. This evaluates normalized text, not polyphone accuracy or synthesized speech.

- [Completed local comparison and decision](reports/zh-pilot-v1/REPORT.md): existing TN matches 67/72 authored cases; this is not yet sufficient evidence to start training.
- [Reproduce the local pilot](docs/REPRODUCING_ZH_PILOT.md)
- [Project plan (Chinese)](docs/PROJECT_PLAN.md)
- [Evaluation protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Pilot cases and provenance](eval/zh_reading_pilot_v1/README.md)
- [Data and licensing](docs/DATA_AND_LICENSES.md)

## Research background

The project began with [UGTPhon](https://github.com/naver-ai/UGTPHON), covering English, Vietnamese and Korean. Its [paper, data and dictionary audits](research/ugtphon-2026-09-28/README.md) remain available. The original [Vietnamese candidate plan](docs/archive/VI_PROJECT_PLAN_2026-09-28.md) is historical.

Paper-reported results, dataset audits and actual measurements remain separate. Existing implementations are baselines, not proof that a new model is valuable.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`src/readcue/`](src/readcue/) | Core CLI, asset verification and experiment orchestration |
| [`tests/`](tests/) | Lightweight checks using local fixtures |
| [`scripts/`](scripts/) | Historical pilot entry points; new runs use the core CLI |
| [`tools/`](tools/) | Distribution and storage maintenance checks |
| [`eval/`](eval/) | Original pilot cases, provenance and frozen hashes |
| [`reports/`](reports/) | Recorded baseline outputs and analysis |
| [`research/`](research/) | Dated paper and dataset audits |
| [`docs/`](docs/README.md) | Current plan, protocol, reproduction and release guidance |

Start with the [contribution guide](CONTRIBUTING.md) for changes, [architecture](docs/ARCHITECTURE.md) for code boundaries, or [release guidance](docs/RELEASING.md) for source and future model releases. Local disk maintenance records are separate from the portable setup instructions.

## License

Original code and documentation are Apache-2.0. Original pilot scenarios have an explicit Apache-2.0 statement in their README. Third-party data, baseline models and future trained weights have separate terms; the repository license does not override them.

ReadCue is an independent project, not an official release from the cited authors.
