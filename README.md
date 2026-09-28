# ReadCue

Small models for context-aware reading of informal text.

ReadCue aims to train and publish useful, reproducible small-model weights on Hugging Face. **No ReadCue checkpoint has been trained or published.**

## Lightweight development

Python 3.12 is the tested version. The core package has **no mandatory runtime dependencies**; inspecting evidence and running tests does not require CUDA or model downloads.

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install "setuptools>=68" wheel
python -m pip install --no-deps --no-build-isolation -e .
python -m readcue verify
python -m unittest discover -s tests -v
```

Run these commands from the checkout, or use `readcue --project /path/to/ReadCue ...`. This is a research harness; its historical evaluation/runtime assets live in the checkout. `fetch` downloads a baseline only when explicitly invoked; `run` uses the selected interpreter's optional model environment. Full GPU reproduction remains a separate setup.

- [Architecture and extension boundaries](docs/ARCHITECTURE.md)
- [Disk usage and cache recovery](docs/STORAGE.md)
- [Contribution guide](CONTRIBUTING.md) and [repository settings](docs/REPOSITORY.md)

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

## License

Original code and documentation are Apache-2.0. Original pilot scenarios have an explicit Apache-2.0 statement in their README. Third-party data, baseline models and future trained weights have separate terms; the repository license does not override them.

ReadCue is an independent project, not an official release from the cited authors.
