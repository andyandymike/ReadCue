# ReadCue working guidance

- Read `docs/PROJECT_PLAN.md`, `docs/EXPERIMENT_PROTOCOL.md`, and `docs/DATA_AND_LICENSES.md` before implementing training or changing scope.
- Keep three evidence categories distinct: paper-reported results, our data/dictionary audits, and measured ReadCue model results. No ReadCue checkpoint exists at project inception.
- Build inference inputs and hint dictionaries without gold evaluation labels. Gold labels may be used for training targets and offline scoring/grouping.
- The official UGTPhon test set was inspected during project selection; never describe it as an untouched blind test.
- Preserve original data, hashes, splits, and an explicit record of any normalization or annotation changes. Do not silently alter gold labels to improve scores.
- Keep raw datasets, weights, secrets, and paid-run outputs out of Git. Original project code licensing does not override upstream data or future model terms.
- The $200–500 estimate is planning, not permission to incur charges. Start paid work only within an explicitly authorized budget and retain run/cost records.
- Prioritize a reproducible small research release. Do not expand to a new architecture, more languages, or an app UI without a task-driven reason.
