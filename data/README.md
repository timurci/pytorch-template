# `data/` — a suggested layout

This is a **convention, not a mechanism**. Every path this template uses is
config-supplied — `data.train.path` / `data.test.path`,
`training.checkpoints[].path`, `inference.output_path`, `report.output_path`,
and a raw step's `levels_path` — and nothing in the library assumes a
directory exists or that a path lives here. Follow the layout below, or
ignore it entirely; it is only what `configs/example.yaml.example`
demonstrates.

```
data/
  raw/      # inputs as they arrive — never written by any command here
  model/    # checkpoints and exports; the run's model registry
  report/   # generated artifacts: predictions, reports
  feature/  # optional: read-time artifacts the config declares
```

## Why the split

One question decides where a file belongs: **can it be regenerated?**

| Directory | Written by | Regenerable | Back it up |
| --- | --- | --- | --- |
| `raw/` | you, or an upstream export | no — it is the source | yes |
| `model/` | `template-train` | no — retraining is not reproducibility | yes, or accept the retrain |
| `report/` | `template-infer`, `template-report` | yes — from `raw/` + `model/` + config | no; safe to wipe |
| `feature/` | tools outside the run | depends; treat it as an input | yes, if not reproducible |

- **`raw/`** holds immutable inputs: the train and test tables an experiment
  reads. No command in this repository writes here, so a changed file in
  `raw/` means the experiment's inputs changed — exactly what you want to
  notice.
- **`model/`** is the registry: the recovery and best checkpoints
  (`training.checkpoints[].path`) and any export you add. `template-train`
  writes it; `template-infer` and `template-report` read it. Keeping learned
  artifacts out of `raw/` keeps "what we read" and "what we learned"
  distinguishable in a listing.
- **`report/`** holds everything derived: `inference.output_path` (the
  `id,<target>` predictions) and `report.output_path` (the markdown
  analysis). Nothing here is needed to reproduce anything else, which is why
  it is the first directory to delete when a run goes stale.
- **`feature/`** is optional and only exists if you declare artifacts instead
  of inline values — `levels_path` pointing at a vocabulary too large for
  YAML, a feature-selection list, and so on. These are *inputs*: the raw
  pipeline never fits its own state (`features.raw` steps are declared, not
  trained), so whatever is here was produced outside the run and is consumed
  like any other input. Fitting and persisting preprocessing *inside* a run
  is out of scope for this template — see
  [../docs/architecture.md](../docs/architecture.md).

## What it buys the trackers

Trackers receive the flattened config as `params`, so a run's provenance is
already there: `data.train.path` and `data.test.path` name the dataset, and
the JSON-encoded `training.checkpoints` list names the model files. A shared
layout makes those lines read the same way across runs and gives the derived
artifacts one directory to collect from. Nothing in `tracking` depends on it.

## In the config

```yaml
data:
  train: {kind: csv, path: data/raw/train.csv}
  test:  {kind: csv, path: data/raw/test.csv}
training:
  checkpoints:
    - {kind: recovery, path: data/model/recovery.pt}
    - {kind: best, path: data/model/best.pt}
inference:
  output_path: data/report/predictions.csv
report:
  output_path: data/report/report.md
```

Only this README is tracked by git: `raw/`, `model/`, `report/`, and
`feature/` are run products, not sources.
