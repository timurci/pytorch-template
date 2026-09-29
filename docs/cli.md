# CLI reference

Two console scripts, both driven by one YAML experiment config:

```bash
uv run template-train --config configs/example.yaml
uv run template-infer --config configs/example.yaml
```

| Command | Config sections used | Result |
| --- | --- | --- |
| `template-train` | `data`, `processing`, `model`, `optimizer`, `loss`, `training` | `training.checkpoint_path` (`state_dict`) |
| `template-infer` | `data`, `processing`, `model`, `inference`, `training.checkpoint_path` / `training.seed` | `inference.output_path` (`id,<target>`) |

Both commands validate the whole file, so a typo anywhere fails fast. Run
`uv run template-train --help` for the flags.

## Config file

- `-c/--config` is required and must point to a YAML mapping.
- Paths are relative to the working directory, not to the config file.
- Unknown fields are rejected, so misspelled keys are errors.
- Target mapping values must be exactly `0..model.n_classes-1`.

`configs/example.yaml.example` is the commented reference. `configs/*.yaml`
is gitignored, so copy the template before running:

```bash
cp configs/example.yaml.example configs/example.yaml
```

## Sections

### `data`

Column *roles* (what a column is) and *encodings* (how it becomes a tensor)
are declared here. Features are whatever remains after metadata, target, and
exclusions — declaring exclusions is enough. The resolved feature set and
width are logged at startup.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `train_path` | path | required | used by `template-train` |
| `test_path` | path | required | used by `template-infer` |
| `id_column` | str | `id` | always metadata: kept untouched and reassembled into the predictions (merged into `metadata`) |
| `target.column` | str | required | label column; also the prediction column name |
| `target.mapping` | map[str, int] | required | class indices; non-empty, injective, must cover exactly `0..n_classes-1` |
| `metadata` | [str] | `[]` | columns kept untouched as sample identity; must not overlap `target` / `exclude` |
| `exclude` | [str] | `[]` | dropped columns; must not overlap `metadata` / `target` |
| `encodings` | list | `[]` | feature encodings; see below |

A metadata column is never a feature; the target is never a feature
(leakage). A column cannot be both feature and metadata through this config
(the derived roles are disjoint); the library `TableSchema` allows that
duality via direct construction.

#### `data.encodings`

| `kind` | Fields | Behavior |
| --- | --- | --- |
| `one_hot` | `column`, `levels: [str]` or `levels_path` (exactly one) | One-hot block of width `len(levels)`; values outside the levels raise |

Feature columns without an encoding pass through numerically (width 1) and
must be numeric in the data (error otherwise). The target always uses
`target.mapping`. `levels_path` points at a vocab file with one level per
line, for vocabularies too large for the config.

### `processing`

One ordered list of tensor-only steps applied to every batch after the move
to the training device; preprocessing and augmentation interleave freely —
stacking is list order (`pre -> aug -> pre` needs no phases). Column names
resolve to feature blocks at composition, so a step on a non-feature column
fails at build. `stages` picks where a step runs:

| Stage | Runs on |
| --- | --- |
| `train` | training batches |
| `eval` | validation batches |
| `predict` | inference batches |

| `kind` | Fields | Behavior |
| --- | --- | --- |
| `scale_by_cap` | `column`, `cap`, `floor` (default `0.0`), `stages` | `(x - floor) / (cap - floor)`; requires `cap > floor` |
| `log_scale_by_cap` | `column`, `cap`, `stages` | `log(max(x, 1)) / log(cap)`; requires `cap > 1` |
| `gaussian_noise` | `std` (required, > 0), `stages` | iid Gaussian noise on all features; rows and labels preserved |

`stages` defaults to `[train, eval, predict]` for the deterministic scales
(predict-time scaling is part of the feature definition) and to `[train]`
for `gaussian_noise`. Steps never mutate their input, and row-changing steps
keep `source_indices` / `targets` aligned; see
[architecture.md](architecture.md#feature-processing) for the contract.

Seeded validation-time augmentation is a stage tag, not a code change:

```yaml
processing:
  steps:
    - {kind: gaussian_noise, std: 0.005, stages: [eval]}
```

Training/validation batches feed each step a generator seeded per
(`seed`, stage, epoch, batch), so a seeded run reproduces exactly — and
noise draws are identical on every device. At inference the pipeline runs at
stage `predict` with a generator seeded from `training.seed`.

### `model`

| Field | Default | Notes |
| --- | --- | --- |
| `hidden_size` | 128 | width of each hidden block |
| `hidden_depth` | 2 | number of `Linear -> ReLU -> LayerNorm -> Dropout` blocks |
| `n_classes` | 2 | logits; pair with `cross_entropy` |
| `dropout` | 0.0 | dropout probability per hidden block, in `[0, 1)`; `0` disables |

`input_size` is the schema's `feature_width` (known before any row is read),
so it is not configurable.

### `optimizer`

| Field | Default | Notes |
| --- | --- | --- |
| `kind` | `adamw` | `adam`, `adamw`, or `sgd` |
| `lr` | 1e-3 | learning rate |
| `weight_decay` | 0.0 | all kinds |
| `momentum` | 0.0 | `sgd` only |

### `loss`

| Field | Default | Notes |
| --- | --- | --- |
| `kind` | `cross_entropy` | the model returns logits |
| `class_weights` | `null` | `"balanced"` weights each class by `N / (C * n_c)` from the train partition (requires every class to have samples); applies to both train and validation loss |

### `training`

| Field | Default | Notes |
| --- | --- | --- |
| `epochs` | 10 | |
| `batch_size` | 4096 | train and validation loaders |
| `val_fraction` | 0.1 | seed-random train/val index partition; `null` disables validation |
| `seed` | 42 | partition, DataLoader shuffle, processing RNG streams (and the inference `predict` stream) |
| `shuffle` | true | training loader only |
| `device` | `auto` | `auto` picks CUDA, then MPS, then CPU; or force `cpu` / `cuda` / `mps` |
| `trackers` | `[{kind: stdout}]` | see below |
| `track_gradients` | false | log `train/grad_norm`, the mean over batches of the total parameter gradient L2 norm, measured after backward and before the optimizer step |
| `checkpoint_path` | required | `state_dict` saved here when training ends |

### `inference`

| Field | Default | Notes |
| --- | --- | --- |
| `batch_size` | 8192 | inference loader |
| `save` | true | write `output_path` |
| `report` | true | print row count and probability summary |
| `output_path` | `outputs/predictions.csv` | `id,<target column>` |

## Trackers and logging

`training.trackers` is a list; each entry has a `kind`.

| `kind` | Fields | Output |
| --- | --- | --- |
| `stdout` | `every_n_steps` (default `1`, min `1`) | one `params {...}` line, then per-split `loss`, `accuracy`, per-class `precision_<i>` / `recall_<i>` / `f1_<i>`, and `precision_macro` / `recall_macro` / `f1_macro` every `every_n_steps`-th epoch and on the final epoch; adds `train/grad_norm` when `training.track_gradients` is true |
| `null` | — | nothing |
| `mlflow` | `experiment_name` (default `template`), `run_name`, `tracking_uri` | params and per-epoch metrics in the active run |

The CLI owns the run lifecycle: before training it calls
`mlflow.set_tracking_uri` (when set), `mlflow.set_experiment`, and
`mlflow.start_run(run_name=...)`, then ends the run afterwards. Metrics are
logged with the epoch as `step`.

All human-facing output goes through stdlib logging. The CLI attaches a
message-only (`%(message)s`) handler for the `"template"` logger to stdout at
INFO, so the examples below keep their exact format.

A training run prints:

```
features: ('age', 'income', 'city') (width 5)
train target distribution: 0=900 (75.00%), 1=300 (25.00%)
val target distribution: 0=99 (73.88%), 1=35 (26.12%)
device=cpu rows=1200+134 epochs=10
params {'data.train_path': 'data/train.csv', ..., 'processing.steps': '[...]', ...}
step=0 metrics {'train/loss': 0.2421, 'train/accuracy': 0.8897, ...}
step=0 metrics {'val/loss': 0.2379, 'val/accuracy': 0.8925, ...}
saved checkpoint to outputs/model.pt
```

`features: ...` is the resolved feature set and width — logged on purpose, so
a stray or leaky column surfaces here (and in the tracked params) instead of
inside the model. `params` is the flattened config with dotted keys; lists
such as `processing.steps` are JSON-encoded. With
`loss.class_weights: balanced`, the resolved weights are logged and tracked
as `loss.class_weights_effective`. Classes with no predictions or no support
during an epoch log `0.0`, so trackers never see NaN. An inference run
prints:

```
rows=500
probability mean=0.241765 min=0.000055 max=0.929446
saved predictions to outputs/predictions.csv
```

## What the commands do

`template-train`:

1. `CsvSource(data.train_path)`; `TableSchema.from_columns` resolves roles
   and encodings from the raw columns (`metadata` + `id_column` as metadata,
   `exclude` and the target out, everything else features) and logs the
   resolved feature set and width. Bad declarations (unknown columns,
   overlapping roles, encodings for non-feature columns) fail here.
2. `ValidatedSource(raw, schema)` validates every read at the boundary.
3. `partition_indices(count, val_fraction, seed)`; skipped when
   `val_fraction` is `null` (all rows train). Each partition's per-class
   target distribution is logged.
4. Build `TableDataset` over the source and `Subset` train/validation
   `DataLoader`s; `input_size` is `schema.feature_width`.
5. Build `MLPClassifier` on `device`, the optimizer, the loss; the processing
   pipeline is built from `processing.steps` against the schema.
   `loss.class_weights: balanced` resolves inverse-frequency weights from
   the train partition's class counts.
6. Open the trackers, log the flattened config as params, and run `Trainer`
   for `epochs` with `processing=...` and `seed=...` (batches are processed
   per stage: `train` for training, `eval` for validation).
7. `save_checkpoint(model.state_dict(), checkpoint_path)`.

`template-infer`:

1. `CsvSource(data.test_path)`; the schema is built with
   `include_target=False` (the target is excluded when present, absent
   otherwise). Ids are read through the validated source as metadata: they
   ride through ingestion untouched and row order is preserved.
2. Rebuild the model from the same config, `load_state_dict`, `eval()` on
   CPU; build the processing pipeline from the same `processing.steps`.
3. Per batch: `processing.process(batch, stage="predict", rng=...)` (the
   generator is seeded from `training.seed`), then
   `softmax(logits, dim=1)[:, 1]`.
4. Optionally report and/or save `id,<target>`.

## Artifacts

- `training.checkpoint_path` is a plain `torch` `state_dict` with no
  architecture inside, so the inference config must describe the same model.
- `inference.output_path` is a CSV with the configured id column and the target
  column, holding probabilities of class `1` in `[0, 1]`.
- Both paths create parent directories automatically.
