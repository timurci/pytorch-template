# CLI reference

Two console scripts, both driven by one YAML experiment config:

```bash
uv run template-train --config configs/example.yaml
uv run template-infer --config configs/example.yaml
```

| Command | Config sections used | Result |
| --- | --- | --- |
| `template-train` | `data`, `pipeline`, `model`, `optimizer`, `loss`, `training` | `training.checkpoint_path` (`state_dict`) |
| `template-infer` | `data`, `pipeline`, `model`, `inference` | `inference.output_path` (`id,<target>`) |

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

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `train_path` | path | required | used by `template-train` |
| `test_path` | path | required | used by `template-infer` |
| `id_column` | str | `id` | captured before the pipeline drops it, reassembled into the predictions |
| `target.column` | str | required | label column; also the prediction column name |
| `target.mapping` | map[str, int] | required | class indices; must cover exactly `0..n_classes-1` |

### `pipeline.steps`

Applied in order to the raw frame, before the dataset is built. Transforms
declare their state in the config and never fit from data; unknown columns or
values raise instead of being ignored.

| `kind` | Fields | Behavior |
| --- | --- | --- |
| `drop_columns` | `columns: [str]` | Drops columns; not invertible |
| `map_values` | `column: str`, `mapping: {str: int}` | Replaces categories with integers; mapping must be injective and cover all values |
| `scale_by_cap` | `column`, `cap`, `floor: 0.0` | `(x - floor) / (cap - floor)` |
| `log_scale_by_cap` | `column`, `cap` | `log(max(x, 1)) / log(cap)`; requires `cap > 1` |
| `one_hot` | `column`, `levels: [str]` | Emits `<column>__<level>` 0/1 columns and drops the original; levels must cover all values |

### `model`

| Field | Default | Notes |
| --- | --- | --- |
| `hidden_size` | 128 | width of each hidden block |
| `hidden_depth` | 2 | number of `Linear -> ReLU -> LayerNorm -> Dropout` blocks |
| `n_classes` | 2 | logits; pair with `cross_entropy` |
| `dropout` | 0.0 | dropout probability per hidden block, in `[0, 1)`; `0` disables |

`input_size` is inferred from the processed frame, so it is not configurable.

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
| `class_weights` | `null` | `"balanced"` weights each class by `N / (C * n_c)` from the train partition; applies to both train and validation loss |

### `training`

| Field | Default | Notes |
| --- | --- | --- |
| `epochs` | 10 | |
| `batch_size` | 4096 | train and validation loaders |
| `val_fraction` | 0.1 | shuffled train/val split; `null` disables validation |
| `seed` | 42 | split and DataLoader shuffle |
| `shuffle` | true | training loader only |
| `device` | `auto` | `auto` picks CUDA, then MPS, then CPU; or force `cpu` / `cuda` / `mps` |
| `trackers` | `[{kind: stdout}]` | see below |
| `augmenter` | `null` | train-time batch augmentation; see below |
| `track_gradients` | false | log `train/grad_norm`, the mean over batches of the total parameter gradient L2 norm, measured after backward and before the optimizer step |
| `checkpoint_path` | required | `state_dict` saved here when training ends |

### `training.augmenter`

Optional train-time augmentation, applied to each training batch on the
training device before the forward pass. Validation is never augmented.
Augmenters return labels aligned with the augmented rows plus the original
row each output row came from (`source_indices`); see
[architecture.md](architecture.md#augmenter) for the contract.

| `kind` | Fields | Behavior |
| --- | --- | --- |
| `identity` | — | Pass-through; useful as a placeholder while wiring a pipeline |
| `gaussian_noise` | `std` (required, > 0) | Adds iid Gaussian noise to features; labels and row count preserved |

```yaml
training:
  augmenter:
    kind: gaussian_noise
    std: 0.01
```

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
train target distribution: 0=900 (75.00%), 1=300 (25.00%)
val target distribution: 0=99 (73.88%), 1=35 (26.12%)
device=cpu rows=1200+134 epochs=10
params {'data.train_path': 'data/train.csv', ..., 'optimizer.kind': 'adamw', ...}
step=0 metrics {'train/loss': 0.2421, 'train/accuracy': 0.8897, ...}
step=0 metrics {'val/loss': 0.2379, 'val/accuracy': 0.8925, ...}
saved checkpoint to outputs/model.pt
```

`params` is the flattened config with dotted keys; lists such as
`pipeline.steps` are JSON-encoded. Classes with no predictions or no support
during an epoch log `0.0`, so trackers never see NaN. An inference run
prints:

```
rows=500
probability mean=0.241765 min=0.000055 max=0.929446
saved predictions to outputs/predictions.csv
```

## What the commands do

`template-train`:

1. `load_table(data.train_path)`.
2. Apply `pipeline`, then the target mapping.
3. `TableDataset.from_frame(..., target_column=...)`, then `split(val_fraction, seed)`; skipped when `val_fraction` is `null`. The per-class target distribution of each partition is logged.
4. Build train/validation `DataLoader`s and infer `input_size`.
5. Build `MLPClassifier` on `device`, the optimizer, the loss, and the augmenter; `loss.class_weights: balanced` resolves inverse-frequency weights from the train partition's class counts.
6. Open the trackers, log the flattened config as params, and run `Trainer` for `epochs`.
7. `save_checkpoint(model.state_dict(), checkpoint_path)`.

`template-infer`:

1. `load_table(data.test_path)` and keep `id_column` (row order is preserved).
2. Apply `pipeline` (the same steps as training, minus the target mapping).
3. Rebuild the model from the same config, `load_state_dict`, `eval()` on CPU.
4. `softmax(logits, dim=1)[:, 1]` over the test loader.
5. Optionally report and/or save `id,<target>`.

## Artifacts

- `training.checkpoint_path` is a plain `torch` `state_dict` with no
  architecture inside, so the inference config must describe the same model.
- `inference.output_path` is a CSV with the configured id column and the target
  column, holding probabilities of class `1` in `[0, 1]`.
- Both paths create parent directories automatically.
