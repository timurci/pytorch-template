# pytorch-training-template

A template repository for PyTorch training projects: a small core library
split into independent layers that interact through protocols, wired together
at the entrypoints by one validated YAML config. Not specific to any task or
dataset — copy it, rename the package, and adapt the marked pieces.

The architecture is documented in [docs/architecture.md](docs/architecture.md)
— start there. The config pattern — every schema's shape and how to make a
new component (model, source, step, tracker) configurable — is
[docs/config-pattern.md](docs/config-pattern.md). In short:

| Layer | Role |
| --- | --- |
| `persistence` | Raw table readers (`CsvSource` / `ParquetSource`) + table/checkpoint I/O; caller supplies paths |
| `data` | Ingestion: `RawTableSource` / `DataSource` ports, `TableSchema` (column roles), `ProcessedSource` (applies the raw pipeline), `ValidatedTable` / `ValidatedSource`, the `Batch` contract, the torch-side `TableDataset`, split/target-statistics helpers; no file I/O |
| `features` | Preprocessing, split by file: `features.raw` — a `RawPipeline` of `RawStep`s over the named raw columns (selection, encoding) at read time — and `features.tensor` — `TensorStep` / `TensorPipeline`, stage-tagged, seeded batch steps applied on the target device |
| `models` | `nn.Module` architectures (reference: `MLPClassifier`) |
| `tracking` | `ExperimentTracker` protocol; `null` / `stdout` / `mlflow` adapters |
| `training` | `Trainer`: the epoch loop; optional epoch `LRScheduler`; applies the tensor pipeline per batch; trackers and checkpoint strategies injected |
| `cli` | Entrypoints (`template-train` / `template-infer`); the composition-root schema (`cli/config.py`) and wiring |

Preprocessing splits in two by *where it runs*. **Raw processing**
(`features.raw`) is a pipeline of steps over the named raw columns, applied
at read time on the CPU; it is the only place raw values and column names
are visible, so selection, renaming, deriving, and encoding live there.
**Tensor processing** (`features.tensor`) runs on the target device,
vectorized over a batch, during training and inference — prefer it for any
transform that can wait.

## Quickstart

```bash
uv sync
cp configs/example.yaml.example configs/example.yaml   # edit paths/columns
uv run template-train --config configs/example.yaml
uv run template-infer --config configs/example.yaml
```

Training writes the recovery checkpoint and, when configured, the best
checkpoint. Inference loads the best file when it exists, otherwise the
recovery file, and writes `inference.output_path` (`id,<target>` with
class-1 probabilities). Full field reference: [docs/cli.md](docs/cli.md).

## Using it as a template

1. Rename `src/template` (package), `name` / `[project.scripts]`
   (pyproject), and the `"template"` logger in `cli/runtime.py` — all
   greppable as `template`.
2. Adapt the marked layers to your task: `data.Batch` + the dataset adapter
   (+ the source ports if your raw form is not a table), your raw steps in
   `features/raw.py` and tensor steps in `features/tensor.py`, your models,
   your trainer specialization.
3. Give each new implementation an `XConfig` alongside its class (its module
   or the layer's `config.py`) and register it in the slot's `kind` union
   ([docs/config-pattern.md](docs/config-pattern.md) has recipes).

Step-by-step: [docs/architecture.md#instantiating-the-template](docs/architecture.md#instantiating-the-template).

## Development

```bash
uv run ty check
uv run ruff check src tests
uv run pytest
```
