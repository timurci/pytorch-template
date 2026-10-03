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
| `data` | Ingestion: `RawTableSource` / `DataSource` ports, `TableSchema` (column roles + encodings), `ValidatedTable` / `ValidatedSource`, the `Batch` contract, and the torch-side `TableDataset` adapter; no file I/O |
| `features` | Feature processing: `ProcessingStep` / `ProcessingPipeline` — tensor-only preprocessing and augmentation in one stage-tagged, seeded machinery |
| `models` | `nn.Module` architectures (reference: `MLPClassifier`) |
| `tracking` | `ExperimentTracker` protocol; `null` / `stdout` / `mlflow` adapters |
| `training` | `Trainer`: the epoch loop; optional epoch `LRScheduler`; applies the processing pipeline per batch; trackers injected |
| `cli` | Entrypoints (`template-train` / `template-infer`); the composition-root schema (`cli/config.py`) and wiring |

## Quickstart

```bash
uv sync
cp configs/example.yaml.example configs/example.yaml   # edit paths/columns
uv run template-train --config configs/example.yaml
uv run template-infer --config configs/example.yaml
```

Training saves `training.checkpoint_path`; inference writes
`inference.output_path` (`id,<target>` with class-1 probabilities). Full
field reference: [docs/cli.md](docs/cli.md).

## Using it as a template

1. Rename `src/template` (package), `name` / `[project.scripts]`
   (pyproject), and the `"template"` logger in `cli/runtime.py` — all
   greppable as `template`.
2. Adapt the marked layers to your task: `data.Batch` + the dataset adapter
   (+ the source ports if your raw form is not a table), your models, your
   processing steps, your trainer specialization.
3. Give each new implementation an `XConfig` in its layer's `config.py` and
   register it in the slot's `kind` union
   ([docs/config-pattern.md](docs/config-pattern.md) has recipes).

Step-by-step: [docs/architecture.md#instantiating-the-template](docs/architecture.md#instantiating-the-template).

## Development

```bash
uv run ty check
uv run ruff check src tests
uv run pytest
```
