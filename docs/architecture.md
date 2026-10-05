# Architecture

A layered template for PyTorch training projects. The goal is not a framework
but a *shape*: independent library layers that interact through
protocols/abstract classes at most, wired together at the entrypoints from a
single declarative config.

Two principles do most of the work:

1. **Layers depend on contracts, not on each other's implementations.**
   Cross-layer references are `typing.Protocol`s (`RawTableSource`,
   `DataSource`, `RawStep`, `TensorStep`, `ExperimentTracker`), `torch`
   abstractions
   (`nn.Module`, `Optimizer`, `LRScheduler`, `DataLoader`), or one shared
   data contract (`Batch`). Swapping an implementation — a tracker, a
   raw or tensor step, a model — touches no other layer.
2. **Config is the composition root.** One YAML file describes the whole
   experiment; Pydantic schemas validate it and `build()` plain library
   objects. Each schema lives in the layer of the class it builds; the CLI
   assembles them into `ExperimentConfig` and does the wiring. Library
   objects never see config objects — they take plain constructor
   arguments, which keeps them testable and reusable from notebooks,
   scripts, or services without the CLI.

The structure resembles an FTI (feature / training / inference) pipeline:
ingestion plus preprocessing and their config are the *feature pipeline*,
training is the *training pipeline*, and the inference entrypoint is the
*inference pipeline* — all three sharing one config so what trains is what
serves.

## Layers

| Layer | Role | Owns | Must not |
| --- | --- | --- | --- |
| `persistence` | Raw table readers + persistent I/O | `CsvSource` / `ParquetSource` (raw, unvalidated rows by index), `save_table`, `save_checkpoint` / `load_checkpoint`, and their `config.py` schemas | Know about validated forms, datasets, models, or training; validate structure; pick its own paths (callers supply them) |
| `data` | Ingestion + the torch-side dataset adapter | `RawTableSource` / `DataSource` ports; `TableSchema` (column roles); `ProcessedSource`, `ValidatedTable` / `ValidatedSource`; `Batch` TypedDict; `TableDataset`; `partition_indices` / `class_counts` | File I/O (rows arrive through the ports); import any sibling layer except `features.raw` (whose pipeline it applies) |
| `features` | Preprocessing, raw and tensor, one file each | `features/raw.py`: `RawStep` protocol + `RawPipeline` over named raw columns (`OneHot`, `DropColumns`), the `TargetEncoder` contract, and their schemas. `features/tensor.py`: `TensorStep` protocol + `TensorPipeline` (ordered, stage-tagged steps over batches), `ScaleByCap`, `LogScaleByCap`, `GaussianNoise`, and their schemas | File I/O; fitting state from data; raw forms or column names in tensor steps (config declares names, resolved at build) |
| `models` | Network definitions | `nn.Module` subclasses, e.g. `MLPClassifier`; `models/config.py` schemas | Anything but `torch` (its `config.py` may add `pydantic`) |
| `tracking` | Observability port | `ExperimentTracker` protocol; `Null` / `Stdout` / `MLflow` adapters, and their `config.py` schemas | Owning external lifecycles (the CLI opens/closes the MLflow run) |
| `training` | The training loop and checkpoint strategies | `Trainer`, `CheckpointStrategy`, the recovery and best strategies, and their `config.py` schemas | Touching sources or splitting data, building models/loaders/schedulers, choosing checkpoint paths, resuming a run, opening tracker runs |
| `cli` | Entrypoints / composition root | The composition-root schema (`cli/config.py`: `ExperimentConfig`, `DataConfig`, `TrainingConfig` / `InferenceConfig` / `ReportConfig`, optimizer/scheduler/loss schemas) and the `build()` wiring, `template-train`, `template-infer`, `template-report` | Contain business logic — it only wires layers |

## Dependency rules

Library layers import nothing from sibling layers except the explicitly
allowed contract imports. The CLI imports everything and composes.
(`features` and `data` exchange one edge each — `features.tensor` takes
`data.Batch`, `data` takes `features.raw`'s encoders — so treat the pair as
one coupled unit rather than a strict order.)

```
                    ┌─────────────┐
                    │     cli     │  (imports all layers; composes)
                    └──────┬──────┘
                           │
   ┌───────────┬───────────┼───────────┬──────────────┐
   │           │           │           │              │
persistence  data ◄─── features     models        tracking
   │           │           │           │              │
   └───────────┴─────┬─────┴───────────┴──────────────┘
                     │ (contracts only)
                  training
```

The allowed edges, exactly:

- `features.tensor` → `data`: **`Batch` only** — steps consume and return
  the batch contract; that single edge is why the batch is owned by `data`
  (below). Step schemas resolve column names through `BlockResolver`
  (`features/tensor.py`), structurally satisfied by `data.TableSchema`,
  so config adds no import edge. `features.raw` imports nothing from
  siblings.
- `data` → `features.raw`: `TableSchema` types against the `TargetEncoder`
  contract, and `ProcessedSource` / `TableDataset` apply the `RawPipeline`
  and `encode_target` defined there. This is the one edge that points *into*
  the feature layer, and it is acyclic: `features.raw` imports nothing, so
  the `data` ↔ `features` pair only looks circular at the package level.
- `training` → `data.Batch`, `features.tensor.TensorPipeline` /
  `features.tensor.Stage`, `tracking.ExperimentTracker`,
  `persistence.save_checkpoint` / `load_checkpoint`, plus `torch`'s own
  `nn.Module`, `Optimizer`, `LRScheduler`, and `DataLoader`. Persistence
  still stores a caller-supplied mapping; the training layer owns the
  checkpoint record.
- `persistence` imports nothing from sibling layers: raw readers are plain
  classes whose methods (`columns`, `count`, `read(indices) -> pl.DataFrame`)
  satisfy `data.RawTableSource` *structurally*, so no import edge exists.
  `data`'s remaining modules import only each other (and `polars` /
  `torch`).
- `models` and `tracking` import nothing from sibling layers.
- `evaluation` (`template/evaluation.py`) is a leaf like `config_pattern`:
  `torch` only, no sibling layer imports. `template-report` imports it; no
  layer does.

Everything else (constructing the model, the optimizer, the scheduler, the
loaders, the trackers, the tensor pipeline, resuming from a checkpoint,
choosing which file inference loads, computing the report's metrics, and
writing its artifact) happens in the entrypoint and is passed in. Checkpoint
strategies, passed into the trainer, decide when to write.

## The batch contract

`Batch` is the one concrete, non-protocol type that crosses layer
boundaries:

```python
class Batch(TypedDict):
    features: Tensor                    # (N, F) float
    targets: NotRequired[Tensor]        # (N,) long class indices
    source_indices: NotRequired[Tensor]  # (N,) pre-processing row per row
```

The training loop is annotated `DataLoader[Batch]`. It is owned by the
**`data` layer**, not by training: datasets produce it, processing
steps and the trainer consume it. `targets` is absent for unlabeled
(inference) data. `source_indices` appears after tensor processing: the
pipeline guarantees it on its output (absent input provenance becomes
`arange(N)`), and steps that change rows keep it aligned with `features` —
that is the provenance for row-changing steps and for consumers that need
sample identity (e.g. contrastive pairs).

Why not a training-owned port? The purer dependency-inversion alternative is
to let `training` define the batch contract and have `data` and `features`
adapt to it. We reject it here on purpose: a training loop's semantics are
inseparable from the shape of its data and task anyway (this trainer computes
classification metrics; a different task rewrites the loop), so abstracting
the batch buys indirection without decoupling. If you ever need two
incompatible batch shapes in one project, that is the signal to revisit this
decision — define a `Batch` protocol in `training` and let the datasets
conform.

Tasks whose samples are not `(features, targets)` pairs redefine `Batch`
here in the data layer; the processing steps, trainer, and entrypoints
follow.

## Ingestion: raw processing, column roles, tensorization

Ingestion (raw rows → tensor batches) is the `persistence` readers, the
`data` contracts, and the raw pipeline in `features.raw`. It is three steps
behind one seam: raw rows arrive through a port, a `RawPipeline` transforms
the named columns, and a `TableSchema` declares the roles that tensorize.

```python
class RawTableSource(Protocol):          # the raw side, satisfied structurally
    columns: Sequence[str]
    def count(self) -> int: ...
    def read(self, indices: Sequence[int]) -> pl.DataFrame: ...

class DataSource(Protocol[T_co]):        # the validated side
    schema: TableSchema                  # declared minimal guarantee of every read
    def count(self) -> int: ...
    def read(self, indices: Sequence[int]) -> T_co: ...
```

- **Raw readers** (`persistence`: `CsvSource`, `ParquetSource`) load a
  whole table at construction and serve row slices by
  index. They are deliberately unvalidated — roles and raw processing would
  couple them to declarations. Lazy readers (e.g. `pl.scan_csv`) are a
  documented pattern, not shipped: a polars lazy scan re-executes its query
  on every `read`, so per-batch reads only pay off on indexed sources such as
  a database.
- **`RawPipeline`** (`features.raw`) is an ordered list of `RawStep`s
  applied to every read, before any tensor exists. A raw step sees the named
  columns and the raw values, so feature selection (`DropColumns`), renaming,
  deriving, and encoding (`OneHot`) are all raw steps; rows and their order
  are preserved. Each step declares its output column names from its input
  names, so the pipeline resolves its output layout — and the model's input
  width — without reading a row. Steps are declared state, never fit from the
  data. `ProcessedSource` wraps a raw reader with a pipeline; its `columns`
  are the pipeline's declared output.
- **`TableSchema`** declares what each *processed* column *is* (a role), over
  columns `features.raw` has already made numeric:
  - **Roles are projections, not partitions.** `metadata_columns` identify
    samples and are kept untouched; `target_column` is the supervised target
    and is never a feature (that would be leakage — the schema rejects it);
    `TableSchema.from_columns` takes `exclude` columns and derives the
    features as whatever remains, keeping processed order — declaring
    exclusions
    is enough, the feature set needs no list of its own. A column MAY be
    both feature and metadata (e.g. an id that is also a model input) — it
    then appears in the feature block and in the metadata side; that duality
    needs direct construction, since `from_columns` derives disjoint roles.
  - **The target** is the one role-level encoding: `MapValues`
    (`features.raw`) maps declared class indices, so train and inference
    encode identically; a target without an encoder must be numeric.
  - **Feature columns are numeric and one tensor column wide**, so
    `feature_width` is `len(feature_columns)` — known before any row is read
    (raw steps that change the layout declare it statically).
    `feature_slice(column)` resolves a name to its position in
    `feature_columns` order.
- **`ValidatedSource`** wraps a processed source with a schema and is the
  `DataSource[ValidatedTable]` adapter: every read is checked at the
  boundary, exactly once. `ValidatedTable` enforces the schema's roles —
  required columns present, feature columns numeric, the target numeric
  unless a `target_encoder` maps it. Extra columns are ignored, so auxiliary
  columns ride through untouched.
- **`TableDataset`** is the only torch-aware piece: the adapter from
  `DataSource[ValidatedTable]` to `Batch` samples. It stacks the numeric
  feature columns in `feature_columns` order into the `features` tensor and
  maps the target through `encode_target`. It implements `__getitem__`
  (per-item fallback) and `__getitems__` (fetches a whole batch's rows with a
  single `read`, which the `DataLoader` uses when present), and nothing else.
- **Random access is mandatory.** Map-style datasets need index-addressable
  rows; pure streams must adapt (materialize a buffer) at the boundary or
  are out of scope for this port.

Splits and target statistics are index-based, so they work identically for
any source and neither the dataset nor the sources know a split exists:
`partition_indices(count, val_fraction, seed=...)` returns disjoint index
sets (seed-random, covering all rows), stock `torch.utils.data.Subset` wraps
the dataset (it forwards `__getitems__`, keeping batch fetch under splits),
and `class_counts(source, indices)` reads the target through the port.

## Preprocessing: raw and tensor

The feature layer holds both kinds of preprocessing, one per file. They are
the same idea — data changing shape before the model — split by *where they
run*:

- **Raw processing** (`features/raw.py`) runs where the raw form exists: a
  `RawPipeline` of `RawStep`s over the named `polars` columns, applied per
  read on the CPU, before any `Batch` exists. It is the only place raw values
  and column names are visible, and it is declared state — steps are never
  fit from the data they process.
- **Tensor processing** (`features/tensor.py`) runs on the target device: a
  `TensorStep` receives a batch of tensors and returns a batch of tensors.
  Raw forms and column names never appear here; names resolve to
  feature-tensor blocks (`feature_slice`) at composition time, so one step
  works regardless of how the data was ingested.

**Prefer tensor processing** for any transform that can wait. It is
vectorized over a whole batch, runs on the accelerator the model already
uses, and draws from a seeded RNG, so it is both cheaper than per-row CPU
work and reproducible. Reach for raw processing only when the raw form is
required (e.g. one-hot expansion of strings, or name-based feature selection)
or the work must happen once at read time rather than every epoch.

### The raw step contract

```python
class RawStep(Protocol):
    def columns(self, inputs: Sequence[str]) -> tuple[str, ...]: ...  # layout
    def process(self, frame: pl.DataFrame) -> pl.DataFrame: ...       # rows kept
```

A raw step declares its output columns from its input names, so the pipeline
(and the schema) resolves the layout without reading rows. `DropColumns` and
`OneHot` are the shipped steps; `OneHot` replaces a categorical column with
`f"{column}={level}"` float columns, so every level is an ordinary numeric
column later steps and roles can name.

### The tensor step contract

```python
class TensorStep(Protocol):
    def process(self, batch: Batch, rng: Generator) -> Batch: ...
```

Invariants (normative, see `features/tensor.py`):

- Steps never mutate their input; they return a `Batch` (cloning on write).
  Fields a step does not touch pass through unchanged.
- Deterministic steps ignore `rng`; stochastic steps MUST draw from `rng`
  (never the global RNG) so a seeded run reproduces exactly.
- Steps that change the row count keep `source_indices` and `targets`
  aligned with `features` (label-preserving steps satisfy
  `targets == input_targets[source_indices]`).

### Composition and stages

`TensorPipeline` applies an ordered list of `(step, stages)` pairs to one
batch at a time. `Stage` is `"train" | "eval" | "predict"`; a step runs in
the stages it is tagged with. Order is construction order: interleaving
deterministic preprocessing with stochastic augmentation (`pre -> aug -> pre`)
is just list order — there is no separate preprocessing and augmentation
phase. The pipeline guarantees `source_indices` on its output. Every step of
one pass sees the same `rng`, so one seed reproduces the whole pass.

Why is preprocessing/augmentation one machinery? Because the two differ in
exactly two respects — **stage tags** (augmentation is skipped where
preprocessing runs, e.g. at predict) and **rng use** (augmentation draws,
preprocessing doesn't). Everything else — tensor-only contract, block
resolution, composition, seeding, provenance — is shared, so splitting them
would duplicate the machinery to express those two flags.

### Shipped steps

| Step | Behavior | Default stages |
| --- | --- | --- |
| `ScaleByCap(block, cap, floor=0.0)` | `(x - floor) / (cap - floor)` on one feature block; requires `cap > floor` | all (incl. `predict`) |
| `LogScaleByCap(block, cap)` | `log(clamp(x, min=1)) / log(cap)` on one feature block; requires `cap > 1` | all (incl. `predict`) |
| `GaussianNoise(std)` | iid Gaussian noise on all features; rows and labels preserved; `std > 0` | `train` only |

Deterministic steps default to every stage: predict-time scaling is part of
the feature definition, not a training trick. Blocks are
`TableSchema.feature_slice` slices resolved when the pipeline is built.

### Seeding and validation-time augmentation

Tagging a stochastic step `stages: [eval]` (or `[train, eval]`) gives seeded
validation-time augmentation — a tagging choice, not a code change. The
trainer runs val batches at stage `eval` and gives every batch its own
generator seeded from `(seed, stage, epoch, batch)`, so a seeded run
reproduces exactly. Generators are CPU-side and steps move drawn tensors to
the batch's device, so the same seed produces the same noise on every
device. At inference the pipeline runs at stage `predict` with one generator
seeded from `training.seed`.

## The model port

Layers that use a model type it as plain `torch.nn.Module` — the trainer
only needs `forward`, `train()`/`eval()`, and `parameters()`, and staying on
the `torch` abstraction means zero coupling to the `models` layer.

Escalate only when the loop genuinely needs richer behavior: define an ABC
that **still inherits `nn.Module`** and adds the required methods, e.g. a
multi-task model whose `forward` returns a structured output, or a model
exposing `embeddings()` for metric learning. Never type against a bare
(non-`nn.Module`) Protocol for models — checkpoints, `.to(device)`, and
`state_dict` must keep working. Plain `nn.Module` remains the default until a
second method is actually required.

## Tracking

```python
class ExperimentTracker(Protocol):
    def log_params(self, params: Mapping[str, object]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None: ...
```

Trackers are passive observers: cheap, non-failing, and unaware of each
other. The trainer fans metrics out to a sequence of them. External
lifecycles belong to the entrypoint — the CLI opens the MLflow run around
training; `MLflowTracker` just logs into the active run. New backends (W&B,
TensorBoard, a database) are added by implementing the two methods and
registering a new `kind` in `tracking/config.py`'s union.

## Training

`Trainer(model, optimizer, scheduler=None, device=...)` owns exactly one
thing: the train/validation epoch loop — moving batches to the device,
applying the tensor pipeline, forward/backward/step, computing metrics,
fanning out to trackers. The pipeline runs on each batch *after* the move
to `device`: training batches at stage `train`, validation batches at stage
`eval` (so seeded validation-time augmentation is a tagging choice in the
pipeline, not trainer code). Processed features and labels replace the
batch — metrics therefore describe the processed data.

An optional `torch.optim.lr_scheduler.LRScheduler` is stepped once after
each epoch, after validation when there is a validation loader, with no
arguments — the call site in PyTorch's own scheduler examples. The
scheduler's unit is that call, so `step_size`, milestones, and `T_max`
count epochs. `None` leaves the optimizer's learning rate unchanged. The
scheduler must wrap the trainer's optimizer. `ReduceLROnPlateau` is
rejected at construction: its `step` requires a metric. Iteration-level
schedules (`OneCycleLR`, `CyclicLR`) are out of scope for the same reason
— the loop has one cadence, the epoch. Each train epoch logs `train/lr`,
the first param group's rate during that epoch, before the scheduler steps.

Checkpoint strategies are called once that step has finished. Each receives
the same training state — the completed epoch, the model weights, the
optimizer state, and the scheduler state or none — and, separately, that
epoch's metrics. Metrics are not part of the training state: a resumed run
does not restore them. `Trainer.state` is the record from the last completed
epoch and raises before one exists. `train(..., start_epoch=)` continues at
that index; `epochs` stays the original total, so the loop is
`range(start_epoch, epochs)`. The shipped strategies are recovery (the whole
training state, every N completed epochs and always the final epoch) and
best (the lowest `val/loss`, or `train/loss` when validation did not run).

Explicit non-responsibilities, all handled by the caller:

- no train/val splitting (`partition_indices` + `Subset` do it),
- no data access (it sees `DataLoader[Batch]`, never a source),
- no model/optimizer/scheduler/loss/loader construction,
- no opening checkpoint files and no choice of paths (strategies write to
  paths they were given; the CLI loads a recovery record and passes
  `start_epoch`),
- no tracker lifecycle management.

The shipped loop is the *classification* specialization: it expects logits
`(N, C)` and reports loss, accuracy, per-class precision/recall/F1,
unweighted macro means, and `train/lr`. This is deliberate — per the
batch-contract section, a loop is wedded to its task, so the template ships
one concrete reference loop instead of a false abstraction. Regression or
ranking tasks rewrite `trainer.py` (and probably `Batch`) for their project.

## Configuration as composition root

One YAML file describes the whole experiment: data sources and column
roles, raw and tensor processing, model, optimizer, scheduler, loss, training
loop, inference output, and the analysis report. The pattern (normative
detail and extension recipes: [config-pattern.md](config-pattern.md)):

- Each schema lives **with the class it builds** — in the class's module
  (`features/raw.py`, `features/tensor.py`) or the layer's `config.py`
  (`template.models.config`, `template.persistence.config`, ...); library
  classes stay constructor-driven and config-free. The
  composition root's own schema (`cli/config.py`) adds `ExperimentConfig`
  (assembly + cross-object invariants) and the run-level schemas no layer
  owns: `DataConfig` (which raw sources feed the run, column roles),
  `TrainingConfig` / `InferenceConfig`, and the optimizer/scheduler/loss
  schemas over plain `torch` objects.
- Every schema has a `build()` method returning plain library objects.
  Column names are config-level declarations; raw steps (`features.raw`)
  resolve them against the raw frame, and tensor steps (`features.tensor`)
  to feature-tensor blocks when the pipeline is built, so the steps stay
  tensor-only.
- Polymorphic slots (sources, raw steps, tensor steps, models,
  optimizers, schedulers, losses, trackers, checkpoints) use a `kind`
  discriminator:
  adding an implementation = new class in its layer + new `*Config` schema
  added to the union. The slot's union is its type from day one, even with
  a single member.
- Cross-field invariants (e.g. the target mapping covers exactly the
  classifier's `0..n_classes-1`) are model validators on `ExperimentConfig`.
- `load_config` is the only place the file is read, and all three
  entrypoints validate the *whole* file, so a typo in the report section
  fails training too — one file is the single source of truth for "what
  experiment is this".

This is where "configurable setups" fit architecturally: configuration is
not a layer of its own, it is the composition root that replaces a
hand-written `main()` wiring function with a declarative, validated file.

## Entrypoints

`template-train`, `template-infer`, and `template-report` merge what DDD
would split into presentation and application layers — for a training project
that split is ceremony, so the CLI both parses args/config and orchestrates
the use case:

`template-train`: `config.data.train.build()` →
`config.raw.build(raw.columns)` + `ProcessedSource` →
`TableSchema.from_columns` over the pipeline's declared output columns (roles
resolved; the resolved feature set and width are logged on purpose — a stray
or leaky column surfaces in the logs and tracked params instead of inside the
model) → `ValidatedSource` → `partition_indices` → `TableDataset` + `Subset`
loaders → `config.model.build(schema.feature_width)` on device → optimizer,
then the scheduler around that optimizer when configured → open trackers, log
flattened config as params → optionally `restore_training_state` from the
recovery checkpoint when `--resume` is set → `Trainer.train(tensor_pipeline=…,
seed=…, start_epoch=…, checkpoints=…)`.

`template-infer` and `template-report` share `cli/predict.py` for one pass
over `data.test`: `config.data.test.build()` →
`config.raw.build(raw.columns)` + `ProcessedSource` → schema built with
`include_target=False` (inference) or `True` (report; the target is then
required) → ids read through the validated source (metadata rides through
untouched, row order preserved) → model rebuilt from the same config, weights
loaded from the best checkpoint when that file exists and otherwise from the
recovery checkpoint (`--checkpoint` overrides it for the report), `eval()` →
per batch: the tensor pipeline at stage `predict`, then
`softmax(logits, dim=1)[:, 1]`.

`template-infer` then reports/saves `id,<target>`; `template-report` passes
the probabilities and the retained targets to
`evaluation.binary_metrics(..., threshold=report.threshold)` and writes the
markdown analysis. One shared pass is what makes the report describe the same
rows as inference.

A backend API entrypoint would do the same wiring inside request handlers
instead of `argparse` mains; nothing in the library changes.

## Instantiating the template

1. Rename the package: `src/template` → `src/<project>` (and `name`,
   `[project.scripts]`, logger name `"template"` in `cli/runtime.py`, and
   the imports). Every reference is greppable as `template`.
2. Adapt `data` to your samples: redefine `Batch`, the `TableDataset`
   adapter, and — if your raw form is not a polars frame — the source ports
   and the validated table they return. Extend `TableSchema` if your columns
   need roles beyond the shipped ones, and add `RawStep`s in
   `features/raw.py` for new raw processing.
3. Add the tensor steps your domain needs to `features/tensor.py` (+ their
   config kinds); extend `TensorStep` invariants if your steps do something
   the current contract forbids.
4. Add your architecture(s) as `kind`s in `models/config.py` (or replace
   `MLPClassifier` outright).
5. If the task isn't binary classification, rewrite `training/trainer.py`
   for your metrics and the model/loss schemas to match, and `evaluation.py`
   (with its rendering in `cli/report.py`) for the report.
6. Give every new component an `XConfig` in its layer — its `config.py`, or
   alongside the class in `features/raw.py` / `features/tensor.py` — and
   register its `kind` in the slot's union
   ([config-pattern.md](config-pattern.md) has recipes); update
   `configs/example.yaml.example` and `docs/cli.md`.
7. Add trackers as needed (W&B etc.) as new `kind`s.

## Development

```bash
uv sync
uv run ty check
uv run ruff check src tests
uv run pytest
```
