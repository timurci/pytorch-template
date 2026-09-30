# Architecture

A layered template for PyTorch training projects. The goal is not a framework
but a *shape*: independent library layers that interact through
protocols/abstract classes at most, wired together at the entrypoints from a
single declarative config.

Two principles do most of the work:

1. **Layers depend on contracts, not on each other's implementations.**
   Cross-layer references are `typing.Protocol`s (`RawTableSource`,
   `DataSource`, `ProcessingStep`, `ExperimentTracker`), `torch` abstractions
   (`nn.Module`, `Optimizer`, `DataLoader`), or one shared data contract
   (`Batch`). Swapping an implementation — a tracker, a processing step, a
   model — touches no other layer.
2. **Config is the composition root.** One YAML file describes the whole
   experiment; Pydantic schemas validate it and `build()` plain library
   objects. Each schema lives in the layer of the class it builds; the CLI
   assembles them into `ExperimentConfig` and does the wiring. Library
   objects never see config objects — they take plain constructor
   arguments, which keeps them testable and reusable from notebooks,
   scripts, or services without the CLI.

The structure resembles an FTI (feature / training / inference) pipeline:
ingestion plus feature processing and their config are the *feature pipeline*,
training is the *training pipeline*, and the inference entrypoint is the
*inference pipeline* — all three sharing one config so what trains is what
serves.

## Layers

| Layer | Role | Owns | Must not |
| --- | --- | --- | --- |
| `persistence` | Raw table readers + persistent I/O | `CsvSource` / `ParquetSource` (raw, unvalidated rows by index), `save_table`, `save_checkpoint` / `load_checkpoint`, and their `config.py` schemas | Know about validated forms, datasets, models, or training; validate structure; pick its own paths (callers supply them) |
| `data` | Ingestion contracts + the torch-side dataset adapter | `RawTableSource` / `DataSource` ports; `TableSchema` (roles + encodings) and the encoders (`OneHot`, `MapValues`) with their `config.py` schemas; `ValidatedTable` / `ValidatedSource`; `Batch` TypedDict; `TableDataset`; `partition_indices` / `class_counts` | File I/O (rows arrive through the ports); import anything from sibling layers |
| `features` | Feature processing | `ProcessingStep` protocol + `ProcessingPipeline` (ordered, stage-tagged steps over batches); `ScaleByCap`, `LogScaleByCap`, `GaussianNoise`, and their `config.py` schemas | File I/O; raw forms or column names in steps (config declares names, resolved at build); fitting state from data |
| `models` | Network definitions | `nn.Module` subclasses, e.g. `MLPClassifier`; `models/config.py` schemas | Anything but `torch` (its `config.py` may add `pydantic`) |
| `tracking` | Observability port | `ExperimentTracker` protocol; `Null` / `Stdout` / `MLflow` adapters, and their `config.py` schemas | Owning external lifecycles (the CLI opens/closes the MLflow run) |
| `training` | The training loop | `Trainer` | Touching sources or splitting data, building models/loaders, checkpointing, opening tracker runs |
| `cli` | Entrypoints / composition root | The composition-root schema (`cli/config.py`: `ExperimentConfig`, `DataConfig`, `TrainingConfig` / `InferenceConfig`, optimizer/loss schemas) and the `build()` wiring, `template-train`, `template-infer` | Contain business logic — it only wires layers |

## Dependency rules

Library layers import nothing from sibling layers except the explicitly
allowed contract imports. The CLI imports everything and composes.

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

- `features` → `data`: **`Batch` only** — steps consume and return the batch
  contract; that single edge is why the batch is owned by `data` (below).
  Step schemas resolve column names through `BlockResolver`
  (`features/protocol.py`), structurally satisfied by `data.TableSchema`,
  so config adds no import edge.
- `training` → `data.Batch`, `features.ProcessingPipeline` /
  `features.Stage`, `tracking.ExperimentTracker`, plus `torch`'s own
  `nn.Module`, `Optimizer`, and `DataLoader`.
- `persistence` and `data` import nothing from sibling layers at all: raw
  readers are plain classes whose methods (`columns`, `count`,
  `read(indices) -> pl.DataFrame`) satisfy `data.RawTableSource`
  *structurally*, so no import edge exists. `data`'s modules only import
  each other (and `polars` / `torch`).
- `models` and `tracking` import nothing from sibling layers.

Everything else (constructing the model, the optimizer, the loaders, the
trackers, the processing pipeline, saving checkpoints) happens in the
entrypoint and is passed in.

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
**`data` layer**, not by training: datasets produce it, feature-processing
steps and the trainer consume it. `targets` is absent for unlabeled
(inference) data. `source_indices` appears after feature processing: the
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

## Ingestion: raw sources, column roles, encodings

Ingestion (raw rows → encoded tensors) lives in `data` + `persistence`. It is
two sides of one seam: the raw side serves unvalidated frames, the validated
side serves rows checked against a declared schema.

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
  index. They are deliberately unvalidated — roles and encoding would couple
  them to schema declarations. Lazy readers (e.g. `pl.scan_csv`) are a
  documented pattern, not shipped: a polars lazy scan re-executes its query
  on every `read`, so per-batch reads only pay off on indexed sources such as
  a database.
- **`TableSchema`** declares what each raw column *is* (a role) and how it
  *becomes a tensor* (an encoding):
  - **Roles are projections, not partitions.** `metadata_columns` identify
    samples and are kept untouched; `target_column` is the supervised target
    and is never a feature (that would be leakage — the schema rejects it);
    `TableSchema.from_columns` takes `exclude` columns and derives the
    features as whatever remains, keeping raw order — declaring exclusions
    is enough, the feature set needs no list of its own. A column MAY be
    both feature and metadata (e.g. an id that is also a model input) — it
    then appears in the feature block and in the metadata side; that duality
    needs direct construction, since `from_columns` derives disjoint roles.
  - **Encodings** map only the columns that need them: `OneHot` for
    categoricals (levels declared inline or one per line in a `levels_path`
    vocab file; width = number of levels), `MapValues` for the target
    (declared class indices), numeric passthrough otherwise (width 1).
    Encoders are declared state — never fit from the data they encode — so
    train and inference encode identically. Uncovered values raise.
  - **`feature_width` / `feature_slice(column)`** resolve names to
    feature-tensor blocks in `feature_columns` order. Because a block's width
    is its encoder's width, the feature count is known before any row is
    read.
- **`ValidatedSource`** wraps a raw source with a schema and is the
  `DataSource[ValidatedTable]` adapter: every read is checked at the
  boundary, exactly once, so raw sources stay schema-agnostic.
  `ValidatedTable` enforces the schema's roles — required columns present,
  and encoderless feature/target columns must be numeric (encoded columns
  are validated by their encoder at encode time). Extra columns are ignored,
  so auxiliary columns ride through untouched.
- **`TableDataset`** is the only torch-aware piece: the adapter from
  `DataSource[ValidatedTable]` to `Batch` samples. It encodes each column
  through the schema's encoders and concatenates the blocks in
  `feature_columns` order — the layout `feature_slice` documents. It
  implements `__getitem__` (per-item fallback) and `__getitems__` (fetches a
  whole batch's rows with a single `read`, which the `DataLoader` uses when
  present), and nothing else.
- **Random access is mandatory.** Map-style datasets need index-addressable
  rows; pure streams must adapt (materialize a buffer) at the boundary or
  are out of scope for this port.

Splits and target statistics are index-based, so they work identically for
any source and neither the dataset nor the sources know a split exists:
`partition_indices(count, val_fraction, seed=...)` returns disjoint index
sets (seed-random, covering all rows), stock `torch.utils.data.Subset` wraps
the dataset (it forwards `__getitems__`, keeping batch fetch under splits),
and `class_counts(source, indices)` reads the target through the port.

## Feature processing

One layer, one contract: `features` handles both preprocessing and
augmentation, and it is **tensor-only** — a step receives a batch of tensors
and returns a batch of tensors. Raw forms, column names, and encodings never
appear here; names resolve to feature-tensor blocks (`feature_slice`) at
composition time, so one step works regardless of how the data was ingested.

### The step contract

```python
class ProcessingStep(Protocol):
    def process(self, batch: Batch, rng: Generator) -> Batch: ...
```

Invariants (normative, see `features/protocol.py`):

- Steps never mutate their input; they return a `Batch` (cloning on write).
  Fields a step does not touch pass through unchanged.
- Deterministic steps ignore `rng`; stochastic steps MUST draw from `rng`
  (never the global RNG) so a seeded run reproduces exactly.
- Steps that change the row count keep `source_indices` and `targets`
  aligned with `features` (label-preserving steps satisfy
  `targets == input_targets[source_indices]`).

### Composition and stages

`ProcessingPipeline` applies an ordered list of `(step, stages)` pairs to one
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

`Trainer(model, optimizer, device=...)` owns exactly one thing: the
train/validation epoch loop — moving batches to the device, applying the
processing pipeline, forward/backward/step, computing metrics, fanning out
to trackers. The pipeline runs on each batch *after* the move to `device`:
training batches at stage `train`, validation batches at stage `eval` (so
seeded validation-time augmentation is a tagging choice in the pipeline, not
trainer code). Processed features and labels replace the batch — metrics
therefore describe the processed data.

Explicit non-responsibilities, all handled by the caller:

- no train/val splitting (`partition_indices` + `Subset` do it),
- no data access (it sees `DataLoader[Batch]`, never a source),
- no model/optimizer/loss/loader construction,
- no checkpointing (persistence functions, called by the CLI),
- no tracker lifecycle management.

The shipped loop is the *classification* specialization: it expects logits
`(N, C)` and reports loss, accuracy, per-class precision/recall/F1, and
unweighted macro means. This is deliberate — per the batch-contract section,
a loop is wedded to its task, so the template ships one concrete reference
loop instead of a false abstraction. Regression or ranking tasks rewrite
`trainer.py` (and probably `Batch`) for their project.

## Configuration as composition root

One YAML file describes the whole experiment: data sources and column
roles, feature processing, model, optimizer, loss, training loop,
inference output. The pattern (normative detail and extension recipes:
[config-pattern.md](config-pattern.md)):

- Each schema lives **with the class it builds**, in its layer's
  `config.py` (`template.models.config`, `template.persistence.config`,
  ...); library classes stay constructor-driven and config-free. The
  composition root's own schema (`cli/config.py`) adds `ExperimentConfig`
  (assembly + cross-object invariants) and the run-level schemas no layer
  owns: `DataConfig` (which raw sources feed the run, column roles),
  `TrainingConfig` / `InferenceConfig`, and the optimizer/loss schemas
  over plain `torch` objects.
- Every schema has a `build()` method returning plain library objects.
  Column names are config-level declarations; they resolve to
  feature-tensor blocks when the processing pipeline is built, so
  processing steps stay tensor-only.
- Polymorphic slots (sources, encodings, processing steps, models,
  optimizers, losses, trackers) use a `kind` discriminator: adding an
  implementation = new class in its layer + new `*Config` schema added to
  the union. The slot's union is its type from day one, even with a single
  member.
- Cross-field invariants (e.g. the target mapping covers exactly the
  classifier's `0..n_classes-1`) are model validators on `ExperimentConfig`.
- `load_config` is the only place the file is read, and both entrypoints
  validate the *whole* file, so a typo in the inference section fails
  training too — one file is the single source of truth for "what
  experiment is this".

This is where "configurable setups" fit architecturally: configuration is
not a layer of its own, it is the composition root that replaces a
hand-written `main()` wiring function with a declarative, validated file.

## Entrypoints

`template-train` and `template-infer` merge what DDD would split into
presentation and application layers — for a training project that split is
ceremony, so the CLI both parses args/config and orchestrates the use case:

`template-train`: `config.data.train.build()` → `TableSchema.from_columns`
(roles + encodings resolved from the raw columns; the resolved feature set
and width are logged on purpose — a stray or leaky column surfaces in the
logs and tracked params instead of inside the model) → `ValidatedSource` →
`partition_indices` → `TableDataset` + `Subset` loaders →
`config.model.build(schema.feature_width)` on device → open trackers, log
flattened config as params → `Trainer.train(processing=…, seed=…)` →
`save_checkpoint`.

`template-infer`: `config.data.test.build()` → schema built with
`include_target=False` (target dropped when present, absent otherwise) →
ids read through the validated source (metadata rides through untouched,
row order preserved) → model rebuilt from the same config, `state_dict`
loaded, `eval()` → per batch: processing at stage `predict`, then
`softmax(logits, dim=1)[:, 1]` → report/save `id,<target>`.

A backend API entrypoint would do the same wiring inside request handlers
instead of `argparse` mains; nothing in the library changes.

## Instantiating the template

1. Rename the package: `src/template` → `src/<project>` (and `name`,
   `[project.scripts]`, logger name `"template"` in `cli/runtime.py`, and
   the imports). Every reference is greppable as `template`.
2. Adapt `data` to your samples: redefine `Batch`, the `TableDataset`
   adapter, and — if your raw form is not a polars frame — the source ports
   and the validated table they return. Extend `TableSchema` if your columns
   need roles or encodings beyond the shipped ones.
3. Add the processing steps your domain needs to `features` (+ their config
   kinds); extend `ProcessingStep` invariants if your steps do something the
   current contract forbids.
4. Add your architecture(s) as `kind`s in `models/config.py` (or replace
   `MLPClassifier` outright).
5. If the task isn't classification, rewrite `training/trainer.py` for your
   metrics and the model/loss schemas to match.
6. Give every new component an `XConfig` in its layer's `config.py` and
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
