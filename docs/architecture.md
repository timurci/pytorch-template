# Architecture

A layered template for PyTorch training projects. The goal is not a framework
but a *shape*: independent library layers that interact through
protocols/abstract classes at most, wired together at the entrypoints from a
single declarative config.

Two principles do most of the work:

1. **Layers depend on contracts, not on each other's implementations.**
   Cross-layer references are `typing.Protocol`s (`Transform`, `Augmenter`,
   `ExperimentTracker`), `torch` abstractions (`nn.Module`, `Optimizer`,
   `DataLoader`), or one shared data contract (`Batch`). Swapping an
   implementation — a tracker, an augmenter, a model — touches no other
   layer.
2. **Config is the composition root.** One YAML file describes the whole
   experiment; Pydantic schemas in the CLI layer validate it and `build()`
   plain library objects. Library layers never see config objects — they take
   plain constructor arguments, which keeps them testable and reusable from
   notebooks, scripts, or services without the CLI.

The structure resembles an FTI (feature / training / inference) pipeline:
the feature layer and preprocessing config are the *feature pipeline*,
training is the *training pipeline*, and the inference entrypoint is the
*inference pipeline* — all three sharing one config so what trains is what
serves.

## Layers

| Layer | Role | Owns | Must not |
| --- | --- | --- | --- |
| `persistence` | Persistent I/O | `load_table` / `save_table`, `save_checkpoint` / `load_checkpoint` | Know about datasets, models, or training; pick its own paths (callers supply them) |
| `data` | The batch contract + in-memory datasets | `Batch` TypedDict; `TableDataset` (`from_frame` / `split` / `class_counts`) | File I/O; preprocessing (frames arrive preprocessed) |
| `features` | Data transformations | `Transform` protocol + `Pipeline` (offline preprocessing); `Augmenter` protocol (train-time augmentation) | File I/O; task-specific column names; fitting state from data |
| `models` | Network definitions | `nn.Module` subclasses, e.g. `MLPClassifier` | Anything but `torch` |
| `tracking` | Observability port | `ExperimentTracker` protocol; `Null` / `Stdout` / `MLflow` adapters | Owning external lifecycles (the CLI opens/closes the MLflow run) |
| `training` | The training loop | `Trainer` | Splitting data, building models/loaders, checkpointing, opening tracker runs |
| `cli` | Entrypoints / composition root | Config schemas, `template-train`, `template-infer` | Contain business logic — it only wires layers |

## Dependency rules

Library layers import nothing from sibling layers except the explicitly
allowed contract imports. The CLI imports everything.

```
                    ┌─────────────┐
                    │     cli     │  (imports all layers)
                    └──────┬──────┘
                           │
   ┌───────────┬───────────┼───────────┬──────────────┐
   │           │           │           │              │
persistence  data      features     models        tracking
   │           │           │           │              │
   └───────────┴─────┬─────┴───────────┴──────────────┘
                     │ (contracts only)
                  training
```

`training` is allowed to import exactly three things from the library:

- `data.Batch` — the data contract (see below),
- `tracking.ExperimentTracker` — the observability port,
- `features.Augmenter` — the augmentation port,

plus `torch`'s own `nn.Module`, `Optimizer`, and `DataLoader`. Everything
else (constructing the model, the optimizer, the loaders, the trackers,
saving checkpoints) happens in the entrypoint and is passed in.

## The batch contract

`Batch` is the one concrete, non-protocol type that crosses layer
boundaries:

```python
class Batch(TypedDict):
    features: Tensor            # (N, F) float
    targets: NotRequired[Tensor]  # (N,) long class indices
```

The training loop is annotated `DataLoader[Batch]`. It is owned by the
**`data` layer**, not by training: datasets produce it, the trainer consumes
it, and neither persistence nor features needs to know it exists.

Why not a training-owned port? The purer dependency-inversion alternative is
to let `training` define the batch contract and have `data` adapt to it. We
reject it here on purpose: a training loop's semantics are inseparable from
the shape of its data and task anyway (this trainer computes classification
metrics; a different task rewrites the loop), so abstracting the batch buys
indirection without decoupling. If you ever need two incompatible batch
shapes in one project, that is the signal to revisit this decision — define a
`Batch` protocol in `training` and let the datasets conform.

Tasks whose samples are not `(features, targets)` pairs redefine `Batch`
here in the data layer; the trainer and entrypoints follow.

## Preprocessing vs. augmentation

Both live in `features`, but they are different things:

| | `Transform` (preprocessing) | `Augmenter` (augmentation) |
| --- | --- | --- |
| When | Offline, once, before the dataset is built | Train-time, every training batch |
| Input | Whole Polars frame | One batch of tensors, on the training device |
| Determinism | Deterministic, declared state, never fit from data | May be stochastic |
| Row count | Preserved | May change (oversampling etc.) |
| Output | Transformed frame (reversible via `revert`) | `AugmentedBatch` with provenance |
| Applied to validation/inference | Yes | **No** |

### `Transform`

`Transform` is `apply` / `revert`; `Pipeline(steps)` applies forward and
reverts in reverse order. Transforms hold declared state (column names, caps,
levels, mappings) from the constructor and never fit from the frame. Missing
columns raise `KeyError`; uncovered values raise `ValueError`. Non-invertible
steps (`DropColumns`) raise `NotImplementedError` from `revert`. Because
steps are reversible, the same config-driven pipeline serves training and
inference and can map predictions back to the original space.

### `Augmenter`

```python
class AugmentedBatch(NamedTuple):
    features: Tensor        # (M, F) augmented rows; M may differ from N
    targets: Tensor         # (M,) labels aligned with the augmented rows
    source_indices: Tensor  # (M,) long: original-batch row that produced row j

class Augmenter(Protocol):
    def augment(self, features: Tensor, targets: Tensor) -> AugmentedBatch: ...
```

The defining trait — what makes this *augmentation* rather than more
preprocessing — is the **provenance mapping**: `source_indices[j]` names the
row of the original batch that produced augmented row `j`. Contract:

- **Label-preserving augmenters** (the default case) MUST satisfy
  `targets == input_targets[source_indices]`. They may reorder, duplicate,
  or drop rows, but never change a row's label. Both reference
  implementations (`IdentityAugmenter`, `GaussianNoiseAugmenter`) are
  label-preserving with `arange` provenance.
- The trainer applies the augmenter after moving the batch to the device,
  before the forward pass, and feeds `augmented.features` / `augmented.targets`
  to the model. Train metrics therefore describe the augmented data.
- Validation is never augmented.
- `source_indices` stays available for provenance-aware logic: e.g.
  importance-weighting oversampled duplicates, or per-original-sample
  statistics in a custom loop.
- *Escape hatch:* a label-**transforming** augmenter (e.g. mixup soft labels)
  breaks the invariant on purpose. That is legal, but the paired loss must
  accept the new target format — task-specific wiring, outside this contract.

Injection mirrors trackers: `Trainer.train(..., augmenter=augmenter)`, built
from `training.augmenter` in the config (`null` disables). Compose your own
`CompositeAugmenter` when you need chaining; the shipped kinds are
deliberately minimal.

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
registering a new `kind` in the config union.

## Training

`Trainer(model, optimizer, device=...)` owns exactly one thing: the
train/validation epoch loop — moving batches to the device, applying the
augmenter, forward/backward/step, computing metrics, fanning out to
trackers. Explicit non-responsibilities, all handled by the caller:

- no train/val splitting (`TableDataset.split` does it),
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

One YAML file describes the whole experiment: data paths, pipeline, model,
optimizer, loss, training loop, inference output. The pattern:

- Pydantic schemas live **only** in `cli/config.py`; library layers never
  import them.
- Every schema has a `build()` method returning plain library objects, so
  the library stays config-free and constructor-driven.
- Polymorphic slots (transforms, augmenters, trackers, optimizers) use a
  `kind` discriminator: adding an implementation = new class in its layer +
  new `*Config` schema added to the union.
- `extra="forbid"` everywhere: misspelled keys fail fast at load time.
- Cross-field invariants (e.g. target mapping covers exactly
  `0..n_classes-1`) are model validators on the schema.
- Both entrypoints validate the *whole* file, so a typo in the inference
  section fails training too — one file is the single source of truth for
  "what experiment is this".

This is where "configurable setups" fit architecturally: configuration is
not a layer of its own, it is the composition root that replaces a
hand-written `main()` wiring function with a declarative, validated file.

## Entrypoints

`template-train` and `template-infer` merge what DDD would split into
presentation and application layers — for a training project that split is
ceremony, so the CLI both parses args/config and orchestrates the use case:

`template-train`: load table → apply pipeline → map target → `TableDataset`
→ split → loaders → model/optimizer/loss on device → open trackers, log
flattened config as params → `Trainer.train` → save checkpoint.

`template-infer`: load table (keep id column) → apply the same pipeline →
rebuild model from config, load `state_dict`, `eval()` → softmax over loader
→ report/save predictions.

A backend API entrypoint would do the same wiring inside request handlers
instead of `argparse` mains; nothing in the library changes.

## Instantiating the template

1. Rename the package: `src/template` → `src/<project>` (and `name`,
   `[project.scripts]`, logger name `"template"` in `cli/runtime.py`, and
   the imports). Every reference is greppable as `template`.
2. Adapt `data` to your samples: redefine `Batch` and the dataset class.
3. Add the transforms your domain needs to `features` (+ their config kinds).
4. Add augmenters if needed (+ config kinds).
5. Replace `MLPClassifier` with your architecture(s).
6. If the task isn't classification, rewrite `training/trainer.py` for your
   metrics and `cli` model/loss schemas to match.
7. Wire the config schema in `cli/config.py` to the new pieces; update
   `configs/example.yaml.example` and `docs/cli.md`.
8. Add trackers as needed (W&B etc.) as new `kind`s.

## Development

```bash
uv sync
uv run ty check
uv run ruff check src tests
uv run pytest
```
