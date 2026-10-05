# The config pattern

How this template turns one YAML file into typed, buildable, extensible
configuration — and how to make a new component (a model, a data source, a
processing step, a tracker) configurable without touching the wiring. The
field reference lives in [cli.md](cli.md); the architectural placement in
[architecture.md](architecture.md#configuration-as-composition-root).

## The shape of a schema

Every configurable component `X` has an `XConfig`: a Pydantic model
(subclass of `ConfigModel`, `template/config_pattern.py`) that lives in
`X`'s layer, declares what YAML may say about `X`, and `build()`s the
plain object. `ConfigModel` sets `extra="forbid"`, so an unknown key is a
load error everywhere, not a silent typo. A real example — the CSV
reader:

```python
class CsvSourceConfig(ConfigModel):
    kind: Literal["csv"] = "csv"
    path: Path

    def build(self) -> CsvSource:
        return CsvSource(self.path)
```

The rules, normatively:

- **`kind` is the type tag.** A schema for a polymorphic slot carries
  `kind: Literal["..."]` with its own tag as the default, and YAML picks
  the implementation. A slot may name a default kind — `model` (mlp),
  `optimizer` (adamw), `loss` (cross_entropy), each via `default_kind(...)`
  after the discriminator in the slot's `Annotated` alias — and then also
  accepts a tag-less mapping; without one the tag is required. (The field
  is called `kind` throughout this repo; the name is arbitrary — `type`
  works the same. Pick one per repo and never mix.)
- **Validation lives in the schema.** Constraints (`Field(gt=0)`, ranges,
  formats) and single-object invariants (`model_validator`) sit next to the
  fields they guard, so every entrypoint validates identically.
- **`build()` returns plain objects.** Its inputs are plain values — an
  input width, class counts, a resolved block — never other config
  objects. Library classes stay constructor-driven and config-free:
  testable and reusable from notebooks or services without the CLI.
- **A config is usable state, not just a DTO.** Beyond `build()`, a schema
  MAY carry domain methods over its fields
  (`CrossEntropyLossConfig.resolve_weights`) when the logic is about the
  declaration itself.

## Where schemas live

A schema lives **with the class it builds**: in the class's own module when
the layer groups classes by concept (`features/raw.py`, `features/tensor.py`)
or in the layer's `config.py` (`template.models.config`,
`template.persistence.config`, ...) otherwise. Shared pieces — the strict
`ConfigModel` base and `default_kind` — live in
`template/config_pattern.py`. Two things stay in `cli/config.py`, the
composition root's own schema:

- `ExperimentConfig` — the assembly of every layer's schemas into one
  file, plus cross-object invariants (below);
- run-level schemas no library layer owns: `DataConfig` (which raw sources
  feed the run, plus column roles), `TrainingConfig` / `InferenceConfig` /
  `ReportConfig` (the run and its outputs), and `OptimizerConfig` /
  `SchedulerConfig` / `LossConfig` (plain `torch` objects — no layer owns
  them).

The *wiring* — who calls `build()` with what and passes the result where —
belongs to the entrypoints (`cli/train.py`, `cli/predict.py`, `cli/infer.py`,
`cli/report.py`). Nothing in the library layers knows about YAML.

| YAML slot | Union (`kind`-tagged) | Schema module | Builds |
| --- | --- | --- | --- |
| `data.train` / `data.test` | `RawSourceConfig` | `persistence/config.py` | `CsvSource` / `ParquetSource` |
| `data.target` | — (single shape) | `features/raw.py` | `MapValues` |
| `raw.steps` | `RawStepConfig` | `features/raw.py` | `RawStep` |
| `tensor.steps` | `TensorStepConfig` | `features/tensor.py` | `(TensorStep, stages)` |
| `model` | `ModelConfig` | `models/config.py` | `nn.Module` |
| `optimizer` | `OptimizerConfig` | `cli/config.py` | `torch.optim.Optimizer` |
| `scheduler` | `SchedulerConfig` | `cli/config.py` | `torch.optim.lr_scheduler.LRScheduler` |
| `loss` | `LossConfig` | `cli/config.py` | `nn.CrossEntropyLoss` |
| `training.trackers` | `TrackerConfig` | `tracking/config.py` | `ExperimentTracker` |
| `training.checkpoints` | `CheckpointConfig` | `training/config.py` | `RecoveryCheckpoint` / `BestCheckpoint` |

## Discriminated unions: types YAML can tell apart

A polymorphic slot is a *tagged union*, not a bag of optional fields:

```python
RawSourceConfig = Annotated[
    CsvSourceConfig | ParquetSourceConfig,
    Field(discriminator="kind"),
]
```

- The `kind` discriminator makes parsing unambiguous and error messages
  exact: an unknown `kind` fails at load time, before
  any data is touched.
- A union with one member is the normal state of a young slot
  (`ModelConfig`): the slot is a union from day
  one, so adding a variant is one line and no call site changes.
- Declaring a field default alone does not make the tag optional: tag
  dispatch needs the tag before it can pick the member. A slot that wants
  omission says so once, with `default_kind("<kind>")` after the
  discriminator (`model` → `mlp`, `optimizer` → `adamw`, `loss` →
  `cross_entropy`); every other slot (`data.train` / `test`,
  `raw.steps[]`, `tensor.steps[]`, `scheduler`,
  `training.trackers[]`, `training.checkpoints[]`) requires the tag in YAML.
  `scheduler` may also be
  omitted entirely, which builds no scheduler.
- Variant fields are typed per variant — `SgdConfig.momentum` does not
  exist for `adam`. "Anything else goes here" means *fields of that
  variant*, each with its own constraints.

## Loading

`load_config(path)` (in `cli/config.py`) is the only place the file is
read: `yaml.safe_load` → `ExperimentConfig.model_validate`. All three
entrypoints call it with `-c/--config` and all validate the *whole* file,
so an invalid value in the report section fails training too — one
file is the single source of truth for "what experiment is this".
Unrecognized keys are errors (`ConfigModel` sets `extra="forbid"`), so a
misspelled field fails at load along with wrong `kind`s, wrong types, and
constraint violations. Paths are relative to the working directory.
Downstream code sees typed objects only; there is no dict plumbing past
this point.

## Cross-object invariants

Invariants that span sections live on `ExperimentConfig` as
`model_validator`s — e.g. the target mapping must cover exactly the class
indices the classifier's head emits:

```python
@model_validator(mode="after")
def _validate_target_classes(self) -> Self:
    if isinstance(self.model, MLPModelConfig):
        values = set(self.data.target.mapping.values())
        expected = set(range(self.model.n_classes))
        if values != expected:
            raise ValueError(...)
    return self
```

Concrete on purpose: the check names the kinds it applies to. A new
classifier kind opts in by being added here; a non-classifier has no
class indices to match and stays out.

## Recipe: a configurable model (worked example: autoencoder)

Say the project needs `model: {kind: autoencoder, latent_dim: 128}` in
YAML. With the pattern established, extension is four steps and
no wiring changes:

1. **The class**, in its layer: `models/autoencoder.py`, plain
   `nn.Module`, plain constructor arguments — unchanged by any of this.
2. **The schema**, in `models/config.py`:

   ```python
   class AutoencoderConfig(ConfigModel):
       kind: Literal["autoencoder"] = "autoencoder"
       latent_dim: int = Field(gt=0)
       # anything else goes here

       def build(self, input_size: int) -> AutoencoderModel:
           return AutoencoderModel(input_size, self.latent_dim)
   ```

3. **Register the kind** in the slot's union:

   ```python
   ModelConfig = Annotated[
       MLPModelConfig | AutoencoderConfig,
       Field(discriminator="kind"),
       default_kind("mlp"),
   ]
   ```

   `default_kind("mlp")` keeps the slot's default: `model: {}` still
   parses as `mlp`.

4. **Use it in YAML**:

   ```yaml
   model:
     kind: autoencoder
     latent_dim: 128
   ```

The entrypoints already do `config.model.build(schema.feature_width)` —
they do not change. Notes that come with the territory:

- A non-classifier simply isn't part of the target-mapping check — and it
  needs a loss it can pair with. Adding a loss kind is the same recipe
  (`MSELossConfig` in `cli/config.py`, one line in the `LossConfig`
  union), plus the slot's member contract: `cli/train.py` calls
  `loss.build(class_counts)` and `loss.resolve_weights(class_counts)`
  unconditionally, so a new kind implements both (`resolve_weights`
  returns `None` when the loss has no class weights) — or the loss wiring
  changes with it. The *training loop* is task-specialized; a
  non-classification task also rewrites `training/trainer.py` (see
  [architecture.md](architecture.md#training)).
- Checkpoints store weights, not the architecture, so inference loads
  them with the model built from the same config. One shared file
  guarantees that by construction.

## Recipe: a new data source kind

Same recipe, one layer down. Say ingestion should also accept JSON files:

1. `persistence/sources.py`: `JsonSource` — loads the table at
   construction, serves rows by index (random access is mandatory for the
   map-style dataset port; see
   [architecture.md](architecture.md#ingestion-raw-processing-column-roles-tensorization)).
2. `persistence/config.py`:

   ```python
   class JsonSourceConfig(ConfigModel):
       kind: Literal["json"] = "json"
       path: Path

       def build(self) -> JsonSource:
           return JsonSource(self.path)
   ```

3. Register: `RawSourceConfig = Annotated[
   CsvSourceConfig | ParquetSourceConfig | JsonSourceConfig,
   Field(discriminator="kind")]`.
4. YAML: `data: {train: {kind: json, path: data/raw/train.json}, ...}`.

Again the CLI is untouched: it already calls `config.data.train.build()`.
Remember that readers stay **unvalidated** — roles and raw steps are
declarations applied at the composition seam, so a new reader needs to
know nothing about schemas.

## Recipe: a raw step, a tensor step, or a tracker

- **Raw step**: a class in `features/raw.py` with
  `columns(inputs) -> tuple[str, ...]` (its static output layout) and
  `process(frame) -> frame` (rows preserved), plus an `XConfig` registered
  in `RawStepConfig`. Column names are visible here; selection, renaming,
  deriving, and encoding all fit. Declare the output columns honestly — the
  schema's roles and the model's input width are resolved from them.
- **Tensor step**: a class in `features/tensor.py` with
  `process(batch, rng) -> batch`, plus an `XConfig` with
  `build(resolve) -> tuple[TensorStep, frozenset[Stage]]` registered in
  `TensorStepConfig` (the tag is required — steps declare no default kind).
  Column names are declarations resolved at build through
  `resolve.feature_slice(column)` (`BlockResolver`, structurally satisfied
  by `TableSchema`), so steps stay tensor-only. Tag `stages` defaults
  honestly: deterministic steps run at every stage, stochastic ones at
  `train` only.
- **Tracker**: `XConfig` in `tracking/config.py`, registered in
  `TrackerConfig` (the tag is required). Trackers stay passive; any
  external lifecycle (an MLflow run, a W&B session) is opened by the
  entrypoint — see `cli/runtime.py`.

## Adding a kind, in checklist form

1. The plain class in its layer (constructor-driven, no config types).
2. `XConfig` alongside the class (its module or the layer's `config.py`): a
   `ConfigModel` subclass, `kind` `Literal`, field constraints, `build()`.
3. One line in the slot's union; add `default_kind(...)` only if the slot
   should also accept a tag-less mapping.
4. `configs/example.yaml.example` and [cli.md](cli.md) learn the new kind.
5. Cross-object invariants? Extend `ExperimentConfig`'s validator, naming
   the kinds the check applies to.
6. `tests/test_config.py`: the shipped example still loads; the new kind
   discriminates; the invariants still hold.
