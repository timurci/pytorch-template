"""The feature layer: preprocessing, split into raw and tensor.

Two clearly separated kinds of preprocessing live here, one per file:

- `features.raw` — **raw processing**: a `RawPipeline` of `RawStep`s over
  the named `polars` columns, applied per read on the CPU, before any
  `Batch` exists. It is the only place raw values and column names are
  visible, so feature selection, renaming, deriving, and encoding live here.
- `features.tensor` — **tensor processing**: a `TensorPipeline` of ordered,
  stage-tagged, seeded `Batch` -> `Batch` steps applied on the target device
  during training and inference.

Prefer tensor processing for any transform that can wait: it is vectorized,
runs on the accelerator, and is reproducible from a seed.

This package re-exports nothing on purpose: `features.tensor` imports
`data.Batch`, while `data` imports `features.raw` for the pipeline it
applies, so a package-level re-export of both would make the import order
matter. Import the module you need explicitly (`from
template.features.tensor import TensorPipeline`).
"""
