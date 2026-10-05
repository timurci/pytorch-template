# PyTorch training template

The language for a training run, its preprocessing, and the checkpoints it
leaves behind.

## Language

**Training state**:
The information required to continue a training run from a finished epoch.
_Avoid_: metrics, snapshot

**Checkpoint**:
A durable record of one finished epoch.
_Avoid_: snapshot, weights file

**Checkpoint strategy**:
A rule for which finished epochs to retain and which parts of the training state a checkpoint holds.
_Avoid_: checkpoint handler, checkpoint policy

**Recovery checkpoint**:
A checkpoint from which an interrupted training run can continue.
_Avoid_: latest checkpoint

**Best checkpoint**:
The checkpoint retained as the preferred model of the run so far.
_Avoid_: early-stopping checkpoint

**Raw processing**:
A `RawPipeline` of `RawStep`s over the named raw columns (`features/raw.py`),
applied per read on the CPU, before any `Batch` exists.
_Avoid_: offline preprocessing, feature engineering, encoding

**Tensor processing**:
A `TensorPipeline` of `TensorStep`s over batches (`features/tensor.py`),
applied on the target device during training and inference; preferred over raw
processing for any transform that can wait, because it is vectorized and runs
on the accelerator.
_Avoid_: online preprocessing

**Augmentation**:
Stochastic tensor processing: a step that draws from the pipeline `rng` and is
tagged for the stages it runs in (typically `train`).
_Avoid_: data augmentation pipeline
