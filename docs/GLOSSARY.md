# PyTorch training template

The language for a training run and the checkpoints it leaves behind.

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
