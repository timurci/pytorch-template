"""The role-declared column schema and the torch-side dataset adapter.

Raw processing — the pipeline of steps over raw columns — lives in
`features.raw`; `ProcessedSource` here applies it, `ValidatedSource` checks
the roles, and `TableDataset` tensorizes.
"""

from template.data.schema import TableSchema
from template.data.source import DataSource, RawTableSource
from template.data.table import (
    Batch,
    ProcessedSource,
    TableDataset,
    ValidatedSource,
    ValidatedTable,
    class_counts,
    partition_indices,
)

__all__ = [
    "Batch",
    "DataSource",
    "ProcessedSource",
    "RawTableSource",
    "TableDataset",
    "TableSchema",
    "ValidatedSource",
    "ValidatedTable",
    "class_counts",
    "partition_indices",
]
