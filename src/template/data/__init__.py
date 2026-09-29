"""The role-declared schema, ingestion encoders, and torch dataset adapter."""

from template.data.encoding import MapValues, OneHot
from template.data.schema import TableSchema
from template.data.source import DataSource, RawTableSource
from template.data.table import (
    Batch,
    TableDataset,
    ValidatedSource,
    ValidatedTable,
    class_counts,
    partition_indices,
)

__all__ = [
    "Batch",
    "DataSource",
    "MapValues",
    "OneHot",
    "RawTableSource",
    "TableDataset",
    "TableSchema",
    "ValidatedSource",
    "ValidatedTable",
    "class_counts",
    "partition_indices",
]
