"""Raw table readers plus table/checkpoint save-load helpers.

Readers (`CsvSource`, `ParquetSource`) load a whole table at
construction and serve unvalidated row frames by index; they implement the
raw side of the composition seam (`data.RawTableSource`), leaving raw
processing, roles, and validation to `features.raw` / `data.TableSchema` /
`data.ValidatedSource`. Save/load helpers write to caller-supplied paths.
"""

from template.persistence.io import (
    load_checkpoint,
    save_checkpoint,
    save_table,
)
from template.persistence.sources import (
    CsvSource,
    ParquetSource,
)

__all__ = [
    "CsvSource",
    "ParquetSource",
    "load_checkpoint",
    "save_checkpoint",
    "save_table",
]
