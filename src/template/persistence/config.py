"""Pydantic schemas for the raw readers of `persistence`.

Each schema is the declarative form of one reader: `kind` picks the
implementation in YAML, `build()` constructs the plain reader. Which path
feeds which run is wiring and stays in the CLI. Built readers satisfy
`data.RawTableSource` structurally — no import edge, same as the readers
themselves.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from template.config_pattern import ConfigModel
from template.persistence.sources import CsvSource, ParquetSource


class CsvSourceConfig(ConfigModel):
    kind: Literal["csv"] = "csv"
    path: Path

    def build(self) -> CsvSource:
        return CsvSource(self.path)


class ParquetSourceConfig(ConfigModel):
    kind: Literal["parquet"] = "parquet"
    path: Path

    def build(self) -> ParquetSource:
        return ParquetSource(self.path)


RawSourceConfig = Annotated[
    CsvSourceConfig | ParquetSourceConfig,
    Field(discriminator="kind"),
]
