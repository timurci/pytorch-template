"""Pydantic schemas for `data`'s declared encoders and target mapping.

Declarative state, mirroring the encoders' own rule: levels and class
indices are declared here (or in the artifact a schema points at), never
fit from the data they encode.
"""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from template.config_pattern import ConfigModel
from template.data.encoding import MapValues, OneHot


class OneHotEncodingConfig(ConfigModel):
    kind: Literal["one_hot"] = "one_hot"
    column: str
    levels: list[str] | None = None
    levels_path: Path | None = None

    @model_validator(mode="after")
    def _validate_levels_source(self) -> Self:
        if (self.levels is None) == (self.levels_path is None):
            raise ValueError("declare exactly one of levels or levels_path")
        if self.levels is not None:
            if not self.levels:
                raise ValueError("levels must not be empty")
            if len(set(self.levels)) != len(self.levels):
                raise ValueError("levels must be unique")
        return self

    def build(self) -> OneHot:
        return OneHot(self.levels, levels_path=self.levels_path)


FeatureEncodingConfig = Annotated[
    OneHotEncodingConfig,
    Field(discriminator="kind"),
]


class TargetConfig(ConfigModel):
    column: str
    mapping: dict[str, int]

    @model_validator(mode="after")
    def _validate_mapping(self) -> Self:
        if not self.mapping:
            raise ValueError("target mapping must not be empty")
        if len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError("target mapping must be injective")
        return self

    def build(self) -> MapValues:
        return MapValues(self.mapping)
