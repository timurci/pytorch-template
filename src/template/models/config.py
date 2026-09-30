"""Pydantic schemas for `models`' `nn.Module` definitions.

Each schema is the declarative form of one architecture: `kind` picks it
in YAML, `build(input_size)` constructs it. The input width is never
config — it is the schema's `feature_width`, resolved at wiring time.
"""

from typing import Annotated, Literal

from pydantic import Field

from template.config_pattern import ConfigModel, default_kind
from template.models.classifier import MLPClassifier


class MLPModelConfig(ConfigModel):
    kind: Literal["mlp"] = "mlp"
    hidden_size: int = Field(default=128, ge=1)
    hidden_depth: int = Field(default=2, ge=0)
    n_classes: int = Field(default=2, ge=2)
    dropout: float = Field(default=0.0, ge=0, lt=1)

    def build(self, input_size: int) -> MLPClassifier:
        return MLPClassifier(
            input_size,
            self.hidden_size,
            self.hidden_depth,
            self.n_classes,
            self.dropout,
        )


ModelConfig = Annotated[
    MLPModelConfig,
    Field(discriminator="kind"),
    default_kind("mlp"),
]
