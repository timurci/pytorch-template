"""Shared pieces of the config pattern: strict schemas and slot default kinds.

`ConfigModel` makes unknown YAML keys load errors in every layer's config
module instead of pydantic's default of ignoring them. `default_kind` gives
a kind-tagged slot a default: the slot's `Annotated` alias names one kind,
and a tag-less mapping is read as that kind instead of failing tag
dispatch. The pattern and its recipes: docs/config-pattern.md.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, BeforeValidator, ConfigDict


class ConfigModel(BaseModel):
    """Base for every config schema: unknown keys are errors, not ignored."""

    model_config = ConfigDict(extra="forbid")


def default_kind(kind: str) -> BeforeValidator:
    """A union-slot validator: inject `kind` before tag dispatch.

    Goes after `Field(discriminator="kind")` in a slot's `Annotated` alias,
    so a tag-less mapping validates as the slot's default kind while every
    other mapping (and every other kind) still dispatches by tag.
    """

    def inject(value: Any) -> Any:
        match value:
            case Mapping() if "kind" not in value:
                return {"kind": kind, **value}
            case _:
                return value

    return BeforeValidator(inject)
