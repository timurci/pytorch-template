"""The declared roles and encodings of the columns a data source serves."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from template.data.encoding import FeatureEncoder, TargetEncoder


@dataclass(frozen=True)
class TableSchema:
    """What each raw column of a table is, and how it becomes a tensor.

    Roles are projections, not partitions: a column may be both a feature
    and metadata (e.g. an id that is also a model input) — it then appears
    in the feature block and in the metadata side. `metadata_columns`
    identify samples and are kept untouched, `target_column` is the
    supervised target (never a feature; that would be leakage), excluded
    columns are gone, and everything else is a feature.
    Feature tensor blocks follow `feature_columns` order; a feature
    column's block width is its encoder's `width` (1 for numeric
    passthrough), so the feature count is known before any row is read.

    `feature_encoders` / `target_encoder` map only the columns that need
    them (non-numeric values); a column without an encoder must be numeric.
    """

    feature_columns: tuple[str, ...]
    feature_encoders: Mapping[str, FeatureEncoder] = field(
        default_factory=dict
    )
    target_column: str | None = None
    target_encoder: TargetEncoder | None = None
    metadata_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Duplicates collapse, first occurrence wins: block layout follows
        # declaration order, so dedup is order-preserving (never a set).
        object.__setattr__(
            self,
            "feature_columns",
            tuple(dict.fromkeys(self.feature_columns)),
        )
        object.__setattr__(
            self,
            "metadata_columns",
            tuple(dict.fromkeys(self.metadata_columns)),
        )
        if not self.feature_columns:
            raise ValueError("schema must declare at least one feature column")
        unknown = set(self.feature_encoders) - set(self.feature_columns)
        if unknown:
            raise ValueError(
                f"encoders declared for non-feature columns: "
                f"{sorted(unknown)}"
            )
        if self.target_column in self.feature_columns:
            raise ValueError(
                f"target column {self.target_column!r} is a feature"
            )
        if self.target_column in self.metadata_columns:
            raise ValueError(
                f"target column {self.target_column!r} is metadata"
            )
        if self.target_encoder is not None and self.target_column is None:
            raise ValueError(
                "target_encoder declared without a target column"
            )

    @classmethod
    def from_columns(
        cls,
        columns: Sequence[str],
        *,
        target: str | None = None,
        metadata: Sequence[str] = (),
        exclude: Sequence[str] = (),
        feature_encoders: Mapping[str, FeatureEncoder] | None = None,
        target_encoder: TargetEncoder | None = None,
    ) -> TableSchema:
        """Derive the schema from raw column names by role.

        Features are what remains of `columns` after the metadata, target,
        and excluded columns, keeping raw order; declaring exclusions is
        enough, the feature set needs no list of its own. Derived roles are
        disjoint by construction; a column that is both feature and
        metadata needs direct construction.
        """
        columns = list(columns)
        metadata = list(dict.fromkeys(metadata))
        exclude = list(dict.fromkeys(exclude))
        missing = [
            name
            for name in (*metadata, *exclude, *([target] if target else []))
            if name not in columns
        ]
        if missing:
            raise KeyError(f"unknown columns declared: {sorted(missing)}")
        keep = set(metadata) | set(exclude)
        if target is not None:
            keep.add(target)
        if len(keep) != len(metadata) + len(exclude) + (target is not None):
            raise ValueError("metadata, target, and exclude must be disjoint")
        features = tuple(name for name in columns if name not in keep)
        return cls(
            feature_columns=features,
            feature_encoders=dict(feature_encoders or {}),
            target_column=target,
            target_encoder=target_encoder,
            metadata_columns=tuple(metadata),
        )

    @property
    def feature_width(self) -> int:
        """Total tensor width of one row's features."""
        return sum(self.width(name) for name in self.feature_columns)

    def width(self, column: str) -> int:
        """Tensor width of one feature column's block."""
        encoder = self.feature_encoders.get(column)
        return encoder.width if encoder is not None else 1

    def feature_slice(self, column: str) -> slice:
        """Block of the feature tensor belonging to one feature column."""
        if column not in self.feature_columns:
            raise ValueError(f"not a feature column: {column!r}")
        start = sum(
            self.width(name)
            for name in self.feature_columns[
                : self.feature_columns.index(column)
            ]
        )
        return slice(start, start + self.width(column))
