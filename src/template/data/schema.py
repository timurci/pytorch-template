"""The declared roles of the columns a data source serves.

Roles over the *processed* columns: raw processing (`features.raw`) has
already run, so feature columns are numeric and each is one tensor column
wide. The target's declared class mapping (`features.raw.MapValues`) is the
one role-level encoding, applied at tensorization.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from template.features.raw import TargetEncoder


@dataclass(frozen=True)
class TableSchema:
    """What each processed column of a table is, and its tensor layout.

    Roles are projections, not partitions: a column may be both a feature
    and metadata (e.g. an id that is also a model input) — it then appears
    in the feature block and in the metadata side. `metadata_columns`
    identify samples and are kept untouched, `target_column` is the
    supervised target (never a feature; that would be leakage), excluded
    columns are gone, and everything else is a feature. Feature tensor
    blocks follow `feature_columns` order; raw processing has already made
    every feature numeric, so the feature count is `len(feature_columns)` —
    known before any row is read.

    `target_encoder` maps only a non-numeric target (declared class
    indices); a target without an encoder must be numeric.
    """

    feature_columns: tuple[str, ...]
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
        target_encoder: TargetEncoder | None = None,
    ) -> TableSchema:
        """Derive the schema from processed column names by role.

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
            target_column=target,
            target_encoder=target_encoder,
            metadata_columns=tuple(metadata),
        )

    @property
    def feature_width(self) -> int:
        """Total tensor width of one row's features."""
        return len(self.feature_columns)

    def feature_slice(self, column: str) -> slice:
        """Block of the feature tensor belonging to one feature column."""
        if column not in self.feature_columns:
            raise ValueError(f"not a feature column: {column!r}")
        index = self.feature_columns.index(column)
        return slice(index, index + 1)
