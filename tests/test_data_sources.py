"""Tests for the ingestion seam: raw processing, roles, validation, datasets."""

from collections.abc import Sequence

import polars as pl
import pytest

from template.data import (
    ProcessedSource,
    TableDataset,
    TableSchema,
    ValidatedSource,
    ValidatedTable,
    class_counts,
    partition_indices,
)
from template.features.raw import MapValues, OneHot, RawPipeline
from template.persistence import CsvSource, save_table


class FrameSource:
    """In-memory `RawTableSource` test double."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    @property
    def columns(self) -> Sequence[str]:
        return self._frame.columns

    def count(self) -> int:
        return self._frame.height

    def read(self, indices: Sequence[int]) -> pl.DataFrame:
        return self._frame[list(indices)]


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7],
            "city": ["a", "b", "c", "a", "b", "c", "a", "b"],
            "age": [20.0, 30.0, 40.0, 50.0, 25.0, 35.0, 45.0, 55.0],
            "label": ["neg", "pos", "neg", "pos", "neg", "pos", "neg", "pos"],
        }
    )


def _processed() -> ProcessedSource:
    return ProcessedSource(
        FrameSource(_frame()),
        RawPipeline([OneHot("city", ["a", "b", "c"])], _frame().columns),
    )


def _schema(columns: Sequence[str]) -> TableSchema:
    return TableSchema.from_columns(
        columns,
        target="label",
        metadata=["id"],
        target_encoder=MapValues({"neg": 0, "pos": 1}),
    )


def _source() -> ValidatedSource:
    processed = _processed()
    return ValidatedSource(processed, _schema(processed.columns))


def test_processed_source_exposes_declared_output_columns() -> None:
    processed = _processed()
    assert processed.columns == (
        "id",
        "age",
        "label",
        "city=a",
        "city=b",
        "city=c",
    )
    assert processed.count() == 8
    # The pipeline's declaration equals what a read actually produces.
    assert processed.read([0]).columns == list(processed.columns)


def test_from_columns_assigns_roles() -> None:
    processed = _processed()
    schema = _schema(processed.columns)
    assert schema.feature_columns == ("age", "city=a", "city=b", "city=c")
    assert schema.metadata_columns == ("id",)
    assert schema.target_column == "label"
    # One-hot levels plus one numeric column; known before any row is read.
    assert schema.feature_width == 4
    assert schema.feature_slice("age") == slice(0, 1)
    assert schema.feature_slice("city=c") == slice(3, 4)


def test_exclude_is_enough_to_shape_the_feature_set() -> None:
    schema = TableSchema.from_columns(
        ["id", "age", "label", "city=a"],
        target="label",
        metadata=["id"],
        exclude=["city=a"],
    )
    assert schema.feature_columns == ("age",)
    with pytest.raises(KeyError):
        TableSchema.from_columns(["age"], target="label")


def test_column_may_be_both_feature_and_metadata() -> None:
    # The duality: an id that is also a model input projects into both.
    schema = TableSchema(
        feature_columns=("user_id", "age"),
        metadata_columns=("user_id",),
    )
    assert schema.feature_slice("user_id") == slice(0, 1)
    assert "user_id" in schema.metadata_columns


def test_target_is_never_a_feature() -> None:
    with pytest.raises(ValueError):
        TableSchema(feature_columns=("label",), target_column="label")


def test_duplicate_columns_collapse_preserving_order() -> None:
    schema = TableSchema(
        feature_columns=("age", "city", "age"),
        metadata_columns=("id", "id"),
    )
    assert schema.feature_columns == ("age", "city")
    assert schema.metadata_columns == ("id",)
    assert schema.feature_width == 2


def test_target_encoder_requires_a_target_column() -> None:
    with pytest.raises(ValueError, match="target_encoder"):
        TableSchema(
            feature_columns=("age",),
            target_encoder=MapValues({"neg": 0, "pos": 1}),
        )


def test_map_values_encodes_targets() -> None:
    encoder = MapValues({"neg": 0, "pos": 1})
    assert encoder.encode(pl.Series("label", ["pos", "neg"])).tolist() == [
        1,
        0,
    ]
    with pytest.raises(ValueError, match="not covered"):
        encoder.encode(pl.Series("label", ["maybe"]))


def test_validated_source_serves_encoded_batches() -> None:
    source = _source()
    assert source.count() == 8
    dataset = TableDataset(source)
    row = dataset[0]
    # city "a" -> 1.0 in city=a, then the age column at index 0.
    assert row["features"].tolist() == [20.0, 1.0, 0.0, 0.0]
    assert row["targets"].item() == 0
    batched = dataset.__getitems__([3, 1, 1])
    assert [item["features"][0].item() for item in batched] == [50.0, 30.0, 30.0]
    assert [item["targets"].item() for item in batched] == [1, 1, 1]


def test_metadata_rides_through_untouched() -> None:
    source = _source()
    ids = source.read([5, 2]).frame.get_column("id")
    assert ids.to_list() == [5, 2]


def test_columns_outside_the_roles_are_ignored() -> None:
    processed = _processed()
    schema = TableSchema.from_columns(
        processed.columns,
        target="label",
        metadata=["id"],
        exclude=["age"],
        target_encoder=MapValues({"neg": 0, "pos": 1}),
    )
    source = ValidatedSource(processed, schema)
    assert TableDataset(source)[0]["features"].shape == (3,)


def test_validation_rejects_missing_and_non_numeric() -> None:
    processed = _processed()
    schema = _schema(processed.columns)
    with pytest.raises(KeyError):
        ValidatedTable(_frame().drop("age"), schema)
    bare = TableSchema(feature_columns=("city",))
    with pytest.raises(TypeError, match="raw step"):
        ValidatedTable(_frame(), bare)


def test_partition_indices_covers_seeds_and_rejects_bad_fractions() -> None:
    train, val = partition_indices(100, 0.2, seed=42)
    assert len(train) == 80
    assert len(val) == 20
    assert set(train) | set(val) == set(range(100))
    assert not set(train) & set(val)
    assert partition_indices(100, 0.2, seed=42) == (train, val)
    assert partition_indices(100, 0.2, seed=43)[0] != train
    with pytest.raises(ValueError):
        partition_indices(100, 0.0, seed=42)
    with pytest.raises(ValueError):
        partition_indices(3, 0.1, seed=42)  # would leave an empty partition


def test_class_counts_reads_through_the_port() -> None:
    source = _source()
    assert class_counts(source, range(8)) == {0: 4, 1: 4}
    assert class_counts(source, [1, 3, 5, 7]) == {0: 0, 1: 4}
    unlabeled = ValidatedSource(
        FrameSource(_frame().drop("label")),
        TableSchema(feature_columns=("age",), metadata_columns=("id",)),
    )
    with pytest.raises(ValueError, match="no target column"):
        class_counts(unlabeled, range(8))


def test_csv_source_round_trips_through_save_table(tmp_path) -> None:
    frame = _frame().drop("label")
    path = tmp_path / "table.csv"
    save_table(frame, path)
    source = CsvSource(path)
    assert list(source.columns) == ["id", "city", "age"]
    assert source.count() == 8
    assert source.read([1, 3]).equals(frame[[1, 3]])
