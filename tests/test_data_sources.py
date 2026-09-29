"""Tests for the ingestion seam: roles, encodings, validation, datasets."""

from collections.abc import Sequence

import polars as pl
import pytest

from template.data import (
    MapValues,
    OneHot,
    TableDataset,
    TableSchema,
    ValidatedSource,
    ValidatedTable,
    class_counts,
    partition_indices,
)
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


def _schema() -> TableSchema:
    return TableSchema.from_columns(
        ["id", "city", "age", "label"],
        target="label",
        metadata=["id"],
        feature_encoders={"city": OneHot(["a", "b", "c"])},
        target_encoder=MapValues({"neg": 0, "pos": 1}),
    )


def _source() -> ValidatedSource:
    return ValidatedSource(FrameSource(_frame()), _schema())


def test_from_columns_assigns_roles() -> None:
    schema = _schema()
    assert schema.feature_columns == ("city", "age")
    assert schema.metadata_columns == ("id",)
    assert schema.target_column == "label"
    # One-hot block plus one numeric column; known before any row is read.
    assert schema.feature_width == 4
    assert schema.feature_slice("city") == slice(0, 3)
    assert schema.feature_slice("age") == slice(3, 4)


def test_exclude_is_enough_to_shape_the_feature_set() -> None:
    schema = TableSchema.from_columns(
        ["id", "city", "age", "label"],
        target="label",
        metadata=["id"],
        exclude=["city"],
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


def test_one_hot_encode_and_uncovered_values() -> None:
    encoder = OneHot(["a", "b", "c"])
    encoded = encoder.encode(pl.Series("city", ["b", "a"]))
    assert encoded.shape == (2, 3)
    assert encoded.tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match="not covered"):
        encoder.encode(pl.Series("city", ["z"]))


def test_one_hot_levels_from_file(tmp_path) -> None:
    levels = tmp_path / "vocab.txt"
    levels.write_text("a\nb\n")
    encoder = OneHot(levels_path=levels)
    assert encoder.width == 2
    assert encoder.encode(pl.Series("city", ["b"])).tolist() == [[0.0, 1.0]]


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
    # city "a" -> one-hot block, then scaled-age column at index 3.
    assert row["features"].tolist() == [1.0, 0.0, 0.0, 20.0]
    assert row["targets"].item() == 0
    batched = dataset.__getitems__([3, 1, 1])
    assert [item["features"][3].item() for item in batched] == [50.0, 30.0, 30.0]
    assert [item["targets"].item() for item in batched] == [1, 1, 1]


def test_metadata_rides_through_untouched() -> None:
    source = _source()
    ids = source.read([5, 2]).frame.get_column("id")
    assert ids.to_list() == [5, 2]


def test_extra_columns_are_ignored() -> None:
    frame = _frame().with_columns(pl.Series("note", ["x"] * 8))
    source = ValidatedSource(FrameSource(frame), _schema())
    assert TableDataset(source)[0]["features"].shape == (4,)


def test_validation_rejects_missing_and_non_numeric() -> None:
    schema = _schema()
    with pytest.raises(KeyError):
        ValidatedTable(_frame().drop("age"), schema)
    bare = TableSchema(feature_columns=("city",))
    with pytest.raises(TypeError, match="need an encoder"):
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
