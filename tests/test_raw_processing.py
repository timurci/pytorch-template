"""Tests for raw processing: frame steps over named columns."""

from collections.abc import Sequence

import polars as pl
import pytest

from template.features.raw import DropColumns, OneHot, RawPipeline


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [0, 1, 2],
            "city": ["a", "b", "c"],
            "age": [20.0, 30.0, 40.0],
        }
    )


def test_one_hot_replaces_the_column_with_named_level_columns() -> None:
    step = OneHot("city", ["a", "b", "c"])
    assert step.columns(["id", "city", "age"]) == (
        "id",
        "age",
        "city=a",
        "city=b",
        "city=c",
    )
    out = step.process(_frame())
    assert out.columns == ["id", "age", "city=a", "city=b", "city=c"]
    assert out.get_column("city=a").dtype == pl.Float32
    assert out.get_column("city=b").to_list() == [0.0, 1.0, 0.0]


def test_one_hot_levels_from_file(tmp_path) -> None:
    levels = tmp_path / "vocab.txt"
    levels.write_text("a\nb\n")
    step = OneHot("city", levels_path=levels)
    out = step.process(pl.DataFrame({"city": ["b", "a"]}))
    assert out.columns == ["city=a", "city=b"]
    assert out.get_column("city=b").to_list() == [1.0, 0.0]


def test_one_hot_rejects_uncovered_values() -> None:
    with pytest.raises(ValueError, match="not covered"):
        OneHot("city", ["a", "b"]).process(pl.DataFrame({"city": ["z"]}))


def test_one_hot_requires_its_column() -> None:
    with pytest.raises(ValueError, match="not an input column"):
        OneHot("city", ["a"]).columns(["age"])


def test_drop_columns_selects_by_name() -> None:
    step = DropColumns(["city", "note"])
    assert step.columns(["id", "city", "age", "note"]) == ("id", "age")
    with pytest.raises(ValueError, match="unknown columns to drop"):
        step.columns(["id", "age"])
    frame = _frame().with_columns(pl.Series("note", ["x", "y", "z"]))
    assert step.process(frame).columns == ["id", "age"]


def test_pipeline_resolves_output_columns_without_rows() -> None:
    pipeline = RawPipeline(
        [OneHot("city", ["a", "b", "c"]), DropColumns(["id"])],
        ["id", "city", "age"],
    )
    assert pipeline.output_columns == ("age", "city=a", "city=b", "city=c")
    assert pipeline.process(_frame()).columns == [
        "age",
        "city=a",
        "city=b",
        "city=c",
    ]


def test_pipeline_rejects_duplicate_output_columns() -> None:
    class _Duplicate:
        def columns(self, inputs: Sequence[str]) -> tuple[str, ...]:
            return ("same", "same")

        def process(self, frame: pl.DataFrame) -> pl.DataFrame:
            return frame

    with pytest.raises(ValueError, match="duplicate"):
        RawPipeline([_Duplicate()], ["x"])
