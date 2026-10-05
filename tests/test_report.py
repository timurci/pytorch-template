"""`template-report`: train, then score the written checkpoint end-to-end."""

from pathlib import Path

import polars as pl
import pytest

from template.cli import report, train

_CONFIG = """\
data:
  train: {kind: csv, path: data/raw/train.csv}
  test: {kind: csv, path: data/raw/test.csv}
  id_column: id
  target:
    column: label
    mapping: {negative: 0, positive: 1}
model:
  kind: mlp
  hidden_size: 8
  hidden_depth: 1
  n_classes: 2
optimizer:
  kind: sgd
  lr: 0.1
training:
  epochs: 2
  batch_size: 16
  val_fraction: 0.5
  seed: 7
  trackers:
    - kind: "null"
  checkpoints:
    - {kind: recovery, path: data/model/recovery.pt}
    - {kind: best, path: data/model/best.pt}
"""

_REPORT = "data/report/report.md"

_TEST_ROWS = 12
_TEST_NEGATIVE = 8
_TEST_POSITIVE = 4


def _dataset(rows: int) -> pl.DataFrame:
    features = [((index % 10) - 5) / 5 for index in range(rows)]
    return pl.DataFrame(
        {
            "id": [f"row-{index}" for index in range(rows)],
            "x": features,
            "y": [float((index * 7) % 3) for index in range(rows)],
            "label": [
                "positive" if value > 0 else "negative" for value in features
            ],
        }
    )


def _write_inputs(directory: Path, *, config: str = _CONFIG) -> None:
    (directory / "data" / "raw").mkdir(parents=True)
    _dataset(40).write_csv(directory / "data" / "raw" / "train.csv")
    _dataset(_TEST_ROWS).write_csv(directory / "data" / "raw" / "test.csv")
    (directory / "config.yaml").write_text(config)


def test_report_scores_the_checkpoint_written_by_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_inputs(tmp_path)

    train.main(["--config", "config.yaml"])
    report.main(["--config", "config.yaml"])

    text = (tmp_path / _REPORT).read_text()
    assert (
        f"rows: {_TEST_ROWS} ({_TEST_NEGATIVE} negative, "
        f"{_TEST_POSITIVE} positive)" in text
    )
    assert "| roc_auc |" in text
    assert "| average_precision |" in text
    assert "| predicted negative |" in text
    assert "| predicted positive |" in text
    # The default rule scored the best file (present after training).
    assert "- checkpoint: `" in text
    assert "data/model/best.pt" in text


def test_report_can_score_an_explicit_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_inputs(tmp_path)

    train.main(["--config", "config.yaml"])
    report.main(
        ["--config", "config.yaml", "--checkpoint", "data/model/recovery.pt"]
    )

    text = (tmp_path / _REPORT).read_text()
    assert "data/model/recovery.pt" in text


def test_report_rejects_a_non_binary_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _CONFIG.replace(
        "mapping: {negative: 0, positive: 1}",
        "mapping: {a: 0, b: 1, c: 2}",
    ).replace("n_classes: 2", "n_classes: 3")
    _write_inputs(tmp_path, config=config)

    with pytest.raises(ValueError, match="binary"):
        report.main(["--config", "config.yaml"])
