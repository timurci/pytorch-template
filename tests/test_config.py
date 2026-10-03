"""The config pattern: tagged, buildable schemas over one YAML file."""

from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from template.cli.config import (
    AdamWConfig,
    CosineAnnealingLRConfig,
    CrossEntropyLossConfig,
    DataConfig,
    ExperimentConfig,
    StepLRConfig,
    load_config,
)
from template.models.config import MLPModelConfig
from template.persistence.config import CsvSourceConfig, ParquetSourceConfig

_TEMPLATE = Path(__file__).parent.parent / "configs" / "example.yaml.example"


def _data(
    train: dict[str, Any] | None = None,
    encodings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    default_train = {"kind": "csv", "path": "data/train.csv"}
    raw: dict[str, Any] = {
        "train": default_train if train is None else train,
        "test": {"kind": "csv", "path": "data/test.csv"},
        "target": {
            "column": "label",
            "mapping": {"negative": 0, "positive": 1},
        },
    }
    if encodings is not None:
        raw["encodings"] = encodings
    return raw


def _experiment(**sections: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "data": _data(),
        "model": {"kind": "mlp"},
        "training": {"checkpoint_path": "outputs/model.pt"},
    }
    raw.update(sections)
    return raw


def test_shipped_example_config_still_loads() -> None:
    config = load_config(_TEMPLATE)
    assert isinstance(config.model, MLPModelConfig)
    assert isinstance(config.data.train, CsvSourceConfig)
    assert isinstance(config.data.test, CsvSourceConfig)


def test_kind_picks_the_variant_in_yaml() -> None:
    config = ExperimentConfig.model_validate(
        _experiment(data=_data({"kind": "parquet", "path": "data/train.parquet"}))
    )
    assert isinstance(config.data.train, ParquetSourceConfig)
    assert isinstance(config.data.test, CsvSourceConfig)


def test_unknown_kind_is_a_load_error() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            _experiment(model={"kind": "autoencoder", "latent_dim": 8})
        )


def test_unknown_keys_are_load_errors() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            _experiment(model={"kind": "mlp", "hidden_sze": 256})
        )


def test_kind_may_be_omitted_where_the_slot_declares_a_default() -> None:
    config = ExperimentConfig.model_validate(
        _experiment(
            model={},
            optimizer={"lr": 0.5},
            loss={"class_weights": "balanced"},
        )
    )
    assert isinstance(config.model, MLPModelConfig)
    assert isinstance(config.optimizer, AdamWConfig)
    assert config.optimizer.lr == 0.5
    assert isinstance(config.loss, CrossEntropyLossConfig)


@pytest.mark.parametrize(
    "sections",
    [
        pytest.param({"data": _data({"path": "data/train.csv"})}, id="source"),
        pytest.param(
            {"data": _data(encodings=[{"column": "city", "levels": ["a"]}])},
            id="encoding",
        ),
        pytest.param(
            {"processing": {"steps": [{"column": "age", "cap": 100.0}]}},
            id="step",
        ),
        pytest.param(
            {"training": {"checkpoint_path": "m.pt", "trackers": [{}]}},
            id="tracker",
        ),
        pytest.param({"scheduler": {"step_size": 1}}, id="scheduler"),
    ],
)
def test_kind_is_required_where_the_slot_has_no_default(
    sections: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_experiment(**sections))


def test_scheduler_is_optional() -> None:
    config = ExperimentConfig.model_validate(_experiment())
    assert config.scheduler is None


def test_scheduler_kind_builds_the_torch_scheduler() -> None:
    step = ExperimentConfig.model_validate(
        _experiment(scheduler={"kind": "step", "step_size": 2, "gamma": 0.5})
    )
    cosine = ExperimentConfig.model_validate(
        _experiment(scheduler={"kind": "cosine", "t_max": 10})
    )
    assert isinstance(step.scheduler, StepLRConfig)
    assert isinstance(cosine.scheduler, CosineAnnealingLRConfig)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    built_step = step.scheduler.build(optimizer)
    built_cosine = cosine.scheduler.build(optimizer)
    assert isinstance(built_step, StepLR)
    assert built_step.step_size == 2
    assert built_step.gamma == 0.5
    assert isinstance(built_cosine, CosineAnnealingLR)
    assert built_cosine.T_max == 10
    assert built_cosine.eta_min == 0.0


def test_multistep_milestones_must_be_strictly_increasing() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        ExperimentConfig.model_validate(
            _experiment(scheduler={"kind": "multistep", "milestones": [4, 2]})
        )


def test_target_mapping_must_match_the_model_head() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            _experiment(model={"kind": "mlp", "n_classes": 3})
        )


def test_inference_schema_excludes_the_target() -> None:
    data = DataConfig.model_validate(_data())
    labelled = data.build_schema(["id", "label", "age"], include_target=False)
    assert labelled.target_column is None
    assert labelled.target_encoder is None
    assert labelled.feature_columns == ("age",)
    assert labelled.metadata_columns == ("id",)
    unlabelled = data.build_schema(["id", "age"], include_target=False)
    assert unlabelled.target_column is None


def test_balanced_weights_need_contiguous_nonzero_class_counts() -> None:
    loss = CrossEntropyLossConfig(class_weights="balanced")
    with pytest.raises(ValueError):
        loss.resolve_weights({0: 5, 2: 3})
    with pytest.raises(ValueError):
        loss.resolve_weights({0: 5, 1: 0})


@pytest.mark.parametrize(
    "sections",
    [
        pytest.param(
            {
                "data": _data(
                    encodings=[
                        {
                            "kind": "one_hot",
                            "column": "city",
                            "levels": ["a"],
                            "levels_path": "vocab.txt",
                        }
                    ]
                )
            },
            id="one-hot-two-level-sources",
        ),
        pytest.param(
            {
                "processing": {
                    "steps": [
                        {
                            "kind": "scale_by_cap",
                            "column": "age",
                            "cap": 0.0,
                            "floor": 0.0,
                        }
                    ]
                }
            },
            id="scale-cap-not-above-floor",
        ),
        pytest.param(
            {
                "processing": {
                    "steps": [
                        {
                            "kind": "log_scale_by_cap",
                            "column": "age",
                            "cap": 1.0,
                        }
                    ]
                }
            },
            id="log-cap-not-above-one",
        ),
        pytest.param(
            {
                "processing": {
                    "steps": [
                        {
                            "kind": "gaussian_noise",
                            "std": 0.01,
                            "stages": [],
                        }
                    ]
                }
            },
            id="no-stages",
        ),
    ],
)
def test_declaration_invariants_fail_at_load(sections: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_experiment(**sections))
