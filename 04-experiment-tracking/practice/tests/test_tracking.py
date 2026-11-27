"""Проверки TODO 1–3: параметры, результаты и восстановление модели."""

from pathlib import Path

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest

import tracking
import train


@pytest.fixture
def experiment(tmp_path):
    # Each test writes to its own database, not to your seminar history.
    tracking.configure(tmp_path / "state")
    features = pd.DataFrame({
        "Pclass": [1, 3] * 20, "Sex": ["female", "male"] * 20,
        "Age": [25.0, 30.0] * 20, "SibSp": [0] * 40,
        "Parch": [0] * 40, "Fare": [60.0, 10.0] * 20,
        "Embarked": ["S"] * 40,
    })
    for name in ("Pclass", "SibSp", "Parch"):
        features[name] = features[name].astype(float)
    labels = pd.Series([1, 0] * 20, name="Survived")
    model = train.fit_model(features, labels, features, labels, depth=2,
                            iterations=8, train_dir=tmp_path / "tensorboard")
    return features, labels, model


def test_parameters_and_data_identity_are_saved(experiment):
    with mlflow.start_run() as run:
        tracking.log_parameters({"depth": 2, "split_seed": 42},
                                "0123456789abcdef", 32, 8)
    saved = MlflowClient().get_run(run.info.run_id)
    assert saved.data.params["depth"] == "2"
    assert saved.data.params["split_seed"] == "42"
    assert saved.data.params["train_rows"] == "32"
    assert saved.data.params["validation_rows"] == "8"
    assert saved.data.tags["data_sha256"] == "0123456789abcdef"


def test_results_include_metric_and_reloadable_model(experiment):
    features, _, model = experiment
    with mlflow.start_run() as run:
        model_uri = tracking.log_results(model, 0.875, features.head(2))
    saved = MlflowClient().get_run(run.info.run_id)
    assert saved.data.metrics["validation_roc_auc"] == 0.875
    restored = mlflow.catboost.load_model(model_uri)
    np.testing.assert_allclose(restored.predict_proba(features),
                               model.predict_proba(features), atol=1e-12)


def test_loading_run_preserves_positive_class_probabilities(experiment):
    features, _, model = experiment
    with mlflow.start_run() as run:
        tracking.log_results(model, 0.875, features.head(2))
    restored = tracking.load_model(run.info.run_id)
    np.testing.assert_allclose(restored.predict_proba(features)[:, 1],
                               model.predict_proba(features)[:, 1], atol=1e-12)


def test_features_exclude_target_and_do_not_mutate_input():
    frame = pd.DataFrame({"Pclass": [3], "Sex": [None], "Age": [float("nan")],
                          "SibSp": [0], "Parch": [0], "Fare": [8.0],
                          "Embarked": [None], "Survived": [1], "PassengerId": [42]})
    before = frame.copy(deep=True)
    actual = train.prepare_features(frame)
    assert list(actual.columns) == train.FEATURES
    assert actual.Sex.tolist() == ["unknown"]
    assert actual.Embarked.tolist() == ["unknown"]
    pd.testing.assert_frame_equal(frame, before)


def test_missing_features_are_rejected():
    with pytest.raises(ValueError, match="Missing columns"):
        train.prepare_features(pd.DataFrame({"Pclass": [3]}))


def test_three_runs_use_same_split_and_generate_tensorboard_files(tmp_path):
    state = tmp_path / "state"
    results = [train.train_experiment(train.DEFAULT_DATA, state=state,
               depth=depth, iterations=12, run_name=f"depth-{depth}")
               for depth in (2, 4, 6)]
    client = MlflowClient()
    runs = [client.get_run(result["run_id"]) for result in results]
    assert len({run.info.run_id for run in runs}) == 3
    assert {run.data.params["split_seed"] for run in runs} == {"42"}
    assert {run.data.params["train_rows"] for run in runs} == {"570"}
    assert {run.data.params["validation_rows"] for run in runs} == {"143"}
    assert len({run.data.tags["data_sha256"] for run in runs}) == 1
    assert all(0 <= run.data.metrics["validation_roc_auc"] <= 1 for run in runs)
    assert all(run.info.status == "FINISHED" for run in runs)
    assert [Path(result["tensorboard_dir"]).name for result in results] == [
        "depth-2", "depth-4", "depth-6",
    ]
    for result in results:
        assert list(Path(result["tensorboard_dir"]).rglob("events.out.tfevents*"))


def test_repeated_run_names_keep_separate_tensorboard_events(tmp_path):
    state = tmp_path / "state"
    first = train.train_experiment(state=state, depth=2, iterations=8)
    first_dir = Path(first["tensorboard_dir"])
    events = {path: path.read_bytes() for path in first_dir.rglob("events.out.tfevents*")}
    second = train.train_experiment(state=state, depth=2, iterations=8)
    assert first["run_id"] != second["run_id"]
    assert first_dir.name == "depth-2"
    assert Path(second["tensorboard_dir"]).name == "depth-2-2"
    assert events
    assert all(path.read_bytes() == content for path, content in events.items())
    assert list(Path(second["tensorboard_dir"]).rglob("events.out.tfevents*"))
