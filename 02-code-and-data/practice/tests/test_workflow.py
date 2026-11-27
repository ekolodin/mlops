"""Small, independently specified fixtures; tests never need the network."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def api():
    assert importlib.util.find_spec("taxi_duration"), (
        "Implement the taxi_duration package"
    )
    from taxi_duration import data, model

    return data, model


def labeled(month=1):
    start = pd.date_range(f"2021-{month:02d}-01", periods=8, freq="h")
    return pd.DataFrame(
        {
            "lpep_pickup_datetime": start,
            "lpep_dropoff_datetime": start
            + pd.to_timedelta([2, 4, 2, 4, 2, 4, 2, 4], unit="min"),
            "PULocationID": [1, 2, 1, 2, 1, 2, 1, 2],
            "DOLocationID": [3] * 8,
        }
    )


def test_training_filter_includes_both_boundaries_without_mutation(api):
    data, _ = api
    frame = labeled().iloc[:6].copy()
    frame["lpep_dropoff_datetime"] = frame.lpep_pickup_datetime + pd.to_timedelta(
        [59, 60, 3600, 3601, -1, float("nan")], unit="s"
    )
    before = frame.copy(deep=True)
    actual = data.prepare_training_data(frame)
    assert actual.duration.tolist() == [1.0, 60.0]
    assert actual.index.tolist() == [1, 2]
    pd.testing.assert_frame_equal(frame, before)


def test_inference_needs_no_target_and_normalizes_missing_ids(api):
    data, _ = api
    frame = pd.DataFrame({"PULocationID": [1.0, None], "DOLocationID": [2, 3]})
    before = frame.copy(deep=True)
    assert data.prepare_features(frame) == [
        {"PULocationID": "1", "DOLocationID": "2"},
        {"PULocationID": "-1", "DOLocationID": "3"},
    ]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("bad", [1.5, float("inf"), "not-a-zone", -2, 266])
def test_invalid_zone_is_not_silently_truncated(api, bad):
    data, _ = api
    with pytest.raises(ValueError):
        data.prepare_features(
            pd.DataFrame({"PULocationID": [bad], "DOLocationID": [2]})
        )


@pytest.mark.parametrize("bad", [True, False, np.bool_(True), np.bool_(False)])
def test_boolean_zone_is_not_treated_as_an_integer(api, bad):
    data, _ = api
    with pytest.raises(ValueError):
        data.prepare_features(
            pd.DataFrame({"PULocationID": [bad], "DOLocationID": [2]})
        )


def test_missing_required_feature_is_an_error(api):
    data, _ = api
    with pytest.raises(ValueError):
        data.prepare_features(pd.DataFrame({"PULocationID": [1]}))


def test_validation_does_not_refit_vectorizer(api):
    _, model = api
    validation = labeled(2)
    validation.loc[0, "PULocationID"] = 200
    bundle, metrics = model.train_model(labeled(), validation)
    assert "PULocationID=200" not in bundle["vectorizer"].vocabulary_
    assert metrics["train_rows"] == metrics["validation_rows"] == 8
    assert metrics["baseline_rmse"] == pytest.approx(1.0)
    assert np.isfinite(metrics["validation_rmse"])


def test_model_learns_signal_and_serialization_preserves_predictions(api, tmp_path):
    _, model = api
    bundle, metrics = model.train_model(labeled(), labeled(2))
    assert metrics["validation_rmse"] < 1e-6
    inputs = pd.DataFrame({"PULocationID": [2, 1], "DOLocationID": [3, 3]})
    expected = [4.0, 2.0]
    np.testing.assert_allclose(model.predict(bundle, inputs), expected)
    path = tmp_path / "model.pkl"
    model.save_model(bundle, path)
    np.testing.assert_allclose(model.predict(model.load_model(path), inputs), expected)


def test_temporal_overlap_is_rejected(api):
    _, model = api
    with pytest.raises(ValueError, match="(?i)temporal"):
        model.train_model(labeled(), labeled())


def test_empty_training_is_rejected_but_empty_prediction_is_valid(api):
    _, model = api
    with pytest.raises(ValueError):
        model.train_model(labeled().iloc[:0], labeled(2))
    bundle, _ = model.train_model(labeled(), labeled(2))
    assert model.predict(bundle, labeled().iloc[:0]).shape == (0,)


def test_bounded_large_batch_keeps_every_row_in_order(api):
    _, model = api
    bundle, _ = model.train_model(labeled(), labeled(2))
    frame = pd.DataFrame({"PULocationID": [2, 1] * 5000, "DOLocationID": [3] * 10000})
    actual = model.predict(bundle, frame)
    assert actual.shape == (10000,)
    np.testing.assert_allclose(actual, [4.0, 2.0] * 5000)


def run_cli(tmp_path, *args):
    import taxi_duration

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(taxi_duration.__file__).resolve().parents[1])
    return subprocess.run(
        [sys.executable, "-m", "taxi_duration.cli", *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def test_cli_train_and_predict_in_another_directory(api, tmp_path):
    labeled().to_parquet(tmp_path / "train.parquet")
    labeled(2).to_parquet(tmp_path / "validation.parquet")
    inputs = pd.DataFrame(
        {
            "ride_id": ["ride-b", "ride-a"],
            "PULocationID": [2, 1],
            "DOLocationID": [3, 3],
        }
    )
    inputs.to_csv(tmp_path / "input.csv", index=False)
    result = run_cli(
        tmp_path,
        "train",
        "--train",
        "train.parquet",
        "--validation",
        "validation.parquet",
        "--output",
        "artifacts",
    )
    assert result.returncode == 0, result.stderr
    passport = json.loads((tmp_path / "artifacts/run.json").read_text())
    assert len(passport["inputs"]["train"]["sha256"]) == 64
    assert passport["environment"]["scikit-learn"]
    assert passport["source_sha256"]
    result = run_cli(
        tmp_path,
        "predict",
        "--model",
        "artifacts/model.pkl",
        "--input",
        "input.csv",
        "--output",
        "predictions.csv",
    )
    assert result.returncode == 0, result.stderr
    actual = pd.read_csv(tmp_path / "predictions.csv")
    assert actual.ride_id.tolist() == ["ride-b", "ride-a"]
    np.testing.assert_allclose(actual.predicted_duration, [4.0, 2.0])


def test_cli_rejects_duplicate_ride_ids(api, tmp_path):
    _, model = api
    bundle, _ = model.train_model(labeled(), labeled(2))
    model.save_model(bundle, tmp_path / "model.pkl")
    pd.DataFrame(
        {"ride_id": ["same", "same"], "PULocationID": [1, 2], "DOLocationID": [3, 3]}
    ).to_csv(tmp_path / "input.csv", index=False)
    result = run_cli(
        tmp_path,
        "predict",
        "--model",
        "model.pkl",
        "--input",
        "input.csv",
        "--output",
        "out.csv",
    )
    assert result.returncode != 0
    assert not (tmp_path / "out.csv").exists()


def test_refactoring_preserves_report_and_training_baseline(api):
    _, model = api
    validation = labeled(2)
    validation["lpep_dropoff_datetime"] += pd.Timedelta(minutes=1)
    _, metrics = model.train_model(labeled(), validation)
    assert metrics == pytest.approx(
        {
            "train_rows": 8,
            "validation_rows": 8,
            "train_dropped": 0,
            "validation_dropped": 0,
            "feature_count": 3,
            "baseline_mean_minutes": 3.0,
            "train_rmse": 0.0,
            "validation_rmse": 1.0,
            "baseline_rmse": 1.4142135623730951,
        },
        abs=1e-10,
    )
