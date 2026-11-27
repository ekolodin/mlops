"""Checks for the CSV split and the train/infer commands, without network."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def cli(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        capture_output=True, text=True,
    )


@pytest.fixture
def passengers(tmp_path):
    # A learnable signal plus missing numerical and categorical values.
    n = 100
    frame = pd.DataFrame({
        "PassengerId": range(1, n + 1), "Survived": [0, 1] * (n // 2),
        "Pclass": [3, 1] * (n // 2), "Sex": ["male", "female"] * (n // 2),
        "Age": [25.0, np.nan] * (n // 2), "SibSp": 0, "Parch": 0,
        "Fare": [7.0, 50.0] * (n // 2), "Embarked": ["S", None] * (n // 2),
        "Name": [f"Passenger, {i}" for i in range(n)],
    })
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    return source


def test_split_is_disjoint_complete_and_reproducible(passengers, tmp_path):
    original = passengers.read_bytes()
    for folder in ("first", "second"):
        result = cli("prepare_data.py", "--input", passengers,
                     "--output", tmp_path / folder)
        assert result.returncode == 0, result.stderr
    train = pd.read_csv(tmp_path / "first/train-split.csv")
    test = pd.read_csv(tmp_path / "first/test-split.csv")
    assert (len(train), len(test)) == (80, 20)
    assert set(train.PassengerId).isdisjoint(test.PassengerId)
    assert set(train.PassengerId) | set(test.PassengerId) == set(range(1, 101))
    assert passengers.read_bytes() == original
    for name in ("train-split.csv", "test-split.csv"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert train.Name.str.contains(",").all()  # CSV quoting survives the split.


def test_prepare_refuses_to_overwrite_existing_split(passengers, tmp_path):
    target = tmp_path / "data"
    target.mkdir()
    (target / "test-split.csv").write_text("keep me\n")
    result = cli("prepare_data.py", "--input", passengers, "--output", target)
    assert result.returncode != 0
    assert (target / "test-split.csv").read_text() == "keep me\n"
    assert not (target / "train-split.csv").exists()


def test_prepare_rejects_unlabeled_competition_test(passengers, tmp_path):
    unlabeled = tmp_path / "unlabeled.csv"
    pd.read_csv(passengers).drop(columns="Survived").to_csv(unlabeled, index=False)
    result = cli("prepare_data.py", "--input", unlabeled, "--output", tmp_path / "out")
    assert result.returncode != 0
    assert "Survived" in result.stderr


def test_train_and_infer_use_saved_model_without_target_leakage(passengers, tmp_path):
    model_path = tmp_path / "artifacts/model.cbm"
    result = cli("train.py", "--train", passengers, "--save-path", model_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["train_rows"] == 80 and report["validation_rows"] == 20
    assert report["validation_roc_auc"] > 0.9
    model_bytes = model_path.read_bytes()
    result = cli("infer.py", "--model-path", model_path, "--test", passengers)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    model = CatBoostClassifier()
    model.load_model(model_path)
    assert "Survived" not in model.feature_names_
    assert "PassengerId" not in model.feature_names_
    frame = pd.read_csv(passengers)
    features = frame[model.feature_names_].copy()
    for name in ("Sex", "Embarked"):
        features[name] = features[name].fillna("unknown").astype(str)
    expected = roc_auc_score(frame.Survived, model.predict_proba(features)[:, 1])
    assert actual["test_roc_auc"] == pytest.approx(expected)
    assert model_path.read_bytes() == model_bytes


def test_infer_rejects_one_class_target(passengers, tmp_path):
    model = tmp_path / "model.cbm"
    trained = cli("train.py", "--train", passengers, "--save-path", model)
    assert trained.returncode == 0, trained.stderr
    frame = pd.read_csv(passengers)
    frame["Survived"] = 1
    test = tmp_path / "single-class.csv"
    frame.to_csv(test, index=False)
    result = cli("infer.py", "--model-path", model, "--test", test)
    assert result.returncode != 0
    assert "Survived" in result.stderr
