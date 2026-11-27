"""Обучить CatBoost, записать запуск MLflow и кривые TensorBoard."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from catboost import CatBoostClassifier
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import tracking

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CATEGORICAL = ["Sex", "Embarked"]
DEFAULT_DATA = Path(__file__).resolve().parent / "data/train.csv"


def prepare_features(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    features = frame[FEATURES].copy()
    for name in CATEGORICAL:
        features[name] = features[name].fillna("unknown").astype(str)
    return features


def fit_model(x_train, y_train, x_valid, y_valid, *, depth, iterations, train_dir):
    train_dir = Path(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    model = CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC", cat_features=CATEGORICAL,
        random_seed=42, thread_count=2, train_dir=str(train_dir), verbose=False,
    )
    # CatBoost writes TensorBoard-compatible events into train_dir/learn and /test.
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
    return model


def tensorboard_run_directory(root, run_name):
    """Readable run name; repeated names get a suffix without overwriting events."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w.-]+", "-", run_name).strip(".-") or "run"
    number = 1
    while True:
        directory = root / (name if number == 1 else f"{name}-{number}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            number += 1


def train_experiment(data=DEFAULT_DATA, *, state=tracking.DEFAULT_STATE,
                     depth=4, iterations=200, run_name=None):
    if not 1 <= depth <= 8 or not 1 <= iterations <= 300:
        raise ValueError("Use depth in [1, 8] and iterations in [1, 300] for this lab")
    data = Path(data)
    frame = pd.read_csv(data)
    if "Survived" not in frame or frame.Survived.isna().any() or set(frame.Survived.unique()) != {0, 1}:
        raise ValueError("Survived must contain both classes, 0 and 1, without missing values")
    features = prepare_features(frame)
    x_train, x_valid, y_train, y_valid = train_test_split(
        features, frame.Survived, test_size=0.2, random_state=42, stratify=frame.Survived,
    )
    state = tracking.configure(state)
    name = run_name or f"depth-{depth}"
    with mlflow.start_run(run_name=name) as run:
        params = {"depth": depth, "iterations": iterations, "learning_rate": 0.05,
                  "split_seed": 42, "model_seed": 42, "validation_fraction": 0.2}
        tracking.log_parameters(params, hashlib.sha256(data.read_bytes()).hexdigest(),
                                len(x_train), len(x_valid))
        tensorboard_dir = tensorboard_run_directory(state / "tensorboard", name)
        model = fit_model(x_train, y_train, x_valid, y_valid, depth=depth,
                          iterations=iterations, train_dir=tensorboard_dir)
        score = float(roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1]))
        # Numerical floats avoid a nullable-integer signature for MLflow's example.
        example = x_train.head(3).copy()
        for name in set(FEATURES) - set(CATEGORICAL):
            example[name] = example[name].astype(float)
        model_uri = tracking.log_results(model, score, example)
        result = {"run_id": run.info.run_id, "validation_roc_auc": score,
                  "model_uri": model_uri, "tensorboard_dir": str(tensorboard_dir),
                  "train_rows": len(x_train), "validation_rows": len(x_valid)}
    # A convenience file for the CLI; all runs remain in the MLflow database.
    (state / "last-run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--state", type=Path, default=tracking.DEFAULT_STATE)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--name")
    args = parser.parse_args()
    result = train_experiment(args.data, state=args.state, depth=args.depth,
                              iterations=args.iterations, run_name=args.name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
