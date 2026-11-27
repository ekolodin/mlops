"""Выбрать run для приложения и проверить восстановление модели."""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

import tracking
from train import DEFAULT_DATA, prepare_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Run ID from MLflow, not experiment ID or model ID")
    parser.add_argument("--state", type=Path, default=tracking.DEFAULT_STATE)
    args = parser.parse_args()
    state = tracking.configure(args.state)
    run = mlflow.get_run(args.run_id)
    if run.info.status != "FINISHED":
        parser.error("Select a FINISHED run")
    model = tracking.load_model(args.run_id)
    features = prepare_features(pd.read_csv(DEFAULT_DATA).head(3))
    result = {"run_id": args.run_id,
              "validation_roc_auc": run.data.metrics["validation_roc_auc"],
              "survival_probabilities": model.predict_proba(features)[:, 1].tolist()}
    (state / "selected-run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
