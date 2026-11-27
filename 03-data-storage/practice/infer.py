"""Evaluate an existing checkpoint on the fixed labeled test set."""

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from train import load_labeled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    args = parser.parse_args()
    features, target = load_labeled(args.test)
    model = CatBoostClassifier()
    model.load_model(str(args.model_path))
    # ROC-AUC needs probabilities, not the thresholded class predictions.
    probabilities = model.predict_proba(features)[:, 1]
    score = roc_auc_score(target, probabilities)
    print(json.dumps({"test_rows": len(target), "test_roc_auc": float(score)}, indent=2))


if __name__ == "__main__":
    main()
