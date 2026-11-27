"""Train a small CatBoost classifier and save its native checkpoint."""

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CATEGORICAL = ["Sex", "Embarked"]


def load_labeled(path):
    frame = pd.read_csv(path)
    missing = set(FEATURES + ["Survived"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.Survived.isna().any() or set(frame.Survived.unique()) != {0, 1}:
        raise ValueError("Survived must contain both classes, 0 and 1, without missing values")
    # Explicit selection excludes the target and the passenger identifier.
    features = frame[FEATURES].copy()
    for name in CATEGORICAL:
        features[name] = features[name].fillna("unknown").astype(str)
    # CatBoost handles missing numerical values itself.
    return features, frame["Survived"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    args = parser.parse_args()
    features, target = load_labeled(args.train)
    # This validation set comes only from train; the external test stays unseen.
    x_train, x_valid, y_train, y_valid = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target,
    )
    model = CatBoostClassifier(
        iterations=200, depth=4, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC", cat_features=CATEGORICAL,
        random_seed=42, thread_count=2, allow_writing_files=False, verbose=False,
    )
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
    score = roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1])
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.save_path))
    print(json.dumps({
        "train_rows": len(x_train), "validation_rows": len(x_valid),
        "validation_roc_auc": float(score), "model": str(args.save_path),
    }, indent=2))


if __name__ == "__main__":
    main()
