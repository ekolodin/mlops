"""Одинаковые шаги ML-пайплайна для CLI и Airflow."""
import argparse
import hashlib
import json
from pathlib import Path

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CATEGORICAL = ["Sex", "Embarked"]


def prepare(data, output):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data, output = Path(data), Path(output)
    manifest = json.loads(data.with_name("manifest.json").read_text())
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("CSV SHA-256 does not match data/manifest.json")
    frame = pd.read_csv(data)
    train, valid = train_test_split(
        frame, test_size=0.2, random_state=42, stratify=frame.Survived,
    )
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train.csv", index=False)
    valid.to_csv(output / "validation.csv", index=False)
    metadata = {"data_sha256": digest, "train_rows": len(train),
                "validation_rows": len(valid), "split_seed": 42}
    (output / "split.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    return str(output)


def features(frame):
    result = frame[FEATURES].copy()
    for name in CATEGORICAL:
        result[name] = result[name].fillna("unknown").astype(str)
    return result


def train(output, depth=4):
    import pandas as pd
    from catboost import CatBoostClassifier

    if not 1 <= depth <= 8:
        raise ValueError("depth must be in [1, 8]")
    output = Path(output)
    frame = pd.read_csv(output / "train.csv")
    model = CatBoostClassifier(
        iterations=60, depth=depth, learning_rate=0.05, random_seed=42,
        cat_features=CATEGORICAL, thread_count=2, verbose=False,
        allow_writing_files=False,
    )
    model.fit(features(frame), frame.Survived)
    model.save_model(str(output / "model.cbm"))
    (output / "params.json").write_text(json.dumps({"depth": depth, "iterations": 60}))
    print(f"Saved {output / 'model.cbm'}", flush=True)
    return str(output)


def evaluate(output, min_auc=0.7):
    import pandas as pd
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    if not 0 <= min_auc <= 1:
        raise ValueError("min_auc must be in [0, 1]")
    output = Path(output)
    frame = pd.read_csv(output / "validation.csv")
    model = CatBoostClassifier()
    model.load_model(str(output / "model.cbm"))
    score = float(roc_auc_score(frame.Survived, model.predict_proba(features(frame))[:, 1]))
    result = {**json.loads((output / "split.json").read_text()),
              **json.loads((output / "params.json").read_text()),
              "validation_roc_auc": score, "min_auc": min_auc, "passed": score >= min_auc,
              "model_sha256": hashlib.sha256((output / "model.cbm").read_bytes()).hexdigest()}
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise ValueError(f"ROC-AUC {score:.6f} < threshold {min_auc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "train", "evaluate", "all", "report"])
    parser.add_argument("--data", type=Path, default=Path("/data/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("/work/manual"))
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--min-auc", type=float, default=0.7)
    args = parser.parse_args()
    if args.step in ("prepare", "all"):
        prepare(args.data, args.output)
    if args.step in ("train", "all"):
        train(args.output, args.depth)
    if args.step in ("evaluate", "all"):
        evaluate(args.output, args.min_auc)
    if args.step == "report":
        print((args.output / "metrics.json").read_text())


if __name__ == "__main__":
    main()
