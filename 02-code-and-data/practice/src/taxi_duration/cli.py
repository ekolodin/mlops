"""File and process boundary; pure transformations live in data.py/model.py."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import pandas as pd

from . import __version__
from .model import load_model, predict, save_model, train_model


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_frame(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"ride_id": "str"})
    raise ValueError("Expected .csv or .parquet input")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def train_command(args):
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory; choose another run name")
    bundle, metrics = train_model(read_frame(args.train), read_frame(args.validation))
    inputs = {name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
              for name, path in (("train", args.train), ("validation", args.validation))}
    output.mkdir(parents=True, exist_ok=True)
    save_model(bundle, output / "model.pkl")
    write_json(output / "metrics.json", metrics)
    write_json(output / "run.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "inputs": inputs, "package_version": __version__,
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "environment": {**{name: version(name) for name in
                        ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow")},
                        "python": platform.python_version(), "platform": platform.platform()},
        "configuration": {"features": bundle["features"], "duration_minutes": [1, 60],
                          "estimator": "LinearRegression", "random_seed": None,
                          "seed_note": "No randomized split or stochastic estimator"},
        "artifacts": {"model.pkl": sha256(output / "model.pkl"),
                      "metrics.json": sha256(output / "metrics.json")},
    })
    print(json.dumps(metrics, indent=2))


def predict_command(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Prediction output exists; choose a new path")
    frame = read_frame(args.input)
    if "ride_id" in frame:
        ids = frame.ride_id.astype("string")
        if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
            raise ValueError("ride_id must be nonempty and unique within the file")
    else:
        ids = pd.Series([f"row-{i}" for i in range(len(frame))], index=frame.index)
    predictions = predict(load_model(args.model), frame)
    result = pd.DataFrame({"ride_id": ids.to_numpy(), "predicted_duration": predictions})
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, mode="x")
    print(f"Predicted {len(result)} rows -> {output}")


def main():
    parser = argparse.ArgumentParser(description="Local taxi-duration training and inference")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--train", required=True)
    train.add_argument("--validation", required=True)
    train.add_argument("--output", required=True)
    train.set_defaults(func=train_command)
    inference = commands.add_parser("predict", help="Load only models you trained or otherwise trust")
    inference.add_argument("--model", required=True)
    inference.add_argument("--input", required=True)
    inference.add_argument("--output", required=True)
    inference.set_defaults(func=predict_command)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
