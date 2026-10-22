"""Проверить env и общий датасет до заполнения TODO."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import pandas as pd


def main():
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Expected Python 3.13")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "data/manifest.json").read_text())
    data = root / "data" / manifest["file"]
    if hashlib.sha256(data.read_bytes()).hexdigest() != manifest["sha256"]:
        raise SystemExit("data/train.csv differs from manifest.json")
    frame = pd.read_csv(data)
    if len(frame) != 713 or set(frame.Survived.unique()) != {0, 1}:
        raise SystemExit("Expected 713 labeled Titanic rows")
    for package in ("catboost", "mlflow", "tensorboard", "prometheus-client", "fastapi"):
        print(f"{package}: {version(package)}")
    print("Python 3.13; данные: 713 строк; SHA-256 совпадает")


if __name__ == "__main__":
    main()
