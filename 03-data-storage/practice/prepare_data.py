"""Create labeled train/test splits from Titanic's original training CSV."""

import argparse
import hashlib
import io
from pathlib import Path
import tarfile
from urllib.request import urlopen

import pandas as pd

URL = "https://storage.mds.yandex.net/get-devtools-opensource/233854/titanic.tar.gz"
ARCHIVE_MD5 = "9c8bc61d545c6af244a1d37494df3fc3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Existing labeled train.csv")
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    train_path = args.output / "train-split.csv"
    test_path = args.output / "test-split.csv"
    if train_path.exists() or test_path.exists():
        parser.error("Splits already exist; choose a new --output directory.")

    source = args.input or args.output / "raw/train.csv"
    if args.input is None and not source.exists():
        with urlopen(URL, timeout=60) as response:
            archive_bytes = response.read()
        if hashlib.md5(archive_bytes).hexdigest() != ARCHIVE_MD5:
            raise ValueError("Titanic archive checksum mismatch")
        # Read only the CSV member; do not extract arbitrary archive paths.
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            member = archive.extractfile("train.csv")
            if member is None:
                raise ValueError("train.csv is absent from the archive")
            raw = member.read()
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(raw)

    frame = pd.read_csv(source)
    if "Survived" not in frame or set(frame.Survived.dropna().unique()) != {0, 1}:
        parser.error("Expected a labeled CSV with both Survived classes (0 and 1).")
    if frame.Survived.isna().any():
        parser.error("Survived must not contain missing values.")
    if "PassengerId" not in frame or frame.PassengerId.isna().any() or frame.PassengerId.duplicated().any():
        parser.error("PassengerId must be present, nonempty and unique.")
    # Shuffle whole CSV records, not text lines: names contain quoted commas.
    shuffled = frame.sample(frac=1, random_state=42)
    test_rows = len(frame) // 5
    if test_rows < 2:
        parser.error("Too few rows for the train/test split.")
    test = shuffled.iloc[:test_rows]
    train = shuffled.iloc[test_rows:]
    if any(set(part.Survived.unique()) != {0, 1} for part in (train, test)):
        parser.error("Both splits must contain both Survived classes for ROC-AUC.")
    args.output.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"train: {len(train)} objects; test: {len(test)} objects")
    print(f"source SHA-256: {hashlib.sha256(source.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
