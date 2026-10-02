"""Fit, evaluate and persist the same feature mapping used during inference."""
from pathlib import Path
import pickle

import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from .data import FEATURES, PICKUP, prepare_features, prepare_training_data


def train_model(train_df, validation_df):
    train = prepare_training_data(train_df)
    validation = prepare_training_data(validation_df)
    if train.empty or validation.empty:
        raise ValueError("Training and validation must each contain eligible rows")
    if train[PICKUP].max() >= validation[PICKUP].min():
        raise ValueError("Temporal split: validation must be strictly later than training")

    # =========================================================================================
    
    # TODO 2: создайте vectorizer = DictVectorizer с sparse=True.
    # Через prepare_features получите записи для train и validation.
    # Определите x_train и x_validation: fit_transform только на train,
    # transform — на validation. Создайте model = LinearRegression
    # и обучите её на x_train и train.duration.to_numpy().

    vectorizer = DictVectorizer(sparse=True)

    train_records = prepare_features(train)
    validation_records = prepare_features(validation)

    x_train = vectorizer.fit_transform(train_records)
    x_validation = vectorizer.transform(validation_records)

    y_train = train.duration.to_numpy()

    model = LinearRegression()
    model.fit(x_train, y_train)

    # =========================================================================================

    bundle = {"format_version": 1, "features": list(FEATURES), "vectorizer": vectorizer,
              "model": model, "sklearn_version": sklearn.__version__}
    metrics = {
        "train_rows": len(train), "validation_rows": len(validation),
        "train_dropped": len(train_df) - len(train),
        "validation_dropped": len(validation_df) - len(validation),
        "feature_count": x_train.shape[1],
        "baseline_mean_minutes": float(train.duration.mean()),
        "baseline_rmse": float(root_mean_squared_error(
            validation.duration, np.full(len(validation), train.duration.mean()))),
        "train_rmse": float(root_mean_squared_error(train.duration, model.predict(x_train))),
        "validation_rmse": float(root_mean_squared_error(validation.duration, model.predict(x_validation))),
    }
    return bundle, metrics


def predict(bundle, frame):
    # =========================================================================================

    # TODO 3: получите records через prepare_features(frame).
    # Для пустого списка верните пустой NumPy-массив с dtype=float.
    # Получите features через transform обученного bundle["vectorizer"],
    # затем result через predict модели из bundle и np.asarray(..., dtype=float).
    # Не вызывайте fit, не требуйте target, сохраняйте число и порядок строк.

    records = prepare_features(frame)

    if not records:
        return np.array([], dtype=float)

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    features = vectorizer.transform(records)
    result = np.asarray(model.predict(features), dtype=float)

    # =========================================================================================
    if result.shape != (len(frame),) or not np.isfinite(result).all():
        raise ValueError("Model returned invalid predictions")
    return result


def save_model(bundle, path):
    """Write a locally trained model without overwriting an existing file."""
    with Path(path).open("xb") as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path):
    """TRUSTED local files ONLY: pickle can execute arbitrary code on load.

    The checks below detect accidental incompatibility, not malicious pickle.
    """
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    if not isinstance(bundle, dict) or bundle.get("format_version") != 1:
        raise ValueError("Unsupported model bundle")
    if bundle.get("features") != list(FEATURES):
        raise ValueError("Incompatible feature schema")
    if bundle.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Use the same scikit-learn version as training")
    return bundle
