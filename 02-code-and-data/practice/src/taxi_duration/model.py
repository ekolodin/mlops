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
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(prepare_features(train))
    x_validation = vectorizer.transform(prepare_features(validation))
    model = LinearRegression()
    model.fit(x_train, train.duration.to_numpy())
    debug_rows = len(validation)
    bundle = {"format_version": 1, "features": list(FEATURES), "vectorizer": vectorizer,
              "model": model, "sklearn_version": sklearn.__version__}
    # TODO 2: используйте regression_metrics для всех трёх RMSE.
    # Имена полей отчёта, выборки и формулу baseline сохраните.
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
    """TODO 3: опишите контракт запросов и предсказаний в Google-style."""
    # TODO 3: добавьте аннотации; тело функции уже работает.
    records = prepare_features(frame)
    if not records:
        return np.empty(0, dtype=float)
    features = bundle["vectorizer"].transform(records)
    result = np.asarray(bundle["model"].predict(features), dtype=float)
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
