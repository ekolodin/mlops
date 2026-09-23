"""Pure transformations: no files, network, model fitting or global state."""
import json
import numpy as np
import pandas as pd

FEATURES = ("PULocationID", "DOLocationID")
PICKUP = "lpep_pickup_datetime"
DROPOFF = "lpep_dropoff_datetime"


def require_columns(frame, columns):
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not supported")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def prepare_features(frame):
    """Zone categories only. Missing=-1; valid IDs 1..265, or explicit -1.

    Prediction does not read timestamps or duration and never filters rows.
    The caller must know the destination at prediction time.
    """
    require_columns(frame, FEATURES)
    result = pd.DataFrame(index=frame.index)
    for name in FEATURES:
        if frame[name].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{name}: boolean values are not zone IDs")
        values = pd.to_numeric(frame[name], errors="raise").fillna(-1)
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all() or not (array == np.floor(array)).all():
            raise ValueError(f"{name}: expected finite integer zone IDs")
        if not (((array >= 1) & (array <= 265)) | (array == -1)).all():
            raise ValueError(f"{name}: expected 1..265 or -1 for missing")
        result[name] = values.astype("int64").astype(str)
    return result.to_dict(orient="records")


def prepare_training_data(frame):
    """TODO 3: опишите вход, результат, фильтр и ошибки в Google-style."""
    # TODO 3: добавьте аннотации входа и выхода, не меняя реализацию.
    require_columns(frame, (*FEATURES, PICKUP, DROPOFF))
    result = frame.copy()
    for name in (PICKUP, DROPOFF):
        result[name] = pd.to_datetime(result[name], errors="coerce")
    result["duration"] = (result[DROPOFF] - result[PICKUP]).dt.total_seconds() / 60
    result = result.loc[result.duration.between(1, 60, inclusive="both")].copy()
    prepare_features(result)  # Validate before any model can consume these rows.
    return result
