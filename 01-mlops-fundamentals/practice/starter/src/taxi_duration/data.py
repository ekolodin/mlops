"""Pure transformations: no files, network, model fitting or global state."""
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
    """Labeled historical cohort only: finite duration in [1, 60] minutes.

    Invalid timestamps are excluded. This is NOT an online serving filter:
    at prediction time the actual duration is unknown.
    """
    require_columns(frame, (*FEATURES, PICKUP, DROPOFF))

    # =========================================================================================

    # TODO 1: подготовьте result, не изменяя исходный frame.
    # Создайте копию; приведите PICKUP и DROPOFF к datetime через
    # pd.to_datetime(..., errors="coerce"): некорректные значения станут NaT.
    # Вычислите duration в минутах и оставьте [1, 60] включительно.
    # Строки с NaT не должны попасть в результат.

    result = frame.copy()

    result[PICKUP] = pd.to_datetime(result[PICKUP], errors="coerce")
    result[DROPOFF] = pd.to_datetime(result[DROPOFF], errors="coerce")

    result["duration"] = (
        result[DROPOFF] - result[PICKUP]
    ).dt.total_seconds() / 60

    result = result.loc[
        result["duration"].between(1, 60, inclusive="both")
    ].copy()

    # =========================================================================================

    return result
