import pandas as pd
from pandas.testing import assert_frame_equal

from taxi_duration.data import prepare_training_data


def test_prepare_training_data_keeps_only_duration_boundaries():
    pickup = pd.Timestamp("2026-01-01 10:00:00")

    frame = pd.DataFrame(
        {
            "PULocationID": [1, 2, 3, 4],
            "DOLocationID": [10, 20, 30, 40],
            "lpep_pickup_datetime": [pickup] * 4,
            "lpep_dropoff_datetime": [
                pickup + pd.Timedelta(seconds=59),    # < 1 минуты: исключить
                pickup + pd.Timedelta(seconds=60),    # ровно 1 минута: оставить
                pickup + pd.Timedelta(seconds=3600),  # ровно 60 минут: оставить
                pickup + pd.Timedelta(seconds=3601),  # > 60 минут: исключить
            ],
        }
    )

    result = prepare_training_data(frame)

    assert result.index.tolist() == [1, 2]
    assert result["duration"].tolist() == [1.0, 60.0]


def test_prepare_training_data_does_not_mutate_input_dataframe():
    frame = pd.DataFrame(
        {
            "PULocationID": [1, 2],
            "DOLocationID": [10, 20],
            "lpep_pickup_datetime": [
                "2026-01-01 10:00:00",
                "2026-01-01 11:00:00",
            ],
            "lpep_dropoff_datetime": [
                "2026-01-01 10:10:00",
                "2026-01-01 11:20:00",
            ],
        }
    )
    original = frame.copy(deep=True)

    prepare_training_data(frame)

    assert_frame_equal(frame, original)
    assert "duration" not in frame.columns