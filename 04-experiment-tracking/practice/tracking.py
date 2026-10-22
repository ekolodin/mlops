"""Параметры, метрика и модель одного запуска MLflow."""

from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.catboost
from mlflow.models import infer_signature

EXPERIMENT = "titanic-tracking"
DEFAULT_STATE = Path(__file__).resolve().parent / "state"


def configure(state=DEFAULT_STATE):
    """SQLite stores run metadata; the artifacts directory stores model files."""
    state = Path(state).resolve()
    state.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{state / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(state / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return state


def log_parameters(params, data_sha256, train_rows, validation_rows):
    """Сохранить параметры и сведения о данных активного run."""
    # TODO 1: log_params(params), затем train_rows и validation_rows как параметры.
    # Хеш CSV сохраните через set_tag под именем data_sha256.
    raise NotImplementedError("TODO 1: параметры и данные")


def log_results(model, score, example):
    """Сохранить validation ROC-AUC и модель; вернуть model URI."""
    # Описание входа и версии библиотек для сохранённой модели уже подготовлены.
    signature = infer_signature(example, model.predict(example))
    requirements = [f"catboost=={version('catboost')}",
                    f"pandas=={version('pandas')}", f"numpy=={version('numpy')}"]
    # TODO 2: log_metric("validation_roc_auc", score).
    # Вызовите mlflow.catboost.log_model(model, name="model", ...):
    # передайте signature, input_example=example и pip_requirements=requirements.
    # Результат содержит model_uri; верните его.
    raise NotImplementedError("TODO 2: метрика и модель")


def load_model(run_id):
    """Загрузить CatBoost из конкретного запуска MLflow."""
    # TODO 3: составьте URI runs:/<run_id>/model.
    # Верните результат mlflow.catboost.load_model(model_uri).
    raise NotImplementedError("TODO 3: загрузка модели")
