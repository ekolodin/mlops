"""Подготовка -> обучение -> оценка. ML-код лежит в pipeline.py."""
from datetime import datetime, timezone
import hashlib

from airflow.sdk import dag, task, get_current_context


@dag(
    dag_id="titanic_training", schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc), catchup=False,
    max_active_runs=1, params={"depth": 4, "min_auc": 0.7},
    tags=["course"],
)
def titanic_training():
    @task
    def prepare():
        import pipeline
        # Каждый DAG run получает собственный каталог, включая ручные запуски.
        run_id = get_current_context()["run_id"]
        key = hashlib.sha256(run_id.encode()).hexdigest()[:16]
        return pipeline.prepare("/data/train.csv", f"/work/{key}")

    @task
    def train(run_dir):
        import pipeline
        depth = int(get_current_context()["params"]["depth"])
        return pipeline.train(run_dir, depth=depth)

    @task
    def evaluate(run_dir):
        import pipeline
        threshold = float(get_current_context()["params"]["min_auc"])
        return pipeline.evaluate(run_dir, min_auc=threshold)

    # TODO 3: соедините три задачи, передавая результат одной следующей.
    # Сейчас DAG содержит только подготовку. Вызовы декораторов строят граф.
    prepare()


titanic_training()
