"""Метрики вычисления предсказаний, без идентификаторов отдельных запросов."""

from prometheus_client import Counter, Histogram


class RequestMetrics:
    def __init__(self, registry):
        self.requests = Counter(
            "prediction_requests_total", "Completed prediction calls, including failures",
            ["variant", "status"], registry=registry,
        )
        self.duration = Histogram(
            "prediction_duration_seconds", "Prediction handling time until response or timeout",
            ["variant"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5),
            registry=registry,
        )
        for variant in ("normal", "slow"):
            for status in ("ok", "error"):
                self.requests.labels(variant=variant, status=status)
            self.duration.labels(variant=variant)

    def count_request(self, variant, status):
        """Прибавить один завершённый вызов к выбранному счётчику."""
        # TODO 4: через self.requests.labels выберите variant и status.
        # У выбранного счётчика вызовите inc().
        raise NotImplementedError("TODO 4: счётчик запросов")

    def observe_duration(self, variant, seconds):
        """Записать одно измерение длительности в секундах."""
        # TODO 5: выберите self.duration.labels(variant=variant).
        # Передайте seconds в observe().
        raise NotImplementedError("TODO 5: длительность запроса")
