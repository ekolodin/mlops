"""Проверки TODO 4–5: измеряем реальные результаты вызовов /predict."""

import logging

from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
import pytest

from metrics import RequestMetrics
from service import create_app


def test_counter_separates_success_and_error():
    registry = CollectorRegistry()
    metrics = RequestMetrics(registry)
    metrics.count_request("normal", "ok")
    metrics.count_request("normal", "ok")
    metrics.count_request("slow", "error")
    assert registry.get_sample_value("prediction_requests_total",
        {"variant": "normal", "status": "ok"}) == 2
    assert registry.get_sample_value("prediction_requests_total",
        {"variant": "slow", "status": "error"}) == 1


def test_histogram_records_seconds_and_count():
    registry = CollectorRegistry()
    metrics = RequestMetrics(registry)
    metrics.observe_duration("normal", 0.1)
    metrics.observe_duration("normal", 0.3)
    assert registry.get_sample_value("prediction_duration_seconds_count",
        {"variant": "normal"}) == 2
    assert registry.get_sample_value("prediction_duration_seconds_sum",
        {"variant": "normal"}) == pytest.approx(0.4)


@pytest.fixture
def passenger():
    return {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
            "Parch": 0, "Fare": 7.25, "Embarked": "S"}


def test_prediction_counts_request_and_does_not_return_offline_quality(passenger):
    # A small deterministic predictor keeps this HTTP test independent of training.
    app = create_app(lambda frame: 0.25, run_id="test-run", timeout_seconds=10)
    with TestClient(app) as client:
        response = client.post("/predict", json=passenger)
        assert response.status_code == 200
        assert response.json()["survival_probability"] == 0.25
        assert "roc_auc" not in response.json()
        text = client.get("/metrics/").text
        assert 'prediction_requests_total{status="ok",variant="normal"} 1.0' in text
        assert 'prediction_duration_seconds_count{variant="normal"} 1.0' in text
        assert "request_id" not in text


def test_timeout_is_error_and_latency_is_recorded(passenger, caplog):
    app = create_app(lambda frame: 0.25, run_id="test-run", variant="slow",
                     slow_every=1, delay_seconds=0.06, timeout_seconds=0.01)
    with caplog.at_level(logging.ERROR, logger="prediction"):
        with TestClient(app) as client:
            response = client.post("/predict", json=passenger)
            assert response.status_code == 504
            assert response.json()["detail"]["error"] == "model_inference_timeout"
            text = client.get("/metrics/").text
            assert 'prediction_requests_total{status="error",variant="slow"} 1.0' in text
            assert 'prediction_duration_seconds_count{variant="slow"} 1.0' in text
    assert "TimeoutError" in caplog.text
    assert response.json()["detail"]["request_id"] in caplog.text


def test_invalid_body_does_not_run_model(passenger):
    app = create_app(lambda frame: pytest.fail("invalid input reached prediction"),
                     run_id="test-run")
    passenger["Pclass"] = 9
    with TestClient(app) as client:
        assert client.post("/predict", json=passenger).status_code == 422
        assert 'prediction_requests_total{status="ok",variant="normal"} 0.0' in client.get("/metrics/").text


def test_normal_profile_has_no_injected_timeouts(passenger):
    # This checks the absence of injected delays, not OS thread scheduling speed.
    app = create_app(lambda frame: 0.25, run_id="test-run", timeout_seconds=10)
    with TestClient(app) as client:
        assert all(client.post("/predict", json=passenger).status_code == 200
                   for _ in range(4))
        text = client.get("/metrics/").text
        assert 'prediction_requests_total{status="ok",variant="normal"} 4.0' in text
