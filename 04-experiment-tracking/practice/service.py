"""Загрузить выбранную модель один раз и обслуживать /predict и /metrics/."""

import argparse
import asyncio
import itertools
import json
import logging
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException
import pandas as pd
from prometheus_client import CollectorRegistry, make_asgi_app
from pydantic import BaseModel, Field
import uvicorn

from metrics import RequestMetrics
import tracking
from train import prepare_features

logger = logging.getLogger("prediction")


class Passenger(BaseModel):
    Pclass: int = Field(ge=1, le=3)
    Sex: str
    Age: float | None = Field(default=None, ge=0)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: str | None = None


def create_app(predict, *, run_id, variant="normal", slow_every=0,
               delay_seconds=1.2, timeout_seconds=1.0):
    if variant not in {"normal", "slow"} or slow_every < 0 or timeout_seconds <= 0 or delay_seconds < 0:
        raise ValueError("Invalid variant, delay or timeout")
    app = FastAPI(title="Titanic prediction")
    registry = CollectorRegistry()
    metrics = RequestMetrics(registry)
    app.mount("/metrics", make_asgi_app(registry=registry))
    sequence = itertools.count(1)

    @app.get("/health")
    def health():
        return {"run_id": run_id, "variant": variant}

    @app.post("/predict")
    async def predict_passenger(passenger: Passenger):
        request_id = uuid4().hex[:12]
        request_number = next(sequence)
        started = time.perf_counter()
        status = "error"

        async def compute():
            # Explicit synthetic incident: every Nth call waits before inference.
            if slow_every and request_number % slow_every == 0:
                await asyncio.sleep(delay_seconds)
            frame = prepare_features(pd.DataFrame([passenger.model_dump()]))
            return await asyncio.to_thread(predict, frame)

        try:
            probability = float(await asyncio.wait_for(compute(), timeout=timeout_seconds))
            status = "ok"
            logger.info("request_id=%s variant=%s status=ok run_id=%s", request_id, variant, run_id)
            return {"request_id": request_id, "survival_probability": probability}
        except TimeoutError:
            logger.error("TimeoutError: model_inference request_id=%s variant=%s timeout=%ss run_id=%s",
                         request_id, variant, timeout_seconds, run_id)
            raise HTTPException(status_code=504, detail={
                "error": "model_inference_timeout", "request_id": request_id,
            }) from None
        except Exception:
            logger.exception("InferenceError request_id=%s variant=%s run_id=%s", request_id, variant, run_id)
            raise HTTPException(status_code=500, detail={
                "error": "model_inference_error", "request_id": request_id,
            }) from None
        finally:
            # Count a completed prediction once, even when it ends in an error.
            metrics.count_request(variant, status)
            metrics.observe_duration(variant, time.perf_counter() - started)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="MLflow run ID; otherwise read state/selected-run.json")
    parser.add_argument("--state", type=Path, default=tracking.DEFAULT_STATE)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--slow-every", type=int, default=0,
                        help="Inject a delay into every Nth prediction; 0 disables it")
    args = parser.parse_args()
    if args.slow_every < 0:
        parser.error("--slow-every must be nonnegative")
    selected = args.state / "selected-run.json"
    if not args.run_id and not selected.exists():
        parser.error("Select a run first: python select_run.py RUN_ID")
    run_id = args.run_id or json.loads(selected.read_text())["run_id"]
    tracking.configure(args.state)
    model = tracking.load_model(run_id)
    variant = "slow" if args.slow_every else "normal"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Model loaded run_id=%s variant=%s slow_every=%s", run_id, variant, args.slow_every)
    app = create_app(lambda frame: model.predict_proba(frame)[0, 1],
                     run_id=run_id, variant=variant, slow_every=args.slow_every)
    # One worker: each process has its own model and metric registry.
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
