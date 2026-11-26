"""Два интерфейса одной модели. Запуск: python service.py http или grpc."""

import argparse
import base64
from concurrent import futures
from contextlib import asynccontextmanager

import grpc
import predict_pb2
import predict_pb2_grpc
import uvicorn
from fastapi import FastAPI, HTTPException
from model import MAX_IMAGE_BYTES, CaptionInput, ModelRunner
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class HttpCaptionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_base64: str = Field(min_length=1, max_length=4 * ((MAX_IMAGE_BYTES + 2) // 3))
    max_new_tokens: int | None = Field(default=None, ge=1, le=48)


@asynccontextmanager
async def lifespan(app):
    app.state.runner = ModelRunner()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    runner = app.state.runner
    return {"ready": True, "model_version": runner.model_version}


@app.post("/predict")
def predict_http(request: HttpCaptionInput):
    try:
        image = base64.b64decode(request.image_base64, validate=True)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"Invalid base64: {error}") from error
    try:
        value = CaptionInput(image=image, max_new_tokens=request.max_new_tokens)
    except ValidationError as error:
        # errors() содержит bytes (input) и ValueError (ctx) — не сериализуются в JSON.
        # str(error) даёт безопасную читаемую строку, а статуса 422 достаточно для контракта.
        raise HTTPException(status_code=422, detail=str(error)) from error
    return app.state.runner.predict(value, protocol="http")


class Predictor(predict_pb2_grpc.PredictorServicer):
    def __init__(self, runner):
        self.runner = runner

    def Predict(self, request, context):
        if not request.HasField("image") or not request.image:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image is required")
        max_tokens = request.max_new_tokens if request.HasField("max_new_tokens") else None
        try:
            value = CaptionInput(image=request.image, max_new_tokens=max_tokens)
        except ValidationError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        result = self.runner.predict(value, protocol="grpc")
        return predict_pb2.PredictionResponse(**result)


def build_grpc_server(runner):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    predict_pb2_grpc.add_PredictorServicer_to_server(Predictor(runner), server)
    return server


def serve_grpc():
    runner = ModelRunner()
    server = build_grpc_server(runner)
    if not server.add_insecure_port("0.0.0.0:50051"):
        raise RuntimeError("Cannot bind gRPC port 50051")
    server.start()
    print(f"gRPC ready model_version={runner.model_version}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", choices=["http", "grpc"])
    if parser.parse_args().protocol == "http":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        serve_grpc()