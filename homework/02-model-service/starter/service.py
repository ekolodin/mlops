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
from pydantic import BaseModel, ConfigDict, Field


class HttpCaptionInput(BaseModel):
    """Тело JSON-запроса. После декодирования картинка проходит общую проверку CaptionInput."""

    model_config = ConfigDict(extra="forbid", strict=True)
    # Каждые три байта файла занимают четыре символа base64 (с округлением вверх).
    image_base64: str = Field(min_length=1, max_length=4 * ((MAX_IMAGE_BYTES + 2) // 3))
    max_new_tokens: int | None = Field(default=None, ge=1, le=48)


@asynccontextmanager
async def lifespan(app):
    """Загружает модель один раз при старте HTTP-сервиса."""
    app.state.runner = ModelRunner()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    """Верните готовность сервиса и версию модели, загруженной при старте."""
    return {"ready": True, "model_version": app.state.runner.model_version}


@app.post("/predict")
def predict_http(request: HttpCaptionInput):
    """Примите картинку в base64 и верните результат модели. Ошибка входных данных даёт HTTP 422."""
    try:
        image = base64.b64decode(request.image_base64, validate=True)
        value = CaptionInput(image=image, max_new_tokens=request.max_new_tokens)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return app.state.runner.predict(value, protocol="http")


class Predictor(predict_pb2_grpc.PredictorServicer):
    """Обработчик метода, объявленного в predict.proto. В тестах вместо модели получает её замену."""

    def __init__(self, runner):
        self.runner = runner

    def Predict(self, request, context):
        """Возвращает PredictionResponse; неверный ввод отклоняет со статусом INVALID_ARGUMENT."""
        if not request.HasField("image"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image is required")
        tokens = request.max_new_tokens if request.HasField("max_new_tokens") else None
        try:
            value = CaptionInput(image=request.image, max_new_tokens=tokens)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        result = self.runner.predict(value, protocol="grpc")
        return predict_pb2.PredictionResponse(**result)


def build_grpc_server(runner):
    """Верните сервер с зарегистрированным Predictor. Порт выбирает вызывающий код."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    predict_pb2_grpc.add_PredictorServicer_to_server(Predictor(runner), server)
    return server


def serve_grpc():
    """Загружает модель и принимает запросы до остановки процесса."""
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
