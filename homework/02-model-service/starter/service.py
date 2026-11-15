"""Два интерфейса одной модели. Запуск: python service.py http или grpc."""

import argparse
from contextlib import asynccontextmanager

import grpc
import predict_pb2_grpc
import uvicorn
from fastapi import FastAPI
from model import MAX_IMAGE_BYTES, ModelRunner
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
    raise NotImplementedError("Верните готовность и версию загруженной модели")


@app.post("/predict")
def predict_http(request: HttpCaptionInput):
    """Примите картинку в base64 и верните результат модели. Ошибка входных данных даёт HTTP 422."""
    raise NotImplementedError("Преобразуйте HTTP-запрос, вызовите модель и верните ответ")


class Predictor(predict_pb2_grpc.PredictorServicer):
    """Обработчик метода, объявленного в predict.proto. В тестах вместо модели получает её замену."""

    def __init__(self, runner):
        self.runner = runner

    def Predict(self, request, context):
        """Возвращает PredictionResponse; неверный ввод отклоняет со статусом INVALID_ARGUMENT."""
        # Отсутствующий лимит и переданный 0 имеют разный смысл для модели.
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Реализуйте Predict")


def build_grpc_server(runner):
    """Верните сервер с зарегистрированным Predictor. Порт выбирает вызывающий код."""
    raise NotImplementedError("Создайте gRPC-сервер и зарегистрируйте Predictor")


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
