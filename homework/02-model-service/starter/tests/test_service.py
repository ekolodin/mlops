"""Проверки HTTP и gRPC без весов: запрос доходит до модели с теми же байтами и лимитом."""

import base64
import hashlib

import grpc
import pytest
from fastapi.testclient import TestClient
from model import DEFAULT_ASSETS


class RecordingRunner:
    """Запоминает вход обработчика и возвращает фиксированную подпись вместо запуска модели."""

    model_version = "contract-test"

    def predict(self, request, *, protocol):
        self.last_request = request
        self.last_protocol = protocol
        return {
            "caption": "A test caption.",
            "model_version": self.model_version,
            "request_id": "test-request-id",
            "image_sha256": hashlib.sha256(request.image).hexdigest(),
        }


@pytest.fixture
def rpc():
    """Поднимает локальный сервер на свободном порту и отдаёт тесту клиент и замену модели."""
    import predict_pb2_grpc
    from service import build_grpc_server

    runner = RecordingRunner()
    server = build_grpc_server(runner)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=5)
    yield predict_pb2_grpc.PredictorStub(channel), runner
    channel.close()
    server.stop(0).wait()


def test_proto_contract():
    """Проверяет схему сообщений до запуска сервера: номера, типы и присутствие полей запроса."""
    import predict_pb2
    from google.protobuf.descriptor import FieldDescriptor as F

    # DESCRIPTOR описывает поля сгенерированного класса, даже если сообщение пустое.
    fields = predict_pb2.CaptionRequest.DESCRIPTOR.fields_by_name
    assert set(fields) == {"image", "max_new_tokens"}
    for name, number, field_type in [
        ("image", 1, F.TYPE_BYTES),
        ("max_new_tokens", 2, F.TYPE_INT32),
    ]:
        field = fields[name]
        assert field.number == number
        assert field.type == field_type
        assert field.has_presence

    expected_numbers = {"caption": 1, "model_version": 2, "request_id": 3, "image_sha256": 4}
    fields = predict_pb2.PredictionResponse.DESCRIPTOR.fields_by_name
    assert set(fields) == set(expected_numbers)
    for name, number in expected_numbers.items():
        assert fields[name].number == number
        assert fields[name].type == F.TYPE_STRING


@pytest.mark.parametrize("tokens", [None, 1, 24, 48])
def test_valid_rpc_preserves_input(rpc, tokens):
    import predict_pb2

    stub, runner = rpc
    image = (DEFAULT_ASSETS / "images/cats.jpg").read_bytes()
    values = {"image": image}
    if tokens is not None:
        values["max_new_tokens"] = tokens
    reply = stub.Predict(predict_pb2.CaptionRequest(**values), timeout=5)
    assert runner.last_request.image == image
    assert runner.last_request.max_new_tokens == tokens
    assert runner.last_protocol == "grpc"
    assert reply.caption == "A test caption."
    assert reply.model_version == runner.model_version
    assert reply.request_id == "test-request-id"
    assert reply.image_sha256 == hashlib.sha256(image).hexdigest()


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"image": b""},
        {"image": b"broken"},
        {"max_new_tokens": 0},
        {"max_new_tokens": -1},
        {"max_new_tokens": 49},
    ],
)
def test_invalid_rpc(rpc, values):
    import predict_pb2

    stub, _ = rpc
    if "max_new_tokens" in values:
        values = {"image": (DEFAULT_ASSETS / "images/cats.jpg").read_bytes(), **values}
    with pytest.raises(grpc.RpcError) as caught:
        stub.Predict(predict_pb2.CaptionRequest(**values), timeout=5)
    assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.fixture
def http(monkeypatch):
    """Запускает HTTP-приложение с той же заменой модели, что и в тестах gRPC."""
    import service

    runner = RecordingRunner()
    monkeypatch.setattr(service, "ModelRunner", lambda: runner)
    with TestClient(service.app) as client:
        yield client, runner


def test_http_health(http):
    client, runner = http
    assert client.get("/health").json() == {"ready": True, "model_version": runner.model_version}


@pytest.mark.parametrize("tokens", [None, 1, 48])
def test_http_prediction(http, tokens):
    client, runner = http
    image = (DEFAULT_ASSETS / "images/cats.jpg").read_bytes()
    body = {"image_base64": base64.b64encode(image).decode()}
    if tokens is not None:
        body["max_new_tokens"] = tokens
    reply = client.post("/predict", json=body)
    assert reply.status_code == 200
    assert runner.last_protocol == "http"
    assert runner.last_request.image == image and runner.last_request.max_new_tokens == tokens
    assert reply.json() == {
        "caption": "A test caption.",
        "model_version": runner.model_version,
        "request_id": "test-request-id",
        "image_sha256": hashlib.sha256(image).hexdigest(),
    }


@pytest.mark.parametrize("text", ["%%%", base64.b64encode(b"not an image").decode("ascii")])
def test_http_invalid_image(http, text):
    client, _ = http
    assert client.post("/predict", json={"image_base64": text}).status_code == 422
