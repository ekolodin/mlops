"""Клиент приёмки: сравнивает два запущенных сервиса с локальным вызовом той же модели."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import grpc
import httpx
import predict_pb2
import predict_pb2_grpc
from model import DEFAULT_ASSETS, CaptionInput, ModelRunner


def compare_case(case, runner, client, stub, timeout):
    """Сравнивает подпись, версию и хеш одного изображения; возвращает оба сетевых ответа."""
    image = (DEFAULT_ASSETS / case["image"]).read_bytes()
    assert hashlib.sha256(image).hexdigest() == case["image_sha256"]
    tokens = case.get("max_new_tokens")
    local = runner.predict(CaptionInput(image=image, max_new_tokens=tokens))
    body = {"image_base64": base64.b64encode(image).decode()}
    fields = {"image": image}
    if tokens is not None:
        # Пропущенный параметр не посылаем: его значение по умолчанию выбирает сервис.
        body["max_new_tokens"] = tokens
        fields["max_new_tokens"] = tokens
    response = client.post("/predict", json=body)
    response.raise_for_status()
    web = response.json()
    rpc = stub.Predict(predict_pb2.CaptionRequest(**fields), timeout=timeout)
    assert local["caption"] == web["caption"] == rpc.caption, case["case_id"]
    assert local["model_version"] == web["model_version"] == rpc.model_version
    assert local["image_sha256"] == web["image_sha256"] == rpc.image_sha256
    assert web["request_id"] and rpc.request_id
    print(f"{case['case_id']}: {local['caption']}", flush=True)
    return [
        {"protocol": "http", **web},
        {
            "protocol": "grpc",
            "caption": rpc.caption,
            "request_id": rpc.request_id,
            "model_version": rpc.model_version,
            "image_sha256": rpc.image_sha256,
        },
    ]


def check_invalid_inputs(client, stub, timeout):
    """Проверяет HTTP 422 и INVALID_ARGUMENT на одинаковых ошибках во входных данных."""
    valid = (DEFAULT_ASSETS / "images/cats.jpg").read_bytes()
    invalid = [{}, {"image": b""}, {"image": b"not an image"}]
    for limit in (0, -1, 49):
        invalid.append({"image": valid, "max_new_tokens": limit})
    for values in invalid:
        body = {}
        if "image" in values:
            body["image_base64"] = base64.b64encode(values["image"]).decode()
        if "max_new_tokens" in values:
            body["max_new_tokens"] = values["max_new_tokens"]
        assert client.post("/predict", json=body).status_code == 422
        try:
            stub.Predict(predict_pb2.CaptionRequest(**values), timeout=timeout)
        except grpc.RpcError as error:
            assert error.code() == grpc.StatusCode.INVALID_ARGUMENT
        else:
            raise AssertionError("gRPC accepted invalid input")
    assert client.post("/predict", json={"image_base64": "%%%"}).status_code == 422


def main():
    """Проверяет четыре запроса и по желанию сохраняет ответы для сверки с журналом."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", default="http://127.0.0.1:8000")
    parser.add_argument("--grpc", default="127.0.0.1:50051")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    cases = json.loads((DEFAULT_ASSETS / "cases.json").read_text())
    # Эталон вычисляем локально: в журнале должны появиться только сетевые вызовы.
    runner = ModelRunner(write_log=False)
    receipts = []
    with (
        httpx.Client(base_url=args.http, timeout=args.timeout) as client,
        grpc.insecure_channel(args.grpc) as channel,
    ):
        grpc.channel_ready_future(channel).result(timeout=60)
        health = client.get("/health")
        health.raise_for_status()
        assert health.json() == {"ready": True, "model_version": runner.model_version}
        stub = predict_pb2_grpc.PredictorStub(channel)
        for case in cases:
            receipts.extend(compare_case(case, runner, client, stub, args.timeout))
        check_invalid_inputs(client, stub, args.timeout)
    assert len({r["request_id"] for r in receipts}) == 2 * len(cases)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
    print(
        f"OK: {len(cases)} captions × Python/HTTP/gRPC; 6 shared invalid inputs + invalid HTTP base64"
    )


if __name__ == "__main__":
    main()
