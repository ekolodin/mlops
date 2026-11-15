"""Проверки готовой части: одна настоящая подпись, ограничения входа и целостность весов."""

import json
from io import BytesIO

import pytest
from model import (
    DEFAULT_ASSETS,
    MAX_IMAGE_BYTES,
    CaptionInput,
    ModelRunner,
    verify_model,
)
from PIL import Image
from pydantic import ValidationError


@pytest.fixture(scope="session")
def runner():
    """Загружает веса один раз на весь запуск pytest и не пишет в журнал сервиса."""
    return ModelRunner(write_log=False)


def test_real_caption(runner):
    image = (DEFAULT_ASSETS / "images/cats.jpg").read_bytes()
    result = runner.predict(CaptionInput(image=image))
    assert result["caption"] and result["model_version"] == runner.model_version
    assert len(result["image_sha256"]) == 64
    assert result["request_id"]


def test_missing_and_explicit_tokens():
    image = (DEFAULT_ASSETS / "images/cats.jpg").read_bytes()
    assert CaptionInput(image=image).max_new_tokens is None
    assert CaptionInput(image=image, max_new_tokens=1).max_new_tokens == 1
    with pytest.raises(ValidationError):
        CaptionInput(image=image, max_new_tokens=0)


@pytest.mark.parametrize("value", [b"", b"not an image", b"x" * (MAX_IMAGE_BYTES + 1)])
def test_invalid_bytes(value):
    with pytest.raises(ValidationError):
        CaptionInput(image=value)


def test_pixel_limit():
    out = BytesIO()
    Image.new("RGB", (1025, 1024)).save(out, format="PNG")
    with pytest.raises(ValidationError, match="pixels"):
        CaptionInput(image=out.getvalue())


def test_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model files missing"):
        verify_model(tmp_path)


def test_model_corruption(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model/config.json").write_text("broken")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"config.json": "0" * 64}}))
    with pytest.raises(ValueError, match="SHA-256"):
        verify_model(tmp_path)


def test_truncated_jpeg_rejected():
    image = (DEFAULT_ASSETS / "images/cats.jpg").read_bytes()
    with pytest.raises(ValidationError):
        CaptionInput(image=image[:-100])
