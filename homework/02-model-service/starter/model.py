"""Готовая часть: проверка изображения, загрузка модели, генерация подписи и журнал."""

import hashlib
import json
import os
import threading
import time
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_IMAGE_BYTES = 1_048_576
MAX_PIXELS = 1_048_576
DEFAULT_TOKENS = 24


class CaptionInput(BaseModel):
    """Общий вход модели для HTTP и gRPC: байты файла и необязательный лимит токенов."""

    model_config = ConfigDict(extra="forbid", strict=True)
    image: bytes = Field(min_length=1, max_length=MAX_IMAGE_BYTES)
    max_new_tokens: int | None = Field(default=None, ge=1, le=48)

    @field_validator("image")
    @classmethod
    def valid_image(cls, value):
        """Проверяет размер и содержимое JPEG или PNG и возвращает исходные байты."""
        try:
            with Image.open(BytesIO(value)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG are supported")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Image exceeds 1048576 pixels")
                image.verify()
            # verify() проверяет структуру файла; для чтения пикселей откроем его заново.
            with Image.open(BytesIO(value)) as image:
                image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError("Invalid image bytes") from error
        return value


DEFAULT_ASSETS = Path(__file__).resolve().parents[1] / "assets"
PROMPT = "Describe this image in one sentence."


def verify_model(model_dir):
    """Возвращает манифест, если все файлы модели на месте и их хеши совпадают."""
    model_dir = Path(model_dir)
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model files missing in {model_dir}")
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["files"].items():
        path = model_dir / "model" / name
        if not path.is_file():
            raise FileNotFoundError(f"Model files missing: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Model SHA-256 does not match manifest: {name}")
    return manifest


class ModelRunner:
    """Хранит одну загруженную модель. Оба протокола вызывают один и тот же метод predict."""

    def __init__(self, model_dir=None, log_path=None, *, write_log=True):
        model_dir = Path(model_dir or os.getenv("MODEL_DIR", str(DEFAULT_ASSETS)))
        manifest = verify_model(model_dir)
        # Контрактные тесты обходятся без весов и без импорта PyTorch.
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch.set_num_threads(2)
        self.model_version = manifest["model_version"]
        self.processor = AutoProcessor.from_pretrained(model_dir / "model", local_files_only=True)
        self.processor.image_processor.size = {"longest_edge": 512}
        self.processor.image_processor.do_image_splitting = False
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir / "model",
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        content = [{"type": "image"}, {"type": "text", "text": PROMPT}]
        self.prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
        )
        self.write_log = write_log
        self.log_path = Path(log_path or os.getenv("LOG_PATH", "state/requests.jsonl"))
        self.lock = threading.Lock()
        if write_log:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def predict(self, request: CaptionInput, *, protocol="python"):
        """Возвращает подпись, версию модели, новый ID запроса и хеш исходных байтов."""
        tokens = DEFAULT_TOKENS if request.max_new_tokens is None else request.max_new_tokens
        caption = self._generate_caption(request.image, tokens)
        result = {
            "caption": caption,
            "model_version": self.model_version,
            "request_id": uuid4().hex,
            "image_sha256": hashlib.sha256(request.image).hexdigest(),
        }
        if self.write_log:
            self._log_response(result, protocol, tokens)
        return result

    def _generate_caption(self, image_bytes, tokens):
        """Готовит изображение и декодирует только токены, сгенерированные моделью."""
        import torch

        with Image.open(BytesIO(image_bytes)) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((512, 512))
        # Модель обслуживает один запрос за раз, чтобы ограничить потребление памяти.
        with self.lock, torch.inference_mode():
            inputs = self.processor(text=self.prompt, images=[image], return_tensors="pt")
            ids = self.model.generate(
                **inputs, max_new_tokens=tokens, do_sample=False, num_beams=1
            )
            # generate() возвращает и инструкцию, и продолжение. Клиенту нужна только подпись.
            prompt_length = inputs["input_ids"].shape[1]
            caption_ids = ids[0, prompt_length:]
            caption = self.processor.decode(caption_ids, skip_special_tokens=True).strip()
        if not caption:
            raise RuntimeError("Model returned an empty caption")
        return caption

    def _log_response(self, result, protocol, tokens):
        """Добавляет одну запись в общий журнал HTTP и gRPC. Байты фотографии в журнал не попадают."""
        record = {"time": time.time(), "protocol": protocol, "max_new_tokens": tokens, **result}
        # Оба процесса дописывают в один файл. Передаём всю строку одним вызовом записи.
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode()
        with self.log_path.open("ab", buffering=0) as log:
            log.write(line)
