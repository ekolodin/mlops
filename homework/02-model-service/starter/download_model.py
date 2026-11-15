"""Один раз скачать публичные веса по зафиксированной ревизии и проверить SHA-256."""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import hashlib
import json

from huggingface_hub import hf_hub_download
from model import DEFAULT_ASSETS, verify_model


def main():
    """Докачивает отсутствующие или повреждённые файлы и проверяет комплект по манифесту."""
    manifest = json.loads((DEFAULT_ASSETS / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = DEFAULT_ASSETS / "model" / name
        if path.is_file():
            with path.open("rb") as stream:
                actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_digest == digest:
                continue
            # Удаляем только повреждённую копию, чтобы скачать файл заново.
            path.unlink()
        print(f"Downloading/checking {name}", flush=True)
        hf_hub_download(
            repo_id=manifest["repo_id"],
            revision=manifest["revision"],
            filename=name,
            local_dir=DEFAULT_ASSETS / "model",
        )
    verify_model(DEFAULT_ASSETS)
    print("Model download OK; all SHA-256 match", flush=True)


if __name__ == "__main__":
    main()
