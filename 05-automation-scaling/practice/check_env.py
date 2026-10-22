"""Проверка до пары: только стандартная библиотека Python, без установки ML-пакетов."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / "data/manifest.json").read_text())
assert hashlib.sha256((root / "data/train.csv").read_bytes()).hexdigest() == manifest["sha256"]
for command in (["docker", "version", "--format", "{{.Server.Version}}"],
                ["docker", "compose", "version"]):
    try:
        result = subprocess.run(command, text=True, capture_output=True)
    except FileNotFoundError:
        raise SystemExit("Команда docker не найдена: установите Docker и откройте новый терминал.")
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise SystemExit("Docker недоступен: запустите Docker Desktop / Docker Engine.")
    print(result.stdout.strip())
print("OK: Docker, Compose; Titanic 713 строк, SHA-256 совпадает")
