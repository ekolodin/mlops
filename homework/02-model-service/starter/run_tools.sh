#!/usr/bin/env bash
# Из starter/: bash run_tools.sh pytest -q tests/test_given.py
set -euo pipefail
cd "$(dirname "$0")"
if ! docker network inspect hw2-network >/dev/null 2>&1; then
  docker network create hw2-network >/dev/null
fi
# Папка задания видна под /homework; отдельный том /venv сохраняет зависимости между запусками.
docker run --rm -i --network hw2-network \
  --mount "type=bind,source=$(pwd)/..,target=/homework" \
  --mount type=volume,source=hw2-tools-env,target=/venv \
  -e UV_PROJECT_ENVIRONMENT=/venv -e UV_CACHE_DIR=/tmp/uv-cache -e UV_LINK_MODE=copy \
  -e HF_HUB_DISABLE_XET=1 -e TOKENIZERS_PARALLELISM=false \
  -w /homework/starter \
  ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca \
  sh -c 'uv sync --locked && exec uv run --locked "$@"' sh "$@"
