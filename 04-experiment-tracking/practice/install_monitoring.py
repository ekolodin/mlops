"""Скачать нативные Prometheus и Grafana только в .tools этой практики."""

import argparse
import hashlib
from pathlib import Path
import platform
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlopen

PROMETHEUS_VERSION = "3.15.0"
GRAFANA_VERSION = "13.2.3"
GRAFANA_BUILD = "36482603486"
# Checksums from the official download pages, checked on 2026-10-07.
CHECKSUMS = {
    ("prometheus", "darwin", "amd64"): "2d79e744c2d7e505db936fbc898e05abc74fcb6e437c25befd26e9c9f00aa58b",
    ("prometheus", "darwin", "arm64"): "920df4d17e78b3b0175af144eb318b0c74d1cf7b1d1251b326966f0e81977260",
    ("prometheus", "linux", "amd64"): "2a542df32eac02ee17b9d844fb2aa1de00dafa5476579ba8a3ba862e9d572ea0",
    ("grafana", "darwin", "amd64"): "59d08d3ef611a49239bf3b54cd1cb1b215611a3f6cd37bb6e6abe40dda181319",
    ("grafana", "darwin", "arm64"): "248a51bcfacdb1ec642006cee46b4de44a34bcf41ddea88d96a4605e2e9d808c",
    ("grafana", "linux", "amd64"): "6107ad27016296aac38e0d7ffa8753ab540b5541ad27e94790f771289d733235",
}


def install_archive(url, checksum, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="download-", dir=destination.parent) as scratch:
        scratch = Path(scratch)
        archive_path = scratch / "release.tar.gz"
        digest = hashlib.sha256()
        with urlopen(url, timeout=60) as response, archive_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError("Archive SHA-256 differs from the official checksum")
        unpacked = scratch / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            # Python's data filter rejects traversal and dangerous links.
            archive.extractall(unpacked, filter="data")
        entries = list(unpacked.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise ValueError("Expected one release directory in the archive")
        entries[0].rename(destination)
        (destination / ".archive-sha256").write_text(checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("all", "prometheus", "grafana"), default="all")
    args = parser.parse_args()
    system = platform.system().lower()
    architecture = {"x86_64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine())
    if ("grafana", system, architecture) not in CHECKSUMS:
        parser.error("Supported: macOS Intel/Apple Silicon, Linux x86_64 (including WSL)")
    root = Path(__file__).resolve().parent / ".tools"
    for tool in ("prometheus", "grafana"):
        if args.tool not in {"all", tool}:
            continue
        release = PROMETHEUS_VERSION if tool == "prometheus" else GRAFANA_VERSION
        destination = root / f"{tool}-{release}"
        checksum = CHECKSUMS[tool, system, architecture]
        marker = destination / ".archive-sha256"
        binary = destination / ("prometheus" if tool == "prometheus" else "bin/grafana")
        if marker.exists() and marker.read_text() == checksum and binary.exists():
            print(f"{tool} {release}: уже установлен", flush=True)
            continue
        if tool == "prometheus":
            url = f"https://github.com/prometheus/prometheus/releases/download/v{release}/prometheus-{release}.{system}-{architecture}.tar.gz"
        else:
            url = f"https://dl.grafana.com/grafana/release/{release}/grafana_{release}_{GRAFANA_BUILD}_{system}_{architecture}.tar.gz"
        print(f"Скачиваем {tool} {release} ({system}/{architecture})", flush=True)
        install_archive(url, checksum, destination)
        print(f"Установлен: {destination}", flush=True)


if __name__ == "__main__":
    main()
