"""Загрузка бинарников не должна портить существующую папку или принимать неверный хеш."""

import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from install_monitoring import install_archive


def archive_at(tmp_path):
    path = tmp_path / "tool.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("release/bin/tool")
        entry.size = 5
        entry.mode = 0o755
        archive.addfile(entry, io.BytesIO(b"hello"))
    return path


def test_archive_checksum_checked_before_install(tmp_path):
    archive = archive_at(tmp_path)
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="SHA-256"):
        install_archive(archive.as_uri(), "0" * 64, destination)
    assert not destination.exists()


def test_valid_archive_is_installed_without_nested_release_dir(tmp_path):
    archive = archive_at(tmp_path)
    destination = tmp_path / "installed"
    install_archive(archive.as_uri(), hashlib.sha256(archive.read_bytes()).hexdigest(), destination)
    assert (destination / "bin/tool").read_bytes() == b"hello"


def test_existing_directory_is_not_overwritten(tmp_path):
    destination = tmp_path / "installed"
    destination.mkdir()
    (destination / "keep").write_text("original")
    with pytest.raises(FileExistsError):
        install_archive("file:///nonexistent", "0" * 64, destination)
    assert (destination / "keep").read_text() == "original"
