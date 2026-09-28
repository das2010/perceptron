"""Caché de modelos preentrenados (RF-TRN-11)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from perceptron.catalog import weights_cache as wc
from perceptron.core.errors import NotFoundError, ValidationError


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "caché de modelos"
    monkeypatch.setenv("HF_HOME", str(root / "huggingface"))
    monkeypatch.setenv("TORCH_HOME", str(root / "torch"))
    monkeypatch.delenv("PERCEPTRON_OFFLINE", raising=False)
    blobs = root / "huggingface" / "hub" / "models--org--encoder" / "blobs"
    blobs.mkdir(parents=True)
    weights = b"pesos" * 1000
    (blobs / hashlib.sha256(weights).hexdigest()).write_bytes(weights)
    config = b'{"hidden": 32}'
    git_sha = hashlib.sha1(b"blob %d\0" % len(config) + config, usedforsecurity=False).hexdigest()
    (blobs / git_sha).write_bytes(config)
    ckpts = root / "torch" / "hub" / "checkpoints"
    ckpts.mkdir(parents=True)
    (ckpts / "resnet18.pth").write_bytes(b"x" * 10)
    return root


def test_configure_points_hubs_to_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("HF_HOME", "TORCH_HOME", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    wc.configure(tmp_path / "models_cache")
    import os

    assert os.environ["HF_HOME"] == str(tmp_path / "models_cache" / "huggingface")
    assert os.environ["TORCH_HOME"] == str(tmp_path / "models_cache" / "torch")
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    monkeypatch.setenv("HF_HOME", "/ya/definido")
    wc.configure(tmp_path / "otro")
    assert os.environ["HF_HOME"] == "/ya/definido"  # lo del usuario no se pisa


def test_list_verify_and_delete(cache: Path) -> None:
    report = wc.list_cached(cache)
    by_id = {m.id: m for m in report.models}
    assert set(by_id) == {"org/encoder", "resnet18.pth"}
    assert by_id["org/encoder"].files == 2 and by_id["org/encoder"].size_bytes == 5014
    assert report.total_bytes == 5024
    assert wc.verify() == wc.VerifyReport(checked=2, corrupt=[])
    blob = next((cache / "huggingface" / "hub" / "models--org--encoder" / "blobs").iterdir())
    blob.write_bytes(b"alterado")
    assert len(wc.verify("org/encoder").corrupt) == 1
    assert wc.delete_cached("org/encoder") > 0
    assert wc.delete_cached("resnet18.pth") == 10
    assert wc.list_cached(cache).models == []
    for bad in ("../fuera", "", "org/../x"):
        with pytest.raises((ValidationError, NotFoundError)):
            wc.delete_cached(bad)


def test_prefetch_only_curated_models_and_never_offline(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValidationError, match="catálogo"):
        wc.prefetch("cualquier/modelo")
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    with pytest.raises(ValidationError, match="sin conexión"):
        wc.prefetch("resnet18")


def test_cache_api(client: TestClient, cache: Path) -> None:
    listed = client.get("/api/v1/system/models-cache").json()
    assert {m["id"] for m in listed["models"]} == {"org/encoder", "resnet18.pth"}
    assert client.post("/api/v1/system/models-cache/verify").json()["corrupt"] == []
    after = client.delete("/api/v1/system/models-cache/org/encoder").json()
    assert [m["id"] for m in after["models"]] == ["resnet18.pth"]
    bad = client.post("/api/v1/system/models-cache/prefetch", json={"model": "x/y"})
    assert bad.status_code == 422
