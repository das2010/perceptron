"""Caché de modelos preentrenados (RF-TRN-11): descarga única, checksum, uso offline y
gestión de espacio.

Hugging Face Hub (encoders de texto y la mayoría de los pesos de timm) y torch hub guardan
todo debajo de `<workspace>/models_cache` (o `PERCEPTRON_MODELS_CACHE`, p. ej. una carpeta de
red). En el Team Server el workspace es un volumen compartido por el servidor y los workers,
así que la caché también. Con `PERCEPTRON_OFFLINE` no se descarga nada (`HF_HUB_OFFLINE`).

El Hub guarda cada archivo LFS con su sha256 como nombre de blob: `verify` lo recalcula.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from perceptron.core.errors import NotFoundError, ValidationError

Source = Literal["huggingface", "torch"]


class CachedModel(BaseModel):
    id: str
    source: Source
    size_bytes: int
    files: int
    last_used: datetime | None = None


class CacheReport(BaseModel):
    path: str
    offline: bool
    total_bytes: int
    models: list[CachedModel]
    available: list[str] = Field(
        default_factory=list, description="Modelos curados que se pueden predescargar"
    )


class VerifyReport(BaseModel):
    checked: int
    corrupt: list[str]


def offline() -> bool:
    return os.environ.get("PERCEPTRON_OFFLINE", "").lower() in {"1", "true", "yes"}


def configure(cache_dir: Path) -> Path:
    """Apunta HF Hub y torch hub a la caché del workspace (sin pisar lo que definió el
    usuario) y respeta el modo offline. Los workers heredan el entorno del Engine."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
    if offline():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return cache_dir


def _hub_dir() -> Path:
    return Path(os.environ.get("HF_HOME", "")) / "hub"


def _torch_dir() -> Path:
    return Path(os.environ.get("TORCH_HOME", "")) / "hub" / "checkpoints"


def _dir_stats(path: Path) -> tuple[int, int, datetime | None]:
    size, count, last = 0, 0, 0.0
    for f in path.rglob("*"):
        if f.is_file() and not f.is_symlink():
            st = f.stat()
            size += st.st_size
            count += 1
            last = max(last, st.st_atime, st.st_mtime)
    return size, count, datetime.fromtimestamp(last, UTC) if last else None


def list_cached(cache_dir: Path) -> CacheReport:
    models: list[CachedModel] = []
    hub = _hub_dir()
    if hub.is_dir():
        for repo in sorted(hub.glob("models--*")):
            size, count, last = _dir_stats(repo / "blobs")
            model_id = repo.name.removeprefix("models--").replace("--", "/")
            models.append(
                CachedModel(
                    id=model_id, source="huggingface", size_bytes=size, files=count, last_used=last
                )
            )
    ckpts = _torch_dir()
    if ckpts.is_dir():
        for f in sorted(ckpts.iterdir()):
            if f.is_file():
                st = f.stat()
                models.append(
                    CachedModel(
                        id=f.name,
                        source="torch",
                        size_bytes=st.st_size,
                        files=1,
                        last_used=datetime.fromtimestamp(max(st.st_atime, st.st_mtime), UTC),
                    )
                )
    return CacheReport(
        path=str(cache_dir),
        offline=offline(),
        total_bytes=sum(m.size_bytes for m in models),
        models=models,
    )


def _entries() -> dict[str, Path]:
    """Id → ruta de lo que hay en la caché. Las rutas salen del listado del disco, nunca del
    id que manda el cliente (no hay forma de armar una ruta fuera de la caché)."""
    found: dict[str, Path] = {}
    hub = _hub_dir()
    if hub.is_dir():
        for repo in hub.glob("models--*"):
            if repo.is_dir() and not repo.is_symlink():
                found[repo.name.removeprefix("models--").replace("--", "/")] = repo
    ckpts = _torch_dir()
    if ckpts.is_dir():
        for f in ckpts.iterdir():
            if f.is_file() and not f.is_symlink():
                found[f.name] = f
    return found


def _entry(model_id: str) -> Path:
    if not model_id or any(part in ("", ".", "..") for part in model_id.split("/")):
        raise ValidationError(f"modelo inválido: {model_id!r}")
    path = _entries().get(model_id)
    if path is None:
        raise NotFoundError(f"{model_id} no está en la caché")
    return path


def delete_cached(model_id: str) -> int:
    """Borra un modelo de la caché; devuelve los bytes liberados."""
    path = _entry(model_id)
    if path.is_file():
        size = path.stat().st_size
        path.unlink()
        return size
    size, _, _ = _dir_stats(path / "blobs")
    shutil.rmtree(path, ignore_errors=True)
    return size


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(model_id: str | None = None) -> VerifyReport:
    """Recalcula el hash de cada blob del Hub (sha256 en LFS, sha1 de git en el resto)."""
    hub = _hub_dir()
    repos = (
        [_entry(model_id)] if model_id else sorted(hub.glob("models--*")) if hub.is_dir() else []
    )
    checked, corrupt = 0, []
    for repo in repos:
        blobs = repo / "blobs"
        if not blobs.is_dir():
            continue
        for blob in blobs.iterdir():
            if not blob.is_file() or blob.name.endswith(".incomplete"):
                continue
            checked += 1
            expected = blob.name
            got = _sha256(blob) if len(expected) == 64 else _git_blob_sha1(blob)
            if got != expected:
                corrupt.append(
                    f"{repo.name.removeprefix('models--').replace('--', '/')}:{blob.name}"
                )
    return VerifyReport(checked=checked, corrupt=corrupt)


def prefetch(model: str) -> str:
    """Descarga ahora un modelo curado del catálogo para usarlo después sin conexión."""
    from perceptron.catalog.registry import HF_TEXT_MODELS, TIMM_WEIGHTS

    if offline():
        raise ValidationError("sin conexión (PERCEPTRON_OFFLINE): no se puede descargar")
    if model in HF_TEXT_MODELS:
        from huggingface_hub import snapshot_download

        snapshot_download(model)
        return model
    if model in TIMM_WEIGHTS:
        import timm

        timm.create_model(model, pretrained=True)
        return model
    raise ValidationError(f"{model} no está en el catálogo de pesos curados")
