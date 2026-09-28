"""Retención de versiones de datos y export compatible con DVC (RF-MON-07).

Retención: se conservan las últimas `keep_last` versiones del proyecto y, siempre, las que
están en uso (runs, etiquetado, reentrenamientos, agente) y los padres de las que quedan
(el linaje no se rompe). El resto se borra: entidad, perfil y, si ninguna otra versión
comparte el contenido, la carpeta.

DVC: `.dvc` en formato DVC 3 (md5 por archivo y hash `.dir` del listado), para agregar la
versión a un repositorio DVC sin volver a hashear.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from perceptron.core.errors import ValidationError
from perceptron.domain.models import (
    AgentRun,
    DatasetVersion,
    LabelSet,
    Profile,
    RetrainRun,
    Run,
)

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

USERS = (Run, LabelSet, RetrainRun, AgentRun)


class RetentionReport(BaseModel):
    dry_run: bool
    keep_last: int
    kept: list[str]
    in_use: list[str] = Field(description="Se conservan por estar en uso o ser padres")
    deleted: list[str]
    freed_bytes: int


def _in_use(ctx: EngineContext, project_id: str) -> set[str]:
    used: set[str] = set()
    for model in USERS:
        for e in ctx.repo(model).list(filters={"project_id": project_id}, limit=100_000):
            dv_id = getattr(e, "dataset_version_id", None)
            if dv_id:
                used.add(str(dv_id))
    return used


def apply_retention(
    ctx: EngineContext, project_id: str, *, keep_last: int, dry_run: bool = True
) -> RetentionReport:
    if keep_last < 1:
        raise ValidationError("hay que conservar al menos una versión")
    ctx.projects.get(project_id)
    versions = list(
        ctx.repo(DatasetVersion).list(filters={"project_id": project_id}, limit=100_000)
    )
    by_id = {v.id: v for v in versions}  # lista del más nuevo al más viejo
    keep = {v.id for v in versions[:keep_last]}
    protected = _in_use(ctx, project_id) & set(by_id)
    # Padres de todo lo que queda: el linaje se mantiene.
    kept_all = keep | protected
    stack = list(kept_all)
    while stack:
        v = by_id.get(stack.pop())
        if v is not None and v.parent_id in by_id and v.parent_id not in kept_all:
            kept_all.add(v.parent_id)
            stack.append(v.parent_id)
    doomed = [v for v in versions if v.id not in kept_all]
    paths = ctx.settings.paths.project(project_id)
    kept_hashes = {v.content_hash for v in versions if v.id in kept_all}
    freed, removed_hashes = 0, set()
    for v in doomed:
        folder = paths.dataset(v.content_hash)
        # El contenido es content-addressed: la carpeta se borra si nadie más la usa.
        owns_folder = v.content_hash not in kept_hashes and v.content_hash not in removed_hashes
        if owns_folder:
            removed_hashes.add(v.content_hash)
            if folder.is_dir():
                freed += sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        if dry_run:
            continue
        for profile in ctx.repo(Profile).list(filters={"dataset_version_id": v.id}, limit=100):
            ctx.repo(Profile).delete(profile.id)
        ctx.repo(DatasetVersion).delete(v.id)
        if owns_folder:
            shutil.rmtree(folder, ignore_errors=True)
    return RetentionReport(
        dry_run=dry_run,
        keep_last=keep_last,
        kept=[v.id for v in versions if v.id in keep],
        in_use=sorted(kept_all - keep),
        deleted=[v.id for v in doomed],
        freed_bytes=freed,
    )


# ------------------------------------------------------------------ DVC


def _md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dvc_tree(folder: Path) -> tuple[str, int, int, list[dict[str, str]]]:
    """(hash .dir, tamaño, archivos, listado) como los calcula DVC 3 para un directorio."""
    entries = [
        {"md5": _md5(f), "relpath": f.relative_to(folder).as_posix()}
        for f in sorted(folder.rglob("*"))
        if f.is_file() and not f.is_symlink()
    ]
    entries.sort(key=lambda e: e["relpath"])
    raw = json.dumps(entries, sort_keys=True).encode("utf-8")
    size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file() and not f.is_symlink())
    return hashlib.md5(raw, usedforsecurity=False).hexdigest() + ".dir", size, len(entries), entries


def dvc_file(name: str, digest: str, size: int, nfiles: int) -> str:
    return (
        f"outs:\n- md5: {digest}\n  size: {size}\n  nfiles: {nfiles}\n  hash: md5\n  path: {name}\n"
    )


def export_dvc(ctx: EngineContext, dataset_version_id: str, out: Path) -> Path:
    """Zip con `<nombre>/` (los archivos de la versión) y `<nombre>.dvc`."""
    dv = ctx.repo(DatasetVersion).get(dataset_version_id)
    folder = ctx.settings.paths.project(dv.project_id).dataset(dv.content_hash)
    if not folder.is_dir():
        raise ValidationError("los archivos de esta versión no están en el workspace")
    name = f"dataset-{dv.content_hash[:12]}"
    digest, size, nfiles, _ = dvc_tree(folder)
    meta: dict[str, Any] = {"perceptron_dataset_version": dv.id, "content_hash": dv.content_hash}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        z.writestr(f"{name}.dvc", dvc_file(name, digest, size, nfiles))
        z.writestr("perceptron.json", json.dumps(meta, indent=2))
        for f in sorted(folder.rglob("*")):
            if f.is_file() and not f.is_symlink():
                z.write(f, f"{name}/{f.relative_to(folder).as_posix()}")
    return out
