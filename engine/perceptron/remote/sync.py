"""Sincronización desktop → servidor y vuelta (RF-SRV-03).

- **push**: el proyecto como proyecto de equipo (mismo id), la versión de datos y su perfil,
  el pipeline y la arquitectura, más sus archivos (subida por chunks, reanudable y
  deduplicada por SHA-256).
- **pull**: el estudio y sus runs (entidades y archivos del run) para verlos en el desktop.

Las versiones del servidor conocidas por el desktop se guardan en `remote.json` del proyecto
(bloqueo optimista: si alguien cambió algo en el servidor, el push da conflicto).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from perceptron.core.errors import ConflictError
from perceptron.domain.enums import ProjectScope
from perceptron.domain.models import (
    ArchSpecRecord,
    DatasetVersion,
    Entity,
    Evaluation,
    Pipeline,
    Profile,
    Run,
    Study,
)
from perceptron.remote.client import RemoteClient, RemoteError

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

CHUNK = 8 * 1024 * 1024
Progress = Callable[..., None]


def _noop(*_: Any, **__: Any) -> None:
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RemoteState:
    """Versiones del servidor que conoce este desktop, por proyecto y servidor."""

    def __init__(self, project_root: Path, server: str) -> None:
        self.file = project_root / "remote.json"
        self.server = server
        raw = json.loads(self.file.read_text(encoding="utf-8")) if self.file.is_file() else {}
        self.data: dict[str, Any] = raw
        self.mine: dict[str, Any] = self.data.setdefault(server, {"versions": {}})

    def version(self, entity_id: str) -> int | None:
        value = self.mine["versions"].get(entity_id)
        return int(value) if value is not None else None

    def remember(self, entity_id: str, version: int) -> None:
        self.mine["versions"][entity_id] = version
        self.file.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


class ProjectSync:
    def __init__(self, ctx: EngineContext, client: RemoteClient, project_id: str) -> None:
        self.ctx = ctx
        self.client = client
        self.project_id = project_id
        self.root = ctx.settings.paths.project(project_id).root
        self.state = RemoteState(self.root, client.server.name)

    # ------------------------------------------------------------------ push

    def push_project(self, workspace_id: str | None = None) -> None:
        project = self.ctx.projects.get(self.project_id)
        data = project.model_dump(mode="json")
        if workspace_id:
            data["workspace_id"] = workspace_id
        res = self.client.json(
            "PUT",
            f"/sync/projects/{project.id}",
            json={"project": data, "base_version": self.state.version(project.id)},
        )
        self.state.remember(project.id, int(res["version"]))

    def push_entity(self, entity: Entity) -> None:
        kind = type(entity).__name__
        body = {
            "data": entity.model_dump(mode="json"),
            "base_version": self.state.version(entity.id),
        }
        try:
            res = self.client.json(
                "PUT", f"/sync/projects/{self.project_id}/entities/{kind}/{entity.id}", json=body
            )
        except RemoteError as exc:
            if exc.details.get("status") == 409:
                raise ConflictError(
                    f"{kind} {entity.id} cambió en el servidor: bajá la versión nueva antes",
                    details=exc.details,
                ) from None
            raise
        self.state.remember(entity.id, int(res["version"]))

    def upload(self, path: Path, progress: Progress = _noop) -> bool:
        """Sube un archivo del proyecto; devuelve False si el servidor ya lo tenía igual."""
        rel = path.relative_to(self.root).as_posix()
        size = path.stat().st_size
        start = self.client.json(
            "POST",
            f"/sync/projects/{self.project_id}/uploads",
            json={"path": rel, "size": size, "sha256": _sha256(path)},
        )
        if start["complete"]:
            return False
        upload_id, offset = start["upload_id"], int(start["offset"])
        with path.open("rb") as f:
            f.seek(offset)
            while offset < size:
                chunk = f.read(CHUNK)
                res = self.client.json(
                    "PUT",
                    f"/sync/uploads/{upload_id}",
                    params={"offset": offset},
                    content=chunk,
                    headers={"Content-Type": "application/octet-stream"},
                )
                offset = int(res["offset"])
                progress("upload", path=rel, sent=offset, size=size)
        self.client.json("POST", f"/sync/uploads/{upload_id}/complete")
        return True

    def _files(self, folders: Iterable[Path]) -> list[Path]:
        out: list[Path] = []
        for folder in folders:
            if folder.is_file():
                out.append(folder)
            elif folder.is_dir():
                out.extend(
                    p for p in sorted(folder.rglob("*")) if p.is_file() and not p.is_symlink()
                )
        return out

    def push_study_inputs(
        self,
        dataset_version_id: str,
        pipeline_id: str,
        archspec_id: str,
        *,
        workspace_id: str | None = None,
        progress: Progress = _noop,
    ) -> dict[str, int]:
        """Todo lo que un worker del servidor necesita para entrenar este estudio."""
        ctx = self.ctx
        paths = ctx.settings.paths.project(self.project_id)
        dv = ctx.repo(DatasetVersion).get(dataset_version_id)
        pipeline = ctx.repo(Pipeline).get(pipeline_id)
        arch = ctx.repo(ArchSpecRecord).get(archspec_id)
        self.push_project(workspace_id)
        progress("entities")
        for entity in (
            dv,
            *ctx.repo(Profile).list(filters={"dataset_version_id": dv.id}),
            pipeline,
            arch,
        ):
            self.push_entity(entity)
        folders = [paths.pipelines_dir, paths.archspecs_dir, paths.code_dir]
        folders.append(self.root / dv.path if dv.path else paths.datasets_dir / dv.id)
        uploaded = skipped = 0
        for file in self._files(folders):
            if self.upload(file, progress):
                uploaded += 1
            else:
                skipped += 1
        progress("pushed", uploaded=uploaded, skipped=skipped)
        return {"uploaded": uploaded, "skipped": skipped}

    def promote(
        self,
        *,
        workspace_id: str | None,
        dataset_version_ids: list[str],
        run_ids: list[str],
        progress: Progress = _noop,
    ) -> dict[str, int]:
        """Proyecto local → proyecto de equipo (RF-PRJ-04): metadata, los datasets elegidos (y
        los de los runs), pipelines, arquitecturas y los runs elegidos con sus evaluaciones y
        artefactos. Lo que no se elige queda solo en el desktop."""
        ctx = self.ctx
        paths = ctx.settings.paths.project(self.project_id)
        runs = [ctx.repo(Run).get(r) for r in run_ids]
        for run in runs:
            if run.project_id != self.project_id:
                raise ConflictError(f"el run {run.id} es de otro proyecto")
        dv_ids = list(dict.fromkeys([*dataset_version_ids, *(r.dataset_version_id for r in runs)]))
        dvs = [ctx.repo(DatasetVersion).get(d) for d in dv_ids]
        self.push_project(workspace_id)
        progress("entities")
        entities: list[Entity] = []
        for dv in dvs:
            entities += [dv, *ctx.repo(Profile).list(filters={"dataset_version_id": dv.id})]
        entities += list(ctx.repo(Pipeline).list(filters={"project_id": self.project_id}))
        entities += list(ctx.repo(ArchSpecRecord).list(filters={"project_id": self.project_id}))
        study_ids = {r.study_id for r in runs if r.study_id}
        entities += [ctx.repo(Study).get(s) for s in sorted(study_ids)]
        for run in runs:
            entities += [run, *ctx.repo(Evaluation).list(filters={"run_id": run.id})]
        for entity in entities:
            self.push_entity(entity)
        folders = [paths.pipelines_dir, paths.archspecs_dir, paths.code_dir]
        folders += [self.root / dv.path if dv.path else paths.datasets_dir / dv.id for dv in dvs]
        folders += [paths.run(r.id) for r in runs]
        uploaded = skipped = 0
        for file in self._files(folders):
            if self.upload(file, progress):
                uploaded += 1
            else:
                skipped += 1
        project = ctx.projects.get(self.project_id)
        if project.scope is not ProjectScope.TEAM:
            ctx.projects.update(project.model_copy(update={"scope": ProjectScope.TEAM}))
        progress("promoted", uploaded=uploaded, skipped=skipped)
        return {
            "entities": len(entities),
            "datasets": len(dvs),
            "runs": len(runs),
            "uploaded": uploaded,
            "skipped": skipped,
        }

    # ------------------------------------------------------------------ pull

    def _upsert(self, entity: Entity) -> None:
        repo = self.ctx.repo(type(entity))
        current = repo.find(entity.id)
        if current is None:
            repo.add(entity)
        else:
            repo.update(entity.model_copy(update={"version": current.version}))
        self.state.remember(entity.id, entity.version)

    def download(self, rel: str) -> Path:
        target = self.root / Path(*rel.split("/"))
        res = self.client.request(
            "GET", f"/sync/projects/{self.project_id}/file", params={"path": rel}
        )
        if res.status_code >= 400:
            raise RemoteError(f"no se pudo bajar {rel}", details={"status": res.status_code})
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(res.content)
        return target

    def pull_study(self, study_id: str) -> list[Run]:
        """Baja el estudio y sus runs (entidades + archivos) al desktop."""
        pid = self.project_id
        for data in self.client.json(
            "GET", f"/sync/projects/{pid}/entities/Study", params={"ids": [study_id]}
        ):
            self._upsert(Study.model_validate(data))
        runs = [
            Run.model_validate(d)
            for d in self.client.json(
                "GET", f"/sync/projects/{pid}/entities/Run", params={"study_id": study_id}
            )
        ]
        # Carpeta del estudio (storage de Optuna) y la de cada trial.
        for folder in (study_id, *(run.id for run in runs)):
            for f in self.client.json(
                "GET", f"/sync/projects/{pid}/files", params={"prefix": f"runs/{folder}"}
            ):
                self.download(str(f["path"]))
        for run in runs:
            self._upsert(run)
        return runs
