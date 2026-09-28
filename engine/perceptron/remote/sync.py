"""Sincronización desktop → servidor y vuelta (RF-SRV-03).

- **push**: el proyecto como proyecto de equipo (mismo id), la versión de datos y su perfil,
  el pipeline y la arquitectura, más sus archivos (subida por chunks, reanudable y
  deduplicada por SHA-256).
- **pull**: el estudio y sus runs (entidades y archivos del run) para verlos en el desktop.
- **edición concurrente**: `status` compara cada entidad editable (proyecto, datos, perfiles,
  pipelines y arquitecturas) con la del servidor y la clasifica en al día, para bajar, para
  subir o en conflicto (cambió en los dos lados); `pull` baja lo nuevo y resuelve cada
  conflicto con la versión del servidor (`theirs`) o con la propia (`mine`, que se sube
  encima). Nada se pisa sin una decisión explícita.

Las versiones del servidor conocidas por el desktop se guardan en `remote.json` del proyecto,
junto con la versión local de ese momento (bloqueo optimista: si alguien cambió algo en el
servidor, el push da conflicto).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from perceptron.core.errors import ConflictError
from perceptron.domain.enums import ProjectScope
from perceptron.domain.models import (
    ArchSpecRecord,
    DatasetVersion,
    Entity,
    Evaluation,
    Pipeline,
    Profile,
    Project,
    Run,
    Study,
)
from perceptron.remote.client import RemoteClient, RemoteError

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

CHUNK = 8 * 1024 * 1024
Progress = Callable[..., None]
# Lo que se edita en los dos lados (estudios y runs los genera quien entrena: se bajan).
EDITABLE: tuple[type[Entity], ...] = (DatasetVersion, Profile, Pipeline, ArchSpecRecord)
_VOLATILE = {"version", "created_at", "updated_at"}
SyncState = Literal["synced", "pull", "push", "conflict", "server_only", "local_only"]
Resolution = Literal["theirs", "mine"]


class EntitySync(BaseModel):
    kind: str
    id: str
    name: str | None = None
    state: SyncState
    local_version: int | None = None
    server_version: int | None = None


class SyncStatus(BaseModel):
    server: str
    items: list[EntitySync]

    def count(self, state: SyncState) -> int:
        return sum(1 for i in self.items if i.state == state)


class PullOutcome(BaseModel):
    pulled: int
    pushed: int
    conflicts_left: list[str]
    downloaded: int


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

    def local_version(self, entity_id: str) -> int | None:
        value = self.mine.get("local", {}).get(entity_id)
        return int(value) if value is not None else None

    def remember(self, entity_id: str, version: int, local_version: int | None = None) -> None:
        self.mine["versions"][entity_id] = version
        if local_version is not None:
            self.mine.setdefault("local", {})[entity_id] = local_version
        self.file.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


class ProjectSync:
    def __init__(self, ctx: EngineContext, client: RemoteClient, project_id: str) -> None:
        self.ctx = ctx
        self.client = client
        # El id que va en las URLs del servidor es el del proyecto guardado, no el del pedido.
        self.project_id = ctx.projects.get(project_id).id
        self.root = ctx.settings.paths.project(self.project_id).root
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
        self.state.remember(project.id, int(res["version"]), project.version)

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
        self.state.remember(entity.id, int(res["version"]), entity.version)

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
            saved = repo.add(entity)
        else:
            saved = repo.update(entity.model_copy(update={"version": current.version}))
        self.state.remember(entity.id, entity.version, saved.version)

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

    # ------------------------------------------------------------------ edición concurrente

    @staticmethod
    def _same(a: Entity, b: Entity) -> bool:
        return a.model_dump(mode="json", exclude=_VOLATILE) == b.model_dump(
            mode="json", exclude=_VOLATILE
        )

    def _classify(self, local: Entity | None, server: Entity | None) -> SyncState:
        if local is None:
            return "server_only"
        if server is None:
            return "local_only"
        if self._same(local, server):
            return "synced"
        known, known_local = self.state.version(local.id), self.state.local_version(local.id)
        server_changed = known is None or server.version != known
        local_changed = known_local is None or local.version != known_local
        if server_changed and local_changed:
            return "conflict"
        return "pull" if server_changed else "push"

    def _server_entities(self) -> dict[str, Entity]:
        out: dict[str, Entity] = {}
        for model in EDITABLE:
            for data in self.client.json(
                "GET", f"/sync/projects/{self.project_id}/entities/{model.__name__}"
            ):
                entity = model.model_validate(data)
                out[entity.id] = entity
        return out

    def _local_entities(self) -> dict[str, Entity]:
        out: dict[str, Entity] = {}
        for model in EDITABLE:
            filters = {"project_id": self.project_id} if "project_id" in model.model_fields else {}
            items = list(self.ctx.repo(model).list(filters=filters, limit=10_000))
            if model is Profile:
                dvs = {e.id for e in out.values() if isinstance(e, DatasetVersion)}
                items = [p for p in items if getattr(p, "dataset_version_id", None) in dvs]
            out.update({e.id: e for e in items})
        return out

    def _server_project(self) -> Project | None:
        try:
            data = self.client.json("GET", f"/sync/projects/{self.project_id}")
        except RemoteError as exc:
            if exc.details.get("status") == 404:
                return None
            raise
        return Project.model_validate(data["project"])

    def _snapshot(
        self,
    ) -> tuple[Project, Project | None, dict[str, Entity], dict[str, Entity], SyncStatus]:
        local_project = self.ctx.projects.get(self.project_id)
        server_project = self._server_project()
        local, server = self._local_entities(), self._server_entities()
        items = [
            EntitySync(
                kind="Project",
                id=local_project.id,
                name=local_project.name,
                state=self._project_state(local_project, server_project),
                local_version=local_project.version,
                server_version=server_project.version if server_project else None,
            )
        ]
        for eid in sorted({*local, *server}):
            lo, se = local.get(eid), server.get(eid)
            ref = lo or se
            if ref is None:
                continue
            items.append(
                EntitySync(
                    kind=type(ref).__name__,
                    id=eid,
                    name=getattr(ref, "name", None),
                    state=self._classify(lo, se),
                    local_version=lo.version if lo else None,
                    server_version=se.version if se else None,
                )
            )
        status = SyncStatus(server=self.client.server.name, items=items)
        return local_project, server_project, local, server, status

    _PROJECT_KEEP = frozenset(
        {"id", "version", "created_at", "updated_at", "workspace_id", "scope"}
    )

    def _project_state(self, local: Project, server: Project | None) -> SyncState:
        if server is None:
            return "local_only"
        a = local.model_dump(mode="json", exclude=set(self._PROJECT_KEEP))
        b = server.model_dump(mode="json", exclude=set(self._PROJECT_KEEP))
        if a == b:
            return "synced"
        known, known_local = self.state.version(local.id), self.state.local_version(local.id)
        server_changed = known is None or server.version != known
        local_changed = known_local is None or local.version != known_local
        if server_changed and local_changed:
            return "conflict"
        return "pull" if server_changed else "push"

    def status(self) -> SyncStatus:
        return self._snapshot()[4]

    def _pull_files(self, entity: Entity) -> int:
        """Archivos de lo que se baja: la carpeta de una versión de datos si falta, el código."""
        prefixes: list[str] = []
        if isinstance(entity, DatasetVersion):
            rel = entity.path or f"datasets/{entity.id}"
            if not (self.root / rel).is_dir():
                prefixes.append(rel)
        elif isinstance(entity, ArchSpecRecord) and entity.code_path:
            prefixes.append(entity.code_path)
        n = 0
        for prefix in prefixes:
            for f in self.client.json(
                "GET", f"/sync/projects/{self.project_id}/files", params={"prefix": prefix}
            ):
                self.download(str(f["path"]))
                n += 1
        return n

    def pull(
        self, resolutions: dict[str, Resolution] | None = None, progress: Progress = _noop
    ) -> PullOutcome:
        """Baja lo que cambió en el servidor; los conflictos se resuelven solo si se indica."""
        resolutions = resolutions or {}
        local_project, server_project, local, server, status = self._snapshot()
        pulled = pushed = downloaded = 0
        left: list[str] = []
        order = {m.__name__: i for i, m in enumerate(EDITABLE)}  # datos antes que perfiles
        for item in sorted(status.items, key=lambda i: order.get(i.kind, -1)):
            choice = resolutions.get(item.id)
            if item.state == "conflict" and choice is None:
                left.append(item.id)
                continue
            take = item.state in ("pull", "server_only") or (
                item.state == "conflict" and choice == "theirs"
            )
            force = item.state == "conflict" and choice == "mine"
            if item.kind == "Project":
                if take and server_project is not None:
                    fields = server_project.model_dump(exclude=set(self._PROJECT_KEEP))
                    saved = self.ctx.projects.update(local_project.model_copy(update=fields))
                    self.state.remember(saved.id, server_project.version, saved.version)
                    pulled += 1
                elif force and server_project is not None:
                    self.state.remember(local_project.id, server_project.version)
                    self.push_project()
                    pushed += 1
                continue
            if take:
                entity = server[item.id]
                downloaded += self._pull_files(entity)
                self._upsert(entity)
                pulled += 1
                progress("pulled", entity=item.kind, entity_id=item.id)
            elif force:
                self.state.remember(item.id, server[item.id].version)
                self.push_entity(local[item.id])
                pushed += 1
                progress("pushed", entity=item.kind, entity_id=item.id)
        return PullOutcome(pulled=pulled, pushed=pushed, conflicts_left=left, downloaded=downloaded)

    def push_changes(self, progress: Progress = _noop) -> PullOutcome:
        """Sube lo que cambió solo en el desktop (los conflictos se resuelven con `pull`)."""
        _, _, local, _, status = self._snapshot()
        pushed = 0
        left = [i.id for i in status.items if i.state == "conflict"]
        order = {m.__name__: i for i, m in enumerate(EDITABLE)}
        for item in sorted(status.items, key=lambda i: order.get(i.kind, -1)):
            if item.state not in ("push", "local_only"):
                continue
            if item.kind == "Project":
                self.push_project()
            else:
                entity = local[item.id]
                folder = None
                if isinstance(entity, DatasetVersion):
                    folder = self.root / (entity.path or f"datasets/{entity.id}")
                self.push_entity(entity)
                for file in self._files([folder] if folder else []):
                    self.upload(file, progress)
            pushed += 1
            progress("pushed", entity=item.kind, entity_id=item.id)
        return PullOutcome(pulled=0, pushed=pushed, conflicts_left=left, downloaded=0)
