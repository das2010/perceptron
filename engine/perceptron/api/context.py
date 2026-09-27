"""Contenedor de servicios del Engine, inyectado en los routers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from fastapi import Request

from perceptron.api.jobs import JobManager
from perceptron.core.config import Settings
from perceptron.core.events import EventBus
from perceptron.domain.models import Entity, Project
from perceptron.storage.db import Database
from perceptron.storage.filesystem import ProjectFiles
from perceptron.storage.repositories import SqlRepository

E = TypeVar("E", bound=Entity)


@dataclass
class EngineContext:
    settings: Settings
    db: Database
    events: EventBus = field(default_factory=EventBus)
    _repos: dict[type[Any], SqlRepository[Any]] = field(default_factory=dict, repr=False)
    _jobs: JobManager | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.projects = SqlRepository(self.db, Project)
        self.files = ProjectFiles(self.settings.paths)

    def repo(self, model: type[E]) -> SqlRepository[E]:
        """Repositorio de cualquier entidad del dominio (cacheado)."""
        if model not in self._repos:
            self._repos[model] = SqlRepository(self.db, model)
        return self._repos[model]

    @classmethod
    def create(cls, settings: Settings) -> EngineContext:
        settings.paths.ensure()
        db = Database.for_file(settings.paths.db_file)
        db.create_all()
        return cls(settings=settings, db=db)

    @property
    def jobs(self) -> JobManager:
        if self._jobs is None:
            self._jobs = JobManager(self.events)
        return self._jobs

    def close(self) -> None:
        if self._jobs is not None:
            self._jobs.shutdown()
        self.db.dispose()


def get_context(request: Request) -> EngineContext:
    return cast(EngineContext, request.app.state.ctx)
