"""Contenedor de servicios del Engine, inyectado en los routers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from fastapi import Request

from perceptron.api.jobs import JobManager
from perceptron.core.config import Settings
from perceptron.core.events import EventBus
from perceptron.domain.models import Entity, Project
from perceptron.storage.db import Database
from perceptron.storage.filesystem import ProjectFiles
from perceptron.storage.repositories import SqlRepository

if TYPE_CHECKING:
    from perceptron.llm.gateway import Gateway

E = TypeVar("E", bound=Entity)


@dataclass
class EngineContext:
    settings: Settings
    db: Database
    events: EventBus = field(default_factory=EventBus)
    _repos: dict[type[Any], SqlRepository[Any]] = field(default_factory=dict, repr=False)
    _jobs: JobManager | None = field(default=None, repr=False)
    _llm: Any = field(default=None, repr=False)

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

    @property
    def llm(self) -> Gateway:
        """LLM Gateway (Capa 2). Se crea al primer uso; en tests se reemplaza con `use_llm`."""
        if self._llm is None:
            from perceptron.llm.gateway import Gateway

            self._llm = Gateway.from_settings(self.settings, self.db)
        return cast("Gateway", self._llm)

    def use_llm(self, gateway: Gateway) -> None:
        self._llm = gateway

    def close(self) -> None:
        if self._jobs is not None:
            self._jobs.shutdown()
        self.db.dispose()


def get_context(request: Request) -> EngineContext:
    return cast(EngineContext, request.app.state.ctx)
