"""Contenedor de servicios del Engine, inyectado en los routers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from fastapi import Request

from perceptron.core.config import Settings
from perceptron.core.events import EventBus
from perceptron.domain.models import Project
from perceptron.storage.db import Database
from perceptron.storage.filesystem import ProjectFiles
from perceptron.storage.repositories import SqlRepository


@dataclass
class EngineContext:
    settings: Settings
    db: Database
    events: EventBus = field(default_factory=EventBus)

    def __post_init__(self) -> None:
        self.projects = SqlRepository(self.db, Project)
        self.files = ProjectFiles(self.settings.paths)

    @classmethod
    def create(cls, settings: Settings) -> EngineContext:
        settings.paths.ensure()
        db = Database.for_file(settings.paths.db_file)
        db.create_all()
        return cls(settings=settings, db=db)

    def close(self) -> None:
        self.db.dispose()


def get_context(request: Request) -> EngineContext:
    return cast(EngineContext, request.app.state.ctx)
