"""Contenedor de servicios del Engine, inyectado en los routers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from fastapi import Request

from perceptron.api.jobs import JobManager
from perceptron.core.config import RuntimeMode, Settings
from perceptron.core.events import EventBus
from perceptron.domain.models import Entity, Project, Study
from perceptron.storage.db import Database
from perceptron.storage.filesystem import ProjectFiles
from perceptron.storage.repositories import SqlRepository

if TYPE_CHECKING:
    from perceptron.api.jobs import Job
    from perceptron.core.telemetry import Telemetry
    from perceptron.llm.gateway import Gateway
    from perceptron.remote.client import RemoteRegistry
    from perceptron.services.studies import StudyLauncher
    from perceptron.tracking.mlflow_ui import MlflowUi

E = TypeVar("E", bound=Entity)


@dataclass
class EngineContext:
    settings: Settings
    db: Database
    events: EventBus = field(default_factory=EventBus)
    _repos: dict[type[Any], SqlRepository[Any]] = field(default_factory=dict, repr=False)
    _jobs: JobManager | None = field(default=None, repr=False)
    _llm: Any = field(default=None, repr=False)
    study_launcher: StudyLauncher | None = field(default=None, repr=False)
    _closers: list[Callable[[], None]] = field(default_factory=list, repr=False)
    # Tests: cliente HTTP alternativo para hablar con un Team Server (Capa 5c).
    remote_http: Callable[[], Any] | None = field(default=None, repr=False)
    _scheduler: Any = field(default=None, repr=False)
    _telemetry: Any = field(default=None, repr=False)
    _mlflow_ui: Any = field(default=None, repr=False)

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
        db = (
            Database(settings.database_url.get_secret_value())
            if settings.database_url
            else Database.for_file(settings.paths.db_file)
        )
        db.create_all()  # idempotente; en el Team Server el esquema lo migra Alembic antes
        from perceptron.licensing import features
        from perceptron.licensing.signed import license_provider

        features.provider = license_provider(settings)
        from perceptron.catalog.weights_cache import configure

        configure(settings.models_cache or settings.paths.models_cache_dir)
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

    def launch_study(self, study: Study) -> Job:
        """Lanza un estudio: en un hilo (desktop) o en la cola del Team Server (Capa 5b)."""
        from perceptron.services.studies import launch_local

        return (self.study_launcher or launch_local)(self, study)

    @property
    def remotes(self) -> RemoteRegistry:
        """Servidores de equipo configurados en este desktop (Capa 5c)."""
        from perceptron.remote.client import RemoteRegistry

        return RemoteRegistry(self.settings.workspace_dir, self.llm.secrets)

    def remote_http_kwargs(self) -> dict[str, Any]:
        return {"http": self.remote_http} if self.remote_http is not None else {}

    def use_llm(self, gateway: Gateway) -> None:
        self._llm = gateway

    @property
    def mlflow_ui(self) -> MlflowUi:
        """Enlace opcional a la UI de MLflow (RF-TRK-02); en el desktop se levanta a pedido."""
        if self._mlflow_ui is None:
            from perceptron.tracking.mlflow_ui import MlflowUi

            root = self.settings.paths.mlflow_dir
            uri = (
                self.settings.tracking_uri
                or f"sqlite:///{(root / 'mlflow.db').resolve().as_posix()}"
            )
            self._mlflow_ui = MlflowUi(
                uri,
                root / "artifacts",
                public_url=self.settings.mlflow_ui_url,
                can_launch=self.settings.mode is not RuntimeMode.SERVER,
            )
            self.on_close(self._mlflow_ui.close)
        return cast("MlflowUi", self._mlflow_ui)

    @property
    def telemetry(self) -> Telemetry:
        """Telemetría opt-in (D7): apagada salvo consentimiento y endpoint configurado."""
        if self._telemetry is None:
            from perceptron.core.telemetry import Telemetry

            self._telemetry = Telemetry(
                self.settings.workspace_dir / "telemetry.json", self.settings.telemetry.endpoint
            )
        return cast("Telemetry", self._telemetry)

    def start_scheduler(self) -> None:
        """Disparadores de reentrenamiento y sondeo de fuentes (Capa 6), una vez por contexto."""
        if self._scheduler is not None or not self.settings.monitoring.scheduler:
            return
        from perceptron.monitoring.scheduler import Scheduler

        scheduler = Scheduler(self, self.settings.monitoring.interval_s)
        scheduler.start()
        self._scheduler = scheduler
        self.on_close(scheduler.stop)

    def on_close(self, callback: Callable[[], None]) -> None:
        """Libera recursos de extensiones (p. ej. el relay de eventos del Team Server)."""
        self._closers.append(callback)

    def close(self) -> None:
        for callback in reversed(self._closers):
            callback()
        if self._jobs is not None:
            self._jobs.shutdown()
        self.db.dispose()


def get_context(request: Request) -> EngineContext:
    return cast(EngineContext, request.app.state.ctx)
