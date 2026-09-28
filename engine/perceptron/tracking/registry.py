"""Espejo del registro de modelos en el Model Registry de MLflow (RF-TRK-03).

El registro de Perceptron (`ModelVersion` con stages, promover, rollback, challenger) es la
fuente de verdad; MLflow lo refleja para quien use sus herramientas:
- un modelo registrado por proyecto (`perceptron-<project_id>`);
- una versión por `ModelVersion`, apuntando a los artefactos del run (`runs:/<id>/run`);
- el stage como tag (`perceptron.stage`) y el champion como alias `champion` (MLflow
  reemplazó los stages por alias).

Es best-effort: si MLflow falla, se registra en el log y el registro nativo sigue igual.
"""

from __future__ import annotations

import logging
from typing import Any

from perceptron.domain.enums import ModelStage
from perceptron.domain.models import ModelVersion

logger = logging.getLogger(__name__)

CHAMPION_ALIAS = "champion"


def registered_name(project_id: str) -> str:
    return f"perceptron-{project_id}"


class RegistryMirror:
    def __init__(self, client: Any) -> None:
        self.client = client  # MlflowClient

    def _ensure_model(self, name: str) -> None:
        from mlflow.exceptions import MlflowException

        try:
            self.client.get_registered_model(name)
        except MlflowException:
            self.client.create_registered_model(name, tags={"perceptron": "true"})

    def register(self, mv: ModelVersion, mlflow_run_id: str | None) -> str | None:
        """Crea la versión en MLflow; devuelve su número (o None si no se pudo)."""
        if not mlflow_run_id:
            return None
        name = registered_name(mv.project_id)
        try:
            self._ensure_model(name)
            version = self.client.create_model_version(
                name,
                source=f"runs:/{mlflow_run_id}/run",
                run_id=mlflow_run_id,
                tags={"perceptron.model_version": mv.id, "perceptron.stage": mv.stage.value},
            )
            return str(version.version)
        except Exception:
            logger.warning("no se pudo registrar el modelo en MLflow", exc_info=True)
            return None

    def sync_stage(self, mv: ModelVersion) -> None:
        """Refleja el stage actual (tag) y mueve el alias `champion` al de producción."""
        if not mv.mlflow_version:
            return
        name = registered_name(mv.project_id)
        try:
            self.client.set_model_version_tag(
                name, mv.mlflow_version, "perceptron.stage", mv.stage.value
            )
            if mv.stage is ModelStage.PRODUCTION:
                self.client.set_registered_model_alias(name, CHAMPION_ALIAS, mv.mlflow_version)
        except Exception:
            logger.warning("no se pudo actualizar el stage en MLflow", exc_info=True)


def mirror_for(ctx: Any) -> RegistryMirror | None:
    """Espejo con el cliente MLflow del Engine; None si el tracking no es MLflow."""
    from perceptron.services.workflow import Workflow
    from perceptron.tracking.tracker import MlflowTracker

    tracker = Workflow(ctx).tracker
    return RegistryMirror(tracker.client) if isinstance(tracker, MlflowTracker) else None
