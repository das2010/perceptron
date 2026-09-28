"""Monitoreo y ciclo de vida de modelos (SPEC §10, RF-MON-01..04, RF-MON-06)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.domain.enums import AlertStatus, DeploymentStatus, Severity
from perceptron.domain.models import Alert, Deployment, DriftReport, ModelVersion
from perceptron.monitoring.alerts import webhook_secret
from perceptron.monitoring.service import ChallengeResult, Monitoring

router = APIRouter(tags=["monitoring"])
Ctx = Annotated[EngineContext, Depends(get_context)]


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window: int = Field(default=200, ge=10, le=100_000, description="Predicciones por chequeo")
    min_labels: int = Field(default=30, ge=5, description="Feedback mínimo para performance")
    drift_alert: Severity = Field(default=Severity.MEDIUM, description="Severidad que alerta")
    performance_drop: float = Field(default=0.05, gt=0, le=1, description="Caída relativa")
    email: list[str] = Field(default_factory=list, description="Destinatarios de alertas")


class DeploymentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="default", min_length=1, max_length=100)
    sample_rate: float = Field(default=1.0, ge=0, le=1)
    key_column: str | None = None
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    webhook_url: str | None = Field(
        default=None, pattern=r"^https?://", description="Solo de escritura: va al keychain"
    )


class DeploymentPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: DeploymentStatus | None = None
    sample_rate: float | None = Field(default=None, ge=0, le=1)
    monitoring: MonitoringConfig | None = None
    webhook_url: str | None = Field(default=None, pattern=r"^(https?://.*)?$")


class PredictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=5000)


class Predictions(BaseModel):
    predictions: list[dict[str, Any]]


class FeedbackItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str | None = None
    key: str | None = None
    label: str | float | int


class FeedbackBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[FeedbackItem] = Field(min_length=1, max_length=10_000)


class FeedbackResult(BaseModel):
    received: int


class ChallengeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    holdout: str = Field(default="feedback", description="`feedback` o un dataset_version_id")
    metric: str | None = None
    min_improvement: float = Field(default=0.0, ge=0)
    promote: bool = True


def _set_webhook(ctx: EngineContext, dep_id: str, url: str | None) -> None:
    if url is None:
        return
    if url:
        ctx.llm.secrets.set(webhook_secret(dep_id), url)
    else:
        ctx.llm.secrets.delete(webhook_secret(dep_id))


# ------------------------------------------------------------------ deployments


@router.post(
    "/models/{model_version_id}/deployments",
    status_code=status.HTTP_201_CREATED,
    operation_id="createDeployment",
)
def create_deployment(model_version_id: str, body: DeploymentCreate, ctx: Ctx) -> Deployment:
    dep = Monitoring(ctx).deploy(
        model_version_id,
        name=body.name,
        sample_rate=body.sample_rate,
        key_column=body.key_column,
        monitoring=body.monitoring.model_dump(mode="json"),
    )
    _set_webhook(ctx, dep.id, body.webhook_url)
    return dep


@router.get("/projects/{project_id}/deployments", operation_id="listDeployments")
def list_deployments(project_id: str, ctx: Ctx) -> list[Deployment]:
    ctx.projects.get(project_id)
    return list(ctx.repo(Deployment).list(filters={"project_id": project_id}))


@router.get("/deployments/{deployment_id}", operation_id="getDeployment")
def get_deployment(deployment_id: str, ctx: Ctx) -> Deployment:
    return ctx.repo(Deployment).get(deployment_id)


@router.patch("/deployments/{deployment_id}", operation_id="updateDeployment")
def update_deployment(deployment_id: str, body: DeploymentPatch, ctx: Ctx) -> Deployment:
    repo = ctx.repo(Deployment)
    dep = repo.get(deployment_id)
    changes: dict[str, Any] = {}
    if body.status is not None:
        changes["status"] = body.status
    if body.sample_rate is not None:
        changes["sample_rate"] = body.sample_rate
    if body.monitoring is not None:
        changes["monitoring"] = {**dep.monitoring, **body.monitoring.model_dump(mode="json")}
    _set_webhook(ctx, dep.id, body.webhook_url)
    return repo.update(dep.model_copy(update=changes)) if changes else dep


@router.post("/deployments/{deployment_id}/predict", operation_id="predictDeployment")
def predict(deployment_id: str, body: PredictBody, ctx: Ctx) -> Predictions:
    return Predictions(predictions=Monitoring(ctx).predict(deployment_id, body.rows))


@router.post("/deployments/{deployment_id}/feedback", operation_id="sendFeedback")
def feedback(deployment_id: str, body: FeedbackBody, ctx: Ctx) -> FeedbackResult:
    items = [i.model_dump() for i in body.items]
    return FeedbackResult(received=Monitoring(ctx).feedback(deployment_id, items))


@router.get("/deployments/{deployment_id}/predictions", operation_id="listPredictions")
def predictions(
    deployment_id: str, ctx: Ctx, limit: Annotated[int, Query(ge=1, le=1000)] = 100
) -> list[dict[str, Any]]:
    mon = Monitoring(ctx)
    df = mon.store(mon.get(deployment_id)).predictions(last=limit)
    return df.reverse().to_dicts() if df.height else []


@router.post("/deployments/{deployment_id}/check", operation_id="checkDeployment")
def check(
    deployment_id: str, ctx: Ctx, last: Annotated[int | None, Query(ge=10)] = None
) -> DriftReport:
    return Monitoring(ctx).check(deployment_id, last=last)


@router.get("/deployments/{deployment_id}/drift", operation_id="listDriftReports")
def drift(deployment_id: str, ctx: Ctx) -> list[DriftReport]:
    ctx.repo(Deployment).get(deployment_id)
    return list(ctx.repo(DriftReport).list(filters={"deployment_id": deployment_id}, limit=200))


@router.get("/deployments/{deployment_id}/performance", operation_id="getPerformance")
def performance(deployment_id: str, ctx: Ctx) -> dict[str, Any] | None:
    mon = Monitoring(ctx)
    return mon.performance(mon.get(deployment_id))


# ------------------------------------------------------------------ champion/challenger


@router.post("/models/{model_version_id}/promote", operation_id="promoteModel")
def promote(model_version_id: str, ctx: Ctx) -> ModelVersion:
    return Monitoring(ctx).promote(model_version_id)


@router.post("/projects/{project_id}/models/rollback", operation_id="rollbackModel")
def rollback(project_id: str, ctx: Ctx) -> ModelVersion:
    ctx.projects.get(project_id)
    return Monitoring(ctx).rollback(project_id)


@router.post("/models/{model_version_id}/challenge", operation_id="challengeModel")
def challenge(model_version_id: str, body: ChallengeBody, ctx: Ctx) -> ChallengeResult:
    return Monitoring(ctx).challenge(
        model_version_id,
        holdout=body.holdout,
        metric=body.metric,
        min_improvement=body.min_improvement,
        promote=body.promote,
    )


# ------------------------------------------------------------------ alertas


@router.get("/projects/{project_id}/alerts", operation_id="listAlerts")
def list_alerts(
    project_id: str,
    ctx: Ctx,
    status_filter: Annotated[AlertStatus | None, Query(alias="status")] = None,
) -> list[Alert]:
    ctx.projects.get(project_id)
    filters: dict[str, Any] = {"project_id": project_id}
    if status_filter is not None:
        filters["status"] = status_filter.value
    return list(ctx.repo(Alert).list(filters=filters, limit=500))


def _set_alert(ctx: EngineContext, alert_id: str, value: AlertStatus) -> Alert:
    repo = ctx.repo(Alert)
    return repo.update(repo.get(alert_id).model_copy(update={"status": value}))


@router.post("/alerts/{alert_id}/acknowledge", operation_id="acknowledgeAlert")
def acknowledge(alert_id: str, ctx: Ctx) -> Alert:
    return _set_alert(ctx, alert_id, AlertStatus.ACKNOWLEDGED)


@router.post("/alerts/{alert_id}/resolve", operation_id="resolveAlert")
def resolve(alert_id: str, ctx: Ctx) -> Alert:
    return _set_alert(ctx, alert_id, AlertStatus.RESOLVED)
