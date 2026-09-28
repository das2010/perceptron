"""Monitoreo de modelos en uso (RF-MON-01..04, RF-MON-06).

- **Deployments** que sirven a la `ModelVersion` champion con el modelo ONNX exportado, con
  registro muestreado de predicciones y feedback con la etiqueta real.
- **Chequeo** por ventana: drift de datos (features de la firma contra el split de
  entrenamiento del champion), drift de salida del modelo y de performance (feedback contra
  el test del champion). Deja un `DriftReport` y levanta alertas según los umbrales.
- **Champion/challenger**: el challenger se compara con el champion sobre el mismo holdout
  reciente y se promueve solo si mejora; rollback al champion anterior en un paso.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from perceptron.core.config import RuntimeMode
from perceptron.core.errors import ConflictError, ValidationError
from perceptron.core.ids import IdPrefix, new_id
from perceptron.data.view import Purpose
from perceptron.domain.enums import (
    AlertKind,
    DeploymentHost,
    DeploymentStatus,
    ModelStage,
    Severity,
)
from perceptron.domain.models import (
    DatasetVersion,
    Deployment,
    DriftReport,
    Evaluation,
    ModelVersion,
    Run,
    utcnow,
)
from perceptron.monitoring.alerts import AlertService
from perceptron.monitoring.drift import data_drift, embedding_drift, numeric_drift
from perceptron.monitoring.store import PredictionStore

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext
    from perceptron.serving.runtime import InferenceModel

DRIFT_TOPIC = "drift.report"
REFERENCE_ROWS = 5000
LOWER_IS_BETTER = frozenset({"mae", "rmse", "mape", "smape", "median_abs_error", "log_loss", "ece"})
DEFAULT_MONITORING: dict[str, Any] = {
    "window": 200,  # predicciones por chequeo automático
    "min_labels": 30,  # feedback mínimo para medir performance
    "drift_alert": Severity.MEDIUM.value,  # severidad de drift que dispara alerta
    "performance_drop": 0.05,  # caída relativa de la métrica que dispara alerta
    "email": [],
}


class ChallengeResult(BaseModel):
    champion_id: str | None
    challenger_id: str
    metric: str
    higher_is_better: bool
    champion_value: float | None
    challenger_value: float
    improvement: float
    min_improvement: float
    promoted: bool
    holdout: str
    n_samples: int
    champion_metrics: dict[str, float] = Field(default_factory=dict)
    challenger_metrics: dict[str, float] = Field(default_factory=dict)


def _sev_max(*values: Severity) -> Severity:
    return max(values, key=lambda s: s.rank, default=Severity.NONE)


def primary_metric(task: str, preferred: str | None) -> str:
    if preferred:
        return preferred
    return "rmse" if task == "regression" else "roc_auc"


def score_rows(
    model: InferenceModel, rows: list[dict[str, Any]], labels: list[str]
) -> dict[str, float]:
    """Métricas del modelo ONNX sobre filas etiquetadas (misma lógica que la evaluación)."""
    from perceptron.evaluation.metrics import classification_metrics, regression_metrics

    preds = model.predict_rows(rows)
    if model.task == "regression":
        y = np.array([float(v) for v in labels])
        p = np.array([float(x.prediction) for x in preds])
        reg, _ = regression_metrics(y, p)
        return {k: v for k, v in reg.model_dump().items() if isinstance(v, float)}
    classes = [str(c) for c in (model.pipeline.classes or [])]
    index = {c: i for i, c in enumerate(classes)}
    keep = [i for i, v in enumerate(labels) if str(v) in index]
    if not keep:
        raise ValidationError("ninguna etiqueta coincide con las clases del modelo")
    y = np.array([index[str(labels[i])] for i in keep])
    proba = np.array([[preds[i].probabilities.get(c, 0.0) for c in classes] for i in keep])
    cls, _ = classification_metrics(y, proba, classes)
    return {
        k: float(v)
        for k, v in cls.model_dump().items()
        if isinstance(v, int | float) and v is not None
    }


class Monitoring:
    def __init__(self, ctx: EngineContext) -> None:
        from perceptron.services.workflow import Workflow

        self.ctx = ctx
        self.wf = Workflow(ctx)
        self.alerts = AlertService(ctx)
        self.deployments = ctx.repo(Deployment)
        self.models = ctx.repo(ModelVersion)

    # ------------------------------------------------------------------ helpers

    def store(self, dep: Deployment) -> PredictionStore:
        root = self.ctx.settings.paths.project(dep.project_id).root / "monitoring" / dep.id
        return PredictionStore(root)

    def model(self, mv: ModelVersion) -> InferenceModel:
        model: InferenceModel = self.wf.inference_model(mv.run_id)
        return model

    def champion(self, project_id: str) -> ModelVersion | None:
        prod = self.models.list(
            filters={"project_id": project_id, "stage": ModelStage.PRODUCTION.value}
        )
        return prod[0] if prod else None

    def _settings(self, dep: Deployment) -> dict[str, Any]:
        return {**DEFAULT_MONITORING, **(dep.monitoring or {})}

    # ------------------------------------------------------------------ deployments

    def deploy(
        self,
        model_version_id: str,
        *,
        name: str = "default",
        sample_rate: float = 1.0,
        key_column: str | None = None,
        monitoring: dict[str, Any] | None = None,
    ) -> Deployment:
        mv = self.models.get(model_version_id)
        model = self.model(mv)  # exige el export ONNX
        if model.kind != "tabular":
            raise ValidationError("por ahora los deployments monitoreados son tabulares")
        dep_id = new_id(IdPrefix.DEPLOYMENT)
        server = self.ctx.settings.mode is RuntimeMode.SERVER
        dep = Deployment(
            id=dep_id,
            project_id=mv.project_id,
            name=name,
            model_version_id=mv.id,
            endpoint=f"/api/v1/deployments/{dep_id}/predict",
            host=DeploymentHost.SERVER if server else DeploymentHost.DESKTOP,
            sample_rate=sample_rate,
            key_column=key_column,
            monitoring={**DEFAULT_MONITORING, **(monitoring or {}), "checked_at_count": 0},
        )
        self.deployments.add(dep)
        if self.champion(mv.project_id) is None:
            self.promote(mv.id, reason="primer deployment")
        return self.deployments.get(dep.id)

    def get(self, deployment_id: str) -> Deployment:
        return self.deployments.get(deployment_id)

    def predict(self, deployment_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dep = self.get(deployment_id)
        if dep.status is not DeploymentStatus.ACTIVE:
            raise ConflictError("el deployment está detenido")
        mv = self.models.get(dep.model_version_id)
        model = self.model(mv)
        preds = [p.to_dict() for p in model.predict_rows(rows)]
        ids = [new_id(IdPrefix.PREDICTION) for _ in rows]
        rng = random.Random()  # noqa: S311 - muestreo, no criptografía
        chosen = [i for i in range(len(rows)) if rng.random() < dep.sample_rate]
        if chosen:
            cols = model.required_columns
            self.store(dep).log_predictions(
                mv.id,
                [{c: rows[i].get(c) for c in cols} for i in chosen],
                [preds[i] for i in chosen],
                [
                    str(rows[i].get(dep.key_column))
                    if dep.key_column and rows[i].get(dep.key_column) is not None
                    else None
                    for i in chosen
                ],
                [ids[i] for i in chosen],
            )
            self._maybe_check(dep)
        return [{"prediction_id": pid, **p} for pid, p in zip(ids, preds, strict=True)]

    def _maybe_check(self, dep: Deployment) -> None:
        cfg = self._settings(dep)
        count = self.store(dep).count()
        if count - int(cfg.get("checked_at_count", 0)) < int(cfg["window"]):
            return
        self.ctx.jobs.submit(
            "drift_check",
            lambda _job: self.check(dep.id).model_dump(mode="json"),
            refs={"deployment_id": dep.id, "project_id": dep.project_id},
        )

    def feedback(self, deployment_id: str, items: list[dict[str, Any]]) -> int:
        dep = self.get(deployment_id)
        for item in items:
            if not item.get("prediction_id") and item.get("key") is None:
                raise ValidationError("cada feedback necesita prediction_id o key")
            if "label" not in item:
                raise ValidationError("falta la etiqueta real (label)")
        self.store(dep).log_feedback(items)
        return len(items)

    # ------------------------------------------------------------------ chequeos

    def reference(self, mv: ModelVersion) -> pl.DataFrame:
        run = self.ctx.repo(Run).get(mv.run_id)
        dv = self.ctx.repo(DatasetVersion).get(run.dataset_version_id)
        df = self.wf.view(dv).scan("train").collect()
        return df.sample(n=REFERENCE_ROWS, seed=0) if df.height > REFERENCE_ROWS else df

    def check(self, deployment_id: str, *, last: int | None = None) -> DriftReport:
        dep = self.get(deployment_id)
        cfg = self._settings(dep)
        store = self.store(dep)
        preds = store.predictions(last=last or int(cfg["window"]))
        if preds.is_empty():
            raise ValidationError("todavía no hay predicciones registradas")
        mv = self.models.get(dep.model_version_id)
        model = self.model(mv)
        sig = mv.signature.get("inputs", {})
        numeric = list(sig.get("numeric_features") or [])
        categorical = list(sig.get("categorical_features") or [])
        ref = self.reference(mv)
        cur = store.features(preds)
        data = data_drift(ref, cur, numeric, categorical)
        output = self._output_drift(model, ref, preds)
        perf = self.performance(dep, mv=mv)
        perf_sev = Severity(perf["severity"]) if perf else Severity.NONE
        severity = _sev_max(data.severity, perf_sev)
        report = DriftReport(
            deployment_id=dep.id,
            window_start=min(preds["ts"].to_list()),
            window_end=max(preds["ts"].to_list()),
            metrics={
                "model_version_id": mv.id,
                "data": data.model_dump(mode="json"),
                "output": output,
                "performance": perf,
            },
            severity=severity,
        )
        actions: list[str] = []
        threshold = Severity(cfg["drift_alert"])
        if data.severity.rank >= threshold.rank:
            top = sorted(data.drifted, key=lambda f: f.severity.rank, reverse=True)[:5]
            alert = self.alerts.raise_alert(
                project_id=dep.project_id,
                kind=AlertKind.DATA_DRIFT,
                severity=data.severity,
                title=f"Drift de datos en {dep.name}",
                message="Features con drift: " + ", ".join(f.feature for f in top),
                deployment=dep,
                details={"report_id": report.id, "features": [f.feature for f in top]},
            )
            if alert:
                actions.append(f"alert:{alert.id}")
        if perf and perf_sev.rank >= Severity.MEDIUM.rank:
            alert = self.alerts.raise_alert(
                project_id=dep.project_id,
                kind=AlertKind.PERFORMANCE,
                severity=perf_sev,
                title=f"Cae la performance de {dep.name}",
                message=(
                    f"{perf['metric']}: {perf['current']:.4f} (test del champion "
                    f"{perf['baseline']:.4f}) sobre {perf['n_labeled']} casos etiquetados"
                ),
                deployment=dep,
                details={"report_id": report.id, **perf},
            )
            if alert:
                actions.append(f"alert:{alert.id}")
        report.action = ",".join(actions) or None
        self.ctx.repo(DriftReport).add(report)
        fresh = self.get(dep.id)
        self.deployments.update(
            fresh.model_copy(
                update={"monitoring": {**fresh.monitoring, "checked_at_count": store.count()}}
            )
        )
        self.ctx.events.publish(
            DRIFT_TOPIC, report_id=report.id, deployment_id=dep.id, severity=severity.value
        )
        return report

    def _output_drift(
        self, model: InferenceModel, ref: pl.DataFrame, preds: pl.DataFrame
    ) -> dict[str, Any]:
        """Drift en el espacio de salida del modelo (probabilidades o valor predicho)."""
        sample = ref.head(1000)
        cols = model.required_columns
        ref_preds = model.predict_rows(
            sample.select([c for c in cols if c in sample.columns]).to_dicts()
        )
        if model.task == "regression":
            ref_v = np.array([float(p.prediction) for p in ref_preds])
            cur_v = preds["prediction"].cast(pl.Float64, strict=False).to_numpy()
            return numeric_drift("prediction", ref_v, cur_v.astype(float)).model_dump(mode="json")
        classes = [str(c) for c in (model.pipeline.classes or [])]
        ref_m = np.array([[p.probabilities.get(c, 0.0) for c in classes] for p in ref_preds])
        cur_m = np.array(
            [[json.loads(s).get(c, 0.0) for c in classes] for s in preds["probabilities"].to_list()]
        )
        if len(cur_m) < 10:
            return {"severity": Severity.NONE.value, "note": "pocas predicciones"}
        return embedding_drift(ref_m, cur_m).model_dump(mode="json")

    def performance(
        self, dep: Deployment, *, mv: ModelVersion | None = None
    ) -> dict[str, Any] | None:
        """Drift de performance (RF-MON-03): feedback reciente contra el test del champion."""
        cfg = self._settings(dep)
        labeled = self.store(dep).labeled(last=int(cfg["window"]) * 5)
        if labeled.height < int(cfg["min_labels"]):
            return None
        mv = mv or self.models.get(dep.model_version_id)
        labeled = labeled.filter(pl.col("model_version_id") == mv.id)
        if labeled.height < int(cfg["min_labels"]):
            return None
        model = self.model(mv)
        rows = self.store(dep).features(labeled).to_dicts()
        current = score_rows(model, rows, labeled["label"].to_list())
        project = self.ctx.projects.get(dep.project_id)
        metric = primary_metric(model.task, project.target_metric)
        baseline = mv.model_card.get("test_metrics", {}).get(metric)
        value = current.get(metric)
        if baseline is None or value is None:
            return {
                "metric": metric,
                "current": value,
                "baseline": baseline,
                "n_labeled": labeled.height,
                "severity": Severity.NONE.value,
                "metrics": current,
            }
        lower = metric in LOWER_IS_BETTER
        drop = (
            (value - baseline) / abs(baseline or 1)
            if lower
            else (baseline - value) / abs(baseline or 1)
        )
        limit = float(cfg["performance_drop"])
        sev = (
            Severity.HIGH
            if drop >= 4 * limit
            else Severity.MEDIUM
            if drop >= 2 * limit
            else Severity.LOW
            if drop >= limit
            else Severity.NONE
        )
        return {
            "metric": metric,
            "current": value,
            "baseline": float(baseline),
            "relative_drop": round(drop, 4),
            "n_labeled": labeled.height,
            "severity": sev.value,
            "metrics": current,
        }

    # ------------------------------------------------------------------ champion/challenger

    def promote(self, model_version_id: str, *, reason: str = "manual") -> ModelVersion:
        mv = self.models.get(model_version_id)
        now = utcnow()
        current = self.champion(mv.project_id)
        if current is not None and current.id != mv.id:
            self.models.update(
                current.model_copy(update={"stage": ModelStage.ARCHIVED, "retired_at": now})
            )
        mv = self.models.update(
            mv.model_copy(update={"stage": ModelStage.PRODUCTION, "promoted_at": now})
        )
        for dep in self.deployments.list(filters={"project_id": mv.project_id}, limit=500):
            if dep.follow_champion and dep.model_version_id != mv.id:
                self.deployments.update(dep.model_copy(update={"model_version_id": mv.id}))
        self.ctx.events.publish(
            "model.promoted", model_version_id=mv.id, project_id=mv.project_id, reason=reason
        )
        return mv

    def rollback(self, project_id: str) -> ModelVersion:
        """Vuelve al champion anterior (el último retirado)."""
        current = self.champion(project_id)
        retired = [
            m
            for m in self.models.list(filters={"project_id": project_id}, limit=1000)
            if m.retired_at is not None and (current is None or m.id != current.id)
        ]
        if not retired:
            raise ConflictError("no hay un champion anterior al que volver")
        previous = max(retired, key=lambda m: m.retired_at or utcnow())
        return self.promote(previous.id, reason="rollback")

    def _holdout(
        self, project_id: str, holdout: str
    ) -> tuple[list[dict[str, Any]], list[str], str]:
        if holdout == "feedback":
            frames = []
            for dep in self.deployments.list(filters={"project_id": project_id}, limit=500):
                store = self.store(dep)
                labeled = store.labeled()
                if labeled.height:
                    frames.append(
                        store.features(labeled).with_columns(labeled["label"].alias("__label"))
                    )
            if not frames:
                raise ValidationError("no hay feedback etiquetado para comparar")
            df = pl.concat(frames, how="diagonal_relaxed")
            return df.drop("__label").to_dicts(), df["__label"].cast(pl.Utf8).to_list(), "feedback"
        dv = self.ctx.repo(DatasetVersion).get(holdout)
        if dv.project_id != project_id or not dv.target:
            raise ValidationError("el holdout debe ser una versión etiquetada del proyecto")
        df = self.wf.view(dv).scan("test", purpose=Purpose.FINAL_EVALUATION).collect()
        labels = df[dv.target].cast(pl.Utf8).to_list()
        return df.to_dicts(), labels, f"dataset:{dv.id}"

    def challenge(
        self,
        challenger_id: str,
        *,
        holdout: str = "feedback",
        metric: str | None = None,
        min_improvement: float = 0.0,
        promote: bool = True,
    ) -> ChallengeResult:
        """Compara con el champion sobre el mismo holdout; promueve solo si mejora."""
        challenger = self.models.get(challenger_id)
        champion = self.champion(challenger.project_id)
        rows, labels, source = self._holdout(challenger.project_id, holdout)
        ch_model = self.model(challenger)
        project = self.ctx.projects.get(challenger.project_id)
        name = primary_metric(ch_model.task, metric or project.target_metric)
        ch_metrics = score_rows(ch_model, rows, labels)
        if name not in ch_metrics:
            raise ValidationError(f"la métrica {name} no está disponible para esta tarea")
        cp_metrics = score_rows(self.model(champion), rows, labels) if champion else {}
        higher = name not in LOWER_IS_BETTER
        ch_value = ch_metrics[name]
        cp_value = cp_metrics.get(name)
        if cp_value is None:
            improvement = float("inf")
        else:
            improvement = (ch_value - cp_value) if higher else (cp_value - ch_value)
        promoted = False
        if promote and (champion is None or improvement > min_improvement):
            self.promote(challenger.id, reason=f"challenger: {name} {improvement:+.4f}")
            promoted = True
        for mv, values in ((challenger, ch_metrics), (champion, cp_metrics)):
            if mv is not None:
                self.ctx.repo(Evaluation).add(
                    Evaluation(
                        run_id=mv.run_id,
                        split="holdout",
                        metrics=values,
                        dataset_version_id=holdout if holdout != "feedback" else None,
                        artifacts={"holdout": source, "challenge_of": challenger.id},
                    )
                )
        return ChallengeResult(
            champion_id=champion.id if champion else None,
            challenger_id=challenger.id,
            metric=name,
            higher_is_better=higher,
            champion_value=cp_value,
            challenger_value=ch_value,
            improvement=improvement if improvement != float("inf") else 0.0,
            min_improvement=min_improvement,
            promoted=promoted,
            holdout=source,
            n_samples=len(rows),
            champion_metrics=cp_metrics,
            challenger_metrics=ch_metrics,
        )
