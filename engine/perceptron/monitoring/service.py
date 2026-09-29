"""Monitoreo de modelos en uso (RF-MON-01..04, RF-MON-06).

- **Deployments** que sirven a la `ModelVersion` champion con el modelo ONNX exportado, con
  registro muestreado de predicciones y feedback con la etiqueta real.
- **Chequeo** por ventana: drift de datos (features de la firma contra el split de
  entrenamiento del champion), drift de salida del modelo y de performance (feedback contra
  el test del champion). Deja un `DriftReport` y levanta alertas según los umbrales.
- **Embeddings internos** (RF-MON-02): en imagen, texto y audio (y también en tabular) se
  registra el embedding de la cabeza del modelo en cada predicción muestreada, y se compara con
  el de una muestra de train (MMD, centroides y clasificador de dominio). De imágenes y audios
  no se guarda el archivo, solo el embedding.
- **Champion/challenger**: el challenger se compara con el champion sobre el mismo holdout
  reciente y se promueve solo si mejora; rollback al champion anterior en un paso.
"""

from __future__ import annotations

import io
import json
import random
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
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
from perceptron.tracking.registry import mirror_for

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext
    from perceptron.serving.runtime import InferenceModel

DRIFT_TOPIC = "drift.report"
REFERENCE_ROWS = 5000
REFERENCE_FILES = 300  # imágenes/audios de train para la referencia de embeddings
REFERENCE_BATCH = 16  # archivos por pasada del modelo al armar la referencia
MONITORED_KINDS = ("tabular", "tokens", "image", "spectrogram")
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

    if model.kind == "tokens":
        col = model.text_column or ""
        preds = model.predict_texts([str(r.get(col) or "") for r in rows])
    elif model.kind == "tabular":
        preds = model.predict_rows(rows)
    else:
        raise ValidationError("champion/challenger compara modelos de tabla o texto")
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


def _image_feed(model: InferenceModel, paths: list[Path]) -> dict[str, np.ndarray]:
    """Abre, transforma y cierra cada imagen del lote (no quedan decodificadas en memoria)."""
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.convert("RGB"))
    return model.feed_images(images)


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
        if model.kind not in MONITORED_KINDS:
            raise ValidationError(f"no hay monitoreo para modelos de entrada {model.kind}")
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
        dep = self._active(deployment_id)
        mv = self.models.get(dep.model_version_id)
        model = self.model(mv)
        if not rows:
            return []
        if model.kind == "tokens":
            col = model.text_column or ""
            if any(col not in r for r in rows):
                raise ValidationError(f"cada fila necesita la columna de texto «{col}»")
            feed = model.feed_texts([str(r[col] or "") for r in rows])
            cols = [col]
        elif model.kind == "tabular":
            feed, cols = model.feed_rows(rows), model.required_columns
        else:
            raise ValidationError("este modelo recibe archivos: usá /predict/file")
        return self._log(dep, mv, model, feed, rows, cols)

    def predict_files(
        self, deployment_id: str, files: list[tuple[str, bytes]]
    ) -> list[dict[str, Any]]:
        """Imagen o audio (RF-MON-02): se registran predicción y embedding, no el archivo."""
        dep = self._active(deployment_id)
        mv = self.models.get(dep.model_version_id)
        model = self.model(mv)
        if not files:
            return []
        if model.kind == "image":
            from PIL import Image, UnidentifiedImageError

            try:
                feed = model.feed_images([Image.open(io.BytesIO(data)) for _, data in files])
            except UnidentifiedImageError as e:
                raise ValidationError("no se pudo leer la imagen") from e
        elif model.kind == "spectrogram":
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for i, (name, data) in enumerate(files):
                    path = Path(tmp) / f"{i:04d}{Path(name).suffix.lower() or '.wav'}"
                    path.write_bytes(data)
                    paths.append(path)
                from perceptron.serving.runtime import InputError

                try:
                    feed = model.feed_audio(paths)
                except InputError as e:
                    raise ValidationError(str(e)) from e
        else:
            raise ValidationError("este modelo recibe filas: usá /predict")
        rows = [{"file_name": name} for name, _ in files]
        return self._log(dep, mv, model, feed, rows, [])

    def _active(self, deployment_id: str) -> Deployment:
        dep = self.get(deployment_id)
        if dep.status is not DeploymentStatus.ACTIVE:
            raise ConflictError("el deployment está detenido")
        return dep

    def _log(
        self,
        dep: Deployment,
        mv: ModelVersion,
        model: InferenceModel,
        feed: dict[str, np.ndarray],
        rows: list[dict[str, Any]],
        cols: list[str],
    ) -> list[dict[str, Any]]:
        results, emb = model.infer(feed, embeddings=True)
        preds = [p.to_dict() for p in results]
        ids = [new_id(IdPrefix.PREDICTION) for _ in rows]
        rng = random.Random()  # noqa: S311 - muestreo, no criptografía
        chosen = [i for i in range(len(rows)) if rng.random() < dep.sample_rate]
        if chosen:
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
                embeddings=[emb[i].tolist() for i in chosen] if emb is not None else None,
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

    def reference(self, mv: ModelVersion, split: str = "train") -> pl.DataFrame:
        """Muestra de referencia. Features: train. Salidas y embeddings del modelo: validación,
        porque sobre lo que ya vio el modelo está más seguro que sobre datos nuevos y eso solo
        parecería drift."""
        run = self.ctx.repo(Run).get(mv.run_id)
        dv = self.ctx.repo(DatasetVersion).get(run.dataset_version_id)
        df = self.wf.view(dv).scan(split).collect()
        if df.is_empty() and split != "train":
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
        ref_out = self._reference_outputs(dep, mv, model, self.reference(mv, "val"))
        output = self._output_drift(model, ref_out, preds)
        embedding = self._embedding_drift(ref_out, preds)
        emb_sev = Severity(embedding["severity"]) if embedding else Severity.NONE
        perf = self.performance(dep, mv=mv)
        perf_sev = Severity(perf["severity"]) if perf else Severity.NONE
        severity = _sev_max(data.severity, emb_sev, perf_sev)
        report = DriftReport(
            deployment_id=dep.id,
            window_start=min(preds["ts"].to_list()),
            window_end=max(preds["ts"].to_list()),
            metrics={
                "model_version_id": mv.id,
                "data": data.model_dump(mode="json"),
                "output": output,
                "embedding": embedding,
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
        elif embedding and emb_sev.rank >= threshold.rank:
            alert = self.alerts.raise_alert(
                project_id=dep.project_id,
                kind=AlertKind.DATA_DRIFT,
                severity=emb_sev,
                title=f"Drift de datos en {dep.name}",
                message=(
                    "Los embeddings internos del modelo se alejan de los de entrenamiento "
                    f"(AUC de dominio {embedding['domain_auc']:.2f})"
                ),
                deployment=dep,
                details={"report_id": report.id, "embedding": embedding},
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

    def _reference_outputs(
        self, dep: Deployment, mv: ModelVersion, model: InferenceModel, ref: pl.DataFrame
    ) -> dict[str, np.ndarray]:
        """Salidas y embeddings del modelo sobre la muestra de train (caché por versión)."""
        cache = self.store(dep).root / f"reference-{mv.id}-val.npz"
        if cache.is_file():
            with np.load(cache) as z:
                return {k: z[k] for k in z.files}
        # Por lotes chicos: 300 imágenes de una vez (decodificadas + activaciones de la red)
        # son varios GB y tiraban el proceso del servidor por falta de memoria.
        feeds: list[Callable[[], dict[str, np.ndarray]]] = []
        if model.kind == "tabular":
            sample = ref.head(1000)
            cols = [c for c in model.required_columns if c in sample.columns]
            rows = sample.select(cols).to_dicts()
            for i in range(0, len(rows), REFERENCE_BATCH * 8):
                feeds.append(partial(model.feed_rows, rows[i : i + REFERENCE_BATCH * 8]))
        elif model.kind == "tokens":
            col = model.text_column or ""
            texts = [str(v or "") for v in ref.head(1000)[col].to_list()]
            for i in range(0, len(texts), REFERENCE_BATCH * 4):
                feeds.append(partial(model.feed_texts, texts[i : i + REFERENCE_BATCH * 4]))
        else:
            run = self.ctx.repo(Run).get(mv.run_id)
            view = self.wf.view(self.ctx.repo(DatasetVersion).get(run.dataset_version_id))
            sample = ref.filter(~pl.col("corrupt")) if "corrupt" in ref.columns else ref
            paths = [view.files_dir / str(p) for p in sample.head(REFERENCE_FILES)["path"]]
            for i in range(0, len(paths), REFERENCE_BATCH):
                chunk = paths[i : i + REFERENCE_BATCH]
                if model.kind == "image":
                    feeds.append(partial(_image_feed, model, chunk))
                else:
                    feeds.append(partial(model.feed_audio, chunk))
        preds: list[Any] = []
        vectors: list[np.ndarray] = []
        has_emb = True
        for make in feeds:
            batch_preds, emb = model.infer(make(), embeddings=True)
            preds.extend(batch_preds)
            if emb is None:
                has_emb = False
            else:
                vectors.append(emb)
        emb = np.concatenate(vectors) if has_emb and vectors else None
        out: dict[str, np.ndarray] = {}
        if model.task == "regression":
            out["values"] = np.array([float(p.prediction) for p in preds])
        else:
            classes = [str(c) for c in (model.pipeline.classes or [])]
            out["proba"] = np.array([[p.probabilities.get(c, 0.0) for c in classes] for p in preds])
        if emb is not None:
            out["embedding"] = emb
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)  # type: ignore[arg-type]
        return out

    def _output_drift(
        self, model: InferenceModel, ref: dict[str, np.ndarray], preds: pl.DataFrame
    ) -> dict[str, Any]:
        """Drift en el espacio de salida del modelo (probabilidades o valor predicho)."""
        if model.task == "regression":
            cur_v = preds["prediction"].cast(pl.Float64, strict=False).to_numpy()
            return numeric_drift("prediction", ref["values"], cur_v.astype(float)).model_dump(
                mode="json"
            )
        classes = [str(c) for c in (model.pipeline.classes or [])]
        cur_m = np.array(
            [[json.loads(s).get(c, 0.0) for c in classes] for s in preds["probabilities"].to_list()]
        )
        if len(cur_m) < 10:
            return {"severity": Severity.NONE.value, "note": "pocas predicciones"}
        return embedding_drift(ref["proba"], cur_m).model_dump(mode="json")

    @staticmethod
    def _embedding_drift(ref: dict[str, np.ndarray], preds: pl.DataFrame) -> dict[str, Any] | None:
        """Drift de los embeddings internos (RF-MON-02); None si el modelo no los expone."""
        if "embedding" not in ref or "embedding" not in preds.columns:
            return None
        vectors = [v for v in preds["embedding"].to_list() if v is not None]
        if len(vectors) < 10:
            return {"severity": Severity.NONE.value, "note": "pocas predicciones"}
        cur = np.asarray(vectors, dtype=np.float32)
        if cur.shape[1] != ref["embedding"].shape[1]:
            return None  # otra versión del modelo
        return embedding_drift(ref["embedding"], cur).model_dump(mode="json")

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
        mirror = mirror_for(self.ctx)
        if current is not None and current.id != mv.id:
            retired = self.models.update(
                current.model_copy(update={"stage": ModelStage.ARCHIVED, "retired_at": now})
            )
            if mirror is not None:
                mirror.sync_stage(retired)
        mv = self.models.update(
            mv.model_copy(update={"stage": ModelStage.PRODUCTION, "promoted_at": now})
        )
        if mirror is not None:
            mirror.sync_stage(mv)  # el alias `champion` pasa a esta versión
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
