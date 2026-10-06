"""Orquestación del flujo de un proyecto.

ingesta → profiling → pipeline → arquitectura → HPO → evaluación → registro.
Las reglas (Capa 1) son el camino por defecto; los roles del LLM (Capa 2) viven en
`services.llm_roles` y caen a estas mismas reglas si el LLM no está disponible.
La API y la CLI llaman a estas funciones; ninguna contiene lógica de ML propia.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from perceptron.api.context import EngineContext
from perceptron.archspec.schema import ArchSpec
from perceptron.archspec.validate import ValidationReport, offline_mode, validate_archspec
from perceptron.catalog import define
from perceptron.catalog.registry import DEFAULT_HF_TEXT_MODEL
from perceptron.catalog.rules import recommend
from perceptron.core.errors import NotFoundError, ValidationError
from perceptron.core.ids import IdPrefix, new_id
from perceptron.core.logging import log_context
from perceptron.data.pipeline.pipeline import FittedPipeline, PipelineSpec, fit_pipeline
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.profiling.card import ProfileCard
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.schema import SemanticType
from perceptron.data.splits import SplitRequest
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView
from perceptron.domain.enums import (
    DataSourceType,
    Device,
    Modality,
    Origin,
    ProjectStatus,
    RunStatus,
    TaskType,
)
from perceptron.domain.models import (
    ArchSpecRecord,
    DatasetVersion,
    DataSource,
    Evaluation,
    LLMCall,
    ModelVersion,
    Pipeline,
    Profile,
    Project,
    Run,
    Study,
    utcnow,
)
from perceptron.evaluation.cost import CostSpec
from perceptron.evaluation.evaluate import (
    EVALUATION_DIR,
    EVALUATION_FILE,
    EvaluationReport,
    build_model_version,
    evaluate_run,
)
from perceptron.export.formats import (
    EXPORT_DIR,
    REPORT_FILE,
    ExportReport,
    ExportRequest,
    export_run,
)
from perceptron.hpo.recommend import objective_metric, recommend_strategy
from perceptron.hpo.strategy import Budget, HPOStrategy
from perceptron.hpo.study import StudyControl, StudyResult, TrialRecord, gpu_plan, run_study
from perceptron.llm.schemas import Report
from perceptron.sandbox.code import CODE_FILE
from perceptron.sandbox.expert import build_code_spec, is_code_spec
from perceptron.sandbox.process import (
    CodeCheck,
    check_code,
    default_limits,
    evaluate_in_sandbox,
    export_in_sandbox,
)
from perceptron.sandbox.static import check_source
from perceptron.tracking.tracker import MlflowTracker, RunRecorder, Tracker
from perceptron.training.config import RunConfig, RunEvent, RunResult
from perceptron.training.hardware import detect_hardware

# Playground: sesiones de ONNX Runtime por run (se recargan si cambia el export).
_INFERENCE_CACHE: dict[str, tuple[tuple[str, int], Any]] = {}

if TYPE_CHECKING:
    from perceptron.services.llm_roles import LLMRoles

logger = logging.getLogger(__name__)
Mode = Literal["auto", "llm", "rules"]

_RUN_STATUS = {
    "succeeded": RunStatus.SUCCEEDED,
    "failed": RunStatus.FAILED,
    "cancelled": RunStatus.CANCELLED,
    "pruned": RunStatus.PRUNED,
    "paused": RunStatus.PAUSED,
}


class Workflow:
    def __init__(self, ctx: EngineContext, tracker: Tracker | None = None) -> None:
        self.ctx = ctx
        self._tracker = tracker

    @property
    def roles(self) -> LLMRoles:
        """Arquitecto, estratega de HPO, diagnosticador, informante y etiquetador (Capa 2)."""
        from perceptron.services.llm_roles import LLMRoles

        return LLMRoles(self)

    @property
    def tracker(self) -> Tracker:
        if self._tracker is None:
            self._tracker = MlflowTracker(
                self.ctx.settings.paths.mlflow_dir, self.ctx.settings.tracking_uri
            )
        return self._tracker

    # ------------------------------------------------------------------ helpers

    def project(self, project_id: str) -> Project:
        return self.ctx.projects.get(project_id)

    def dataset(self, dataset_version_id: str) -> DatasetVersion:
        return self.ctx.repo(DatasetVersion).get(dataset_version_id)

    def view(self, dv: DatasetVersion) -> DatasetView:
        if not dv.path:
            raise NotFoundError(f"la versión {dv.id} no tiene datos materializados")
        return DatasetView(self.ctx.settings.paths.project(dv.project_id).root / dv.path)

    def _fitted_path(self, pipeline: Pipeline, dv: DatasetVersion) -> Path:
        return (
            self.ctx.settings.paths.project(pipeline.project_id).pipelines_dir
            / f"{pipeline.id}.fitted.{dv.content_hash[:16]}.json"
        )

    # ------------------------------------------------------------------ datos

    def ingest(
        self,
        project_id: str,
        source: Path,
        *,
        target: str | None = None,
        split: SplitRequest | None = None,
        overrides: dict[str, SemanticType] | None = None,
        source_record: DataSource | None = None,
        modality: Modality | None = None,
        series_overrides: dict[str, Any] | None = None,
        task: TaskType | None = None,
    ) -> DatasetVersion:
        project = self.project(project_id)
        src = source_record
        if src is None:
            src = DataSource(
                project_id=project.id,
                name=source.name,
                type=DataSourceType.FOLDER if source.is_dir() else DataSourceType.FILE,
                config={"path": str(source)},
            )
            self.ctx.repo(DataSource).add(src)
        with log_context(project_id=project.id):
            dv = ingest(
                self.ctx.settings.paths.project(project.id),
                IngestRequest(
                    project_id=project.id,
                    source=source,
                    target=target,
                    split=split,
                    overrides=overrides,
                    source_id=src.id,
                    modality=modality,
                    series_overrides=series_overrides,
                    task=task,
                ),
            )
        existing = self.ctx.repo(DatasetVersion).list(
            filters={"project_id": project.id, "content_hash": dv.content_hash}, limit=1
        )
        if existing:
            return existing[0]
        self.ctx.repo(DatasetVersion).add(dv)
        updates: dict[str, Any] = {}
        if dv.modality and dv.modality not in project.modalities:
            updates["modalities"] = [*project.modalities, dv.modality]
        if project.status is ProjectStatus.DRAFT:
            updates["status"] = ProjectStatus.ACTIVE
        if updates:
            self.ctx.projects.update(project.model_copy(update=updates))
        self.ctx.events.publish("dataset.ingested", project_id=project.id, dataset_version_id=dv.id)
        return dv

    def profile(self, dataset_version_id: str) -> ProfileCard:
        dv = self.dataset(dataset_version_id)
        card = profile_dataset(
            self.view(dv), dataset_version_id=dv.id, content_hash=dv.content_hash
        )
        repo = self.ctx.repo(Profile)
        for old in repo.list(filters={"dataset_version_id": dv.id}):
            repo.delete(old.id)
        repo.add(
            Profile(
                dataset_version_id=dv.id,
                modality=card.modality,
                stats=card.model_dump(mode="json"),
                alerts=[a.model_dump(mode="json") for a in card.alerts],
            )
        )
        return card

    def profile_card(self, dataset_version_id: str) -> ProfileCard:
        """Perfil del dataset con las alertas ajustadas a la ficha del caso (ADR-0040)."""
        from perceptron.services.brief import contextualize_card, project_use_case

        found = self.ctx.repo(Profile).list(
            filters={"dataset_version_id": dataset_version_id}, limit=1
        )
        card = (
            ProfileCard.model_validate(found[0].stats)
            if found
            else self.profile(dataset_version_id)
        )
        dv = self.dataset(dataset_version_id)
        return contextualize_card(card, project_use_case(self.ctx, dv.project_id))

    # ------------------------------------------------------------------ pipeline

    def propose_pipeline(
        self, dataset_version_id: str, *, pretrained: bool | None = None
    ) -> Pipeline:
        dv = self.dataset(dataset_version_id)
        card = self.profile_card(dv.id)
        use_pretrained = (not offline_mode()) if pretrained is None else pretrained
        spec = propose_pipeline(
            card, pretrained=use_pretrained and self._has_gpu(), hf_model=DEFAULT_HF_TEXT_MODEL
        )
        pipeline = Pipeline(
            project_id=dv.project_id,
            name=f"auto-{dv.content_hash[:8]}",
            graph=spec.model_dump(mode="json"),
            origin=Origin.RULES,
        )
        self.ctx.repo(Pipeline).add(pipeline)
        return pipeline

    def update_pipeline(self, pipeline_id: str, spec: PipelineSpec, version: int) -> Pipeline:
        current = self.ctx.repo(Pipeline).get(pipeline_id)
        return self.ctx.repo(Pipeline).update(
            current.model_copy(
                update={
                    "graph": spec.model_dump(mode="json"),
                    "origin": Origin.MANUAL,
                    "version": version,
                }
            )
        )

    def fitted_pipeline(self, pipeline_id: str, dataset_version_id: str) -> FittedPipeline:
        pipeline = self.ctx.repo(Pipeline).get(pipeline_id)
        dv = self.dataset(dataset_version_id)
        path = self._fitted_path(pipeline, dv)
        if path.is_file():
            fitted = FittedPipeline.model_validate_json(path.read_text(encoding="utf-8"))
            if fitted.spec.model_dump(mode="json") == pipeline.graph:
                return fitted
        spec = PipelineSpec.model_validate(pipeline.graph)
        view = self.view(dv)
        fitted = fit_pipeline(spec, view.read("train"), view.files_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fitted.model_dump_json(indent=2), encoding="utf-8")
        return fitted

    # ------------------------------------------------------------------ arquitectura

    def _has_gpu(self) -> bool:
        return bool(detect_hardware(self.ctx.settings.workspace_dir).gpus)

    def propose_architecture(
        self, dataset_version_id: str, pipeline_id: str
    ) -> tuple[ArchSpecRecord, str]:
        dv = self.dataset(dataset_version_id)
        card = self.profile_card(dv.id)
        fitted = self.fitted_pipeline(pipeline_id, dv.id)
        rec = recommend(card, fitted, detect_hardware(self.ctx.settings.workspace_dir))
        record = self.save_archspec(dv.project_id, rec.spec, origin=Origin.RULES)
        return record, rec.rationale

    def define_plan(
        self, dataset_version_id: str, pipeline_id: str, choices: dict[str, str]
    ) -> define.DefinePlan:
        """Opciones del sub-wizard de definición de arquitectura (SPEC §7.6, paso 6)."""
        dv = self.dataset(dataset_version_id)
        card = self.profile_card(dv.id)
        fitted = self.fitted_pipeline(pipeline_id, dv.id)
        hw = detect_hardware(self.ctx.settings.workspace_dir)
        try:
            return define.plan(card, fitted, hw, choices)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def define_build(
        self, dataset_version_id: str, pipeline_id: str, choices: dict[str, str]
    ) -> ArchSpecRecord:
        """Arma, valida y guarda la ArchSpec elegida en el sub-wizard (origen manual)."""
        dv = self.dataset(dataset_version_id)
        card = self.profile_card(dv.id)
        fitted = self.fitted_pipeline(pipeline_id, dv.id)
        hw = detect_hardware(self.ctx.settings.workspace_dir)
        try:
            spec = define.build(card, fitted, hw, choices)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return self.save_archspec(dv.project_id, spec, origin=Origin.MANUAL)

    def save_archspec(
        self,
        project_id: str,
        spec: ArchSpec,
        *,
        origin: Origin = Origin.MANUAL,
        code: str | None = None,
    ) -> ArchSpecRecord:
        report = validate_archspec(spec)
        if not report.valid:
            raise ValidationError(
                "ArchSpec inválida",
                details={"issues": [i.model_dump(mode="json") for i in report.errors]},
            )
        if is_code_spec(spec) and code is None:
            # Solo `save_code_archspec` (confirmación + validación estática y en sandbox).
            raise ValidationError("las ArchSpec de código se crean desde el modo experto")
        record = ArchSpecRecord(
            project_id=project_id,
            name=spec.name,
            spec=spec.model_dump(mode="json"),
            content_hash=spec.content_hash(),
            origin=origin,
        )
        archspecs_dir = self.ctx.settings.paths.project(project_id).archspecs_dir
        archspecs_dir.mkdir(parents=True, exist_ok=True)
        if code is not None:
            (archspecs_dir / f"{record.id}.py").write_text(code, encoding="utf-8")
            record = record.model_copy(update={"code_path": f"archspecs/{record.id}.py"})
        self.ctx.repo(ArchSpecRecord).add(record)
        path = archspecs_dir / f"{record.id}.json"
        path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        return record

    def archspec_source(self, record: ArchSpecRecord) -> str:
        """Fuente del código experto de una ArchSpec (RF-ARC-06)."""
        if not record.code_path:
            raise NotFoundError(f"la ArchSpec {record.id} no es de código")
        root = self.ctx.settings.paths.project(record.project_id).root
        return (root / record.code_path).read_text(encoding="utf-8")

    def save_code_archspec(
        self,
        project_id: str,
        base_archspec_id: str,
        source: str,
        *,
        name: str | None = None,
        acknowledge_risk: bool = False,
    ) -> tuple[ArchSpecRecord, CodeCheck]:
        """Modo experto (RF-ARC-06, ADR-0025): validación estática, prueba en sandbox y alta."""
        if not acknowledge_risk:
            raise ValidationError(
                "el código experto se ejecuta en un sandbox pero no es declarativo: "
                "confirmá el riesgo (acknowledge_risk)"
            )
        static = check_source(source)
        if not static.valid:
            raise ValidationError(
                "el código no pasa la validación estática",
                details={"issues": [i.model_dump() for i in static.issues]},
            )
        base_record = self.ctx.repo(ArchSpecRecord).get(base_archspec_id)
        base = ArchSpec.model_validate(base_record.spec)
        try:
            spec = build_code_spec(base, source, name)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        work = self.ctx.settings.paths.project(project_id).root / "sandbox" / new_id(IdPrefix.JOB)
        check = check_code(spec.model_dump(mode="json"), source, work)
        if not check.ok:
            raise ValidationError(
                f"el modelo no se pudo construir en el sandbox: {check.error}",
                details={"violations": check.violations},
            )
        return self.save_archspec(project_id, spec, origin=Origin.MANUAL, code=source), check

    @staticmethod
    def validate(data: dict[str, Any]) -> ValidationReport:
        return validate_archspec(data)

    # ------------------------------------------------------------------ HPO / entrenamiento

    def hpo_strategy(
        self,
        archspec_id: str,
        budget: Budget,
        *,
        mode: Mode = "rules",
        dataset_version_id: str | None = None,
    ) -> HPOStrategy:
        if mode != "rules":
            return self.roles.hpo_strategy(
                archspec_id, budget, mode=mode, dataset_version_id=dataset_version_id
            )
        record = self.ctx.repo(ArchSpecRecord).get(archspec_id)
        spec = ArchSpec.model_validate(record.spec)
        project = self.ctx.projects.get(record.project_id)
        # La métrica del proyecto (p. ej. de su plantilla, RF-PRJ-02) es el objetivo del HPO.
        return recommend_strategy(
            spec, budget, metric=objective_metric(spec, project.target_metric)
        )

    def run_study(
        self,
        project_id: str,
        dataset_version_id: str,
        pipeline_id: str,
        archspec_id: str,
        strategy: HPOStrategy,
        *,
        device: Device | None = None,
        control: StudyControl | None = None,
        on_event: Callable[[RunEvent], None] | None = None,
        study: Study | None = None,
        limit_train_batches: float | None = None,
    ) -> tuple[Study, StudyResult]:
        project = self.project(project_id)
        dv = self.dataset(dataset_version_id)
        record = self.ctx.repo(ArchSpecRecord).get(archspec_id)
        fitted = self.fitted_pipeline(pipeline_id, dv.id)
        hw = detect_hardware(self.ctx.settings.workspace_dir)
        st = study or Study(
            project_id=project.id,
            name=f"hpo-{record.name}",
            strategy=strategy.model_dump(mode="json"),
            budget=strategy.budget.model_dump(mode="json"),
            objectives=[o.metric for o in strategy.objectives],
            origin=strategy.origin,
        )
        if self.ctx.repo(Study).find(st.id) is None:
            self.ctx.repo(Study).add(st)
        ppaths = self.ctx.settings.paths.project(project.id)
        code = self.archspec_source(record) if record.code_path else None
        base = RunConfig(
            run_id=st.id,
            run_dir=ppaths.run(st.id),
            dataset_dir=self.view(dv).root,
            archspec=record.spec or {},
            pipeline=fitted.model_dump(mode="json"),
            device=device or hw.recommended_device,
            seed=strategy.seed,
            pretrained_allowed=not offline_mode(),
            limit_train_batches=limit_train_batches,
            code=code,
            sandbox=default_limits(strategy.budget.trial_timeout_s),
        )
        recorders: dict[str, RunRecorder] = {}
        tags = {
            "perceptron.project": project.id,
            "perceptron.study": st.id,
            "perceptron.dataset": dv.content_hash,
            "perceptron.origin": record.origin.value,
            "perceptron.declarative": "false" if code is not None else "true",
        }

        def on_run_event(cfg: RunConfig) -> Callable[[RunEvent], None]:
            recorders[cfg.run_id] = RunRecorder(self.tracker, cfg, experiment=project.id, tags=tags)
            self.ctx.repo(Run).add(
                Run(
                    id=cfg.run_id,
                    project_id=project.id,
                    study_id=st.id,
                    archspec_id=record.id,
                    pipeline_id=pipeline_id,
                    dataset_version_id=dv.id,
                    status=RunStatus.RUNNING,
                    device=cfg.device,
                    hyperparams=dict(cfg.overrides),
                    seed=cfg.seed,
                    started_at=utcnow(),
                )
            )

            def forward(ev: RunEvent) -> None:
                recorders[cfg.run_id](ev)
                self.ctx.events.publish(
                    "run.event", run_id=cfg.run_id, event=ev.model_dump(mode="json")
                )
                if on_event is not None:
                    on_event(ev)

            return forward

        def on_trial_end(rec: TrialRecord, cfg: RunConfig, result: RunResult) -> None:
            mlflow_id = recorders[cfg.run_id].finish(result)
            run = self.ctx.repo(Run).get(cfg.run_id)
            self.ctx.repo(Run).update(
                run.model_copy(
                    update={
                        # El pruner corta el trial como una cancelación: se informa como
                        # «podado» (antes figuraba «cancelado» y confundía).
                        "status": RunStatus.PRUNED
                        if rec.state == "pruned"
                        else _RUN_STATUS.get(result.status, RunStatus.FAILED),
                        "metrics": result.best_metrics,
                        "environment": result.environment,
                        "mlflow_run_id": mlflow_id,
                        "finished_at": utcnow(),
                        "diagnosis": {"error": result.error} if result.error else None,
                    }
                )
            )

        gpus: list[int] | None = None
        if base.device in (Device.CUDA, Device.ROCM):
            # Varias GPUs del mismo tipo: un trial en paralelo por GPU (RF-HPO-04).
            hardware = detect_hardware(self.ctx.settings.workspace_dir)
            gpus = [g.index for g in hardware.gpus if g.backend == base.device] or None
        base, gpus = gpu_plan(base, strategy, gpus)
        with log_context(project_id=project.id, job_id=st.id):
            result = run_study(
                strategy,
                base,
                storage=ppaths.root / "hpo" / "optuna.db",
                study_name=st.id,
                control=control,
                bus=self.ctx.events,
                on_run_event=on_run_event,
                on_trial_end=on_trial_end,
                gpus=gpus,
            )
        summary = result.model_dump(mode="json")
        current = self.ctx.repo(Study).get(st.id)
        self.ctx.repo(Study).update(
            current.model_copy(
                update={
                    "budget": {
                        **current.budget,
                        "result": {
                            "stop_reason": summary["stop_reason"],
                            "best_trial": summary["best_trial"],
                            "duration_s": summary["duration_s"],
                        },
                    }
                }
            )
        )
        return current, result

    # ------------------------------------------------------------------ evaluación y registro

    def _run_dir(self, run: Run) -> Path:
        return self.ctx.settings.paths.project(run.project_id).run(run.id)

    def _cost_spec(self, project_id: str) -> CostSpec | None:
        """Costos de error de la ficha del caso (ADR-0040), si declara cuál es peor y cuánto."""
        from perceptron.services.brief import project_use_case

        brief = project_use_case(self.ctx, project_id) or {}
        worse: dict[str, Literal["false_negative", "false_positive"]] = {
            "false_negative_worse": "false_negative",
            "false_positive_worse": "false_positive",
        }
        kind = worse.get(str(brief.get("error_costs")))
        if kind is None or not brief.get("error_cost_ratio"):
            return None
        return CostSpec(worse=kind, ratio=float(brief["error_cost_ratio"]))

    def evaluate(self, run_id: str) -> tuple[Evaluation, EvaluationReport]:
        run = self.ctx.repo(Run).get(run_id)
        dv = self.dataset(run.dataset_version_id)
        run_dir, dataset_dir = self._run_dir(run), self.view(dv).root
        if (run_dir / CODE_FILE).is_file():
            # Código experto: el modelo se construye y evalúa solo dentro del sandbox.
            evaluate_in_sandbox(run_dir, dataset_dir)
            report = EvaluationReport.model_validate(
                json.loads((run_dir / EVALUATION_DIR / EVALUATION_FILE).read_text(encoding="utf-8"))
            )
        else:
            report = evaluate_run(run_dir, dataset_dir, cost=self._cost_spec(run.project_id))
        evaluation = Evaluation(
            run_id=run.id,
            split=report.split,
            metrics=report.metrics,
            artifacts={"report": f"runs/{run.id}/{EVALUATION_DIR}/{EVALUATION_FILE}"},
        )
        self.ctx.repo(Evaluation).add(evaluation)
        if run.mlflow_run_id:
            self.tracker.log_metrics(
                run.mlflow_run_id, {f"test.{k}": v for k, v in report.metrics.items()}
            )
            self.tracker.log_artifact(
                run.mlflow_run_id,
                self._run_dir(run) / EVALUATION_DIR / EVALUATION_FILE,
                "evaluation",
            )
        return evaluation, report

    # ------------------------------------------------------------------ export (RF-EXP-01)

    def export(self, run_id: str, request: ExportRequest) -> ExportReport:
        """Exporta el modelo del run y verifica cada formato (ADR-0027)."""
        run = self.ctx.repo(Run).get(run_id)
        if run.status is not RunStatus.SUCCEEDED:
            raise ValidationError(f"el run {run_id} no terminó bien ({run.status.value})")
        run_dir = self._run_dir(run)
        dataset_dir = self.view(self.dataset(run.dataset_version_id)).root
        if (run_dir / CODE_FILE).is_file():
            # Código experto: exportar ejecuta el modelo, así que va al sandbox.
            export_in_sandbox(run_dir, dataset_dir, request.model_dump(mode="json"))
            return self.export_report(run_id)
        return export_run(run_dir, dataset_dir, request)

    def export_report(self, run_id: str) -> ExportReport:
        run = self.ctx.repo(Run).get(run_id)
        path = self._run_dir(run) / EXPORT_DIR / REPORT_FILE
        if not path.is_file():
            raise NotFoundError(f"el run {run_id} no fue exportado")
        return ExportReport.model_validate_json(path.read_text(encoding="utf-8"))

    def export_file(self, run_id: str, name: str) -> Path:
        """Ruta de un artefacto exportado (solo los del reporte, la firma o el pipeline)."""
        report = self.export_report(run_id)
        allowed = {a.file for a in report.artifacts if a.file} | {
            "signature.json",
            "pipeline.json",
            REPORT_FILE,
        }
        if name not in allowed:
            raise NotFoundError(f"el export del run {run_id} no tiene {name!r}")
        return self._run_dir(self.ctx.repo(Run).get(run_id)) / EXPORT_DIR / name

    def serving_bundle(self, run_id: str) -> Path:
        """Zip del servidor de inferencia (RF-EXP-03) desde el export ONNX verificado."""
        from perceptron.serving.bundle import build_bundle

        run = self.ctx.repo(Run).get(run_id)
        record = self.ctx.repo(ArchSpecRecord).find(run.archspec_id)
        name = record.name if record else run_id
        return build_bundle(self._run_dir(run), name)

    def project_zip(self, run_id: str) -> Path:
        """Proyecto de código autónomo del run (RF-EXP-04, O5)."""
        from perceptron.export.project import build_project

        run = self.ctx.repo(Run).get(run_id)
        record = self.ctx.repo(ArchSpecRecord).find(run.archspec_id)
        dataset_dir = self.view(self.dataset(run.dataset_version_id)).root
        return build_project(self._run_dir(run), dataset_dir, record.name if record else run_id)

    def inference_model(self, run_id: str) -> Any:
        """Modelo ONNX exportado listo para el playground (RF-EXP-02), cacheado por archivo."""
        from perceptron.serving.runtime import MODEL_FILE, InferenceModel

        run = self.ctx.repo(Run).get(run_id)
        model_dir = self._run_dir(run) / EXPORT_DIR
        onnx = model_dir / MODEL_FILE
        if not onnx.is_file():
            raise NotFoundError(f"el run {run_id} no tiene export ONNX (exportalo primero)")
        key = (str(onnx), onnx.stat().st_mtime_ns)
        cached = _INFERENCE_CACHE.get(run_id)
        if cached is None or cached[0] != key:
            cached = (key, InferenceModel(model_dir))
            _INFERENCE_CACHE[run_id] = cached
        return cached[1]

    # ------------------------------------------------------------------ análisis (RF-EVL-03/04)

    def _eval_paths(self, run_id: str) -> tuple[Run, Path, Path]:
        run = self.ctx.repo(Run).get(run_id)
        dataset_dir = self.view(self.dataset(run.dataset_version_id)).root
        return run, self._run_dir(run) / EVALUATION_DIR, dataset_dir

    def error_analysis(self, run_id: str) -> Any:
        from perceptron.evaluation.errors import analyze_errors

        run, eval_dir, dataset_dir = self._eval_paths(run_id)
        target = self.dataset(run.dataset_version_id).target
        return analyze_errors(eval_dir, dataset_dir, target)

    def fairness(
        self,
        run_id: str,
        attributes: list[str],
        *,
        positive_class: str | None = None,
        threshold: float = 0.1,
    ) -> list[Any]:
        from perceptron.evaluation.fairness import fairness_report

        _, eval_dir, dataset_dir = self._eval_paths(run_id)
        reports = [
            fairness_report(
                eval_dir, dataset_dir, a, positive_class=positive_class, threshold=threshold
            )
            for a in attributes
        ]
        (eval_dir / "fairness.json").write_text(
            json.dumps([r.model_dump(mode="json") for r in reports], indent=2), encoding="utf-8"
        )
        return reports

    def explanation(self, run_id: str) -> Any:
        """Importancia global de las features (RF-EVL-02), cacheada junto a la evaluación."""
        from perceptron.evaluation.explain import GlobalExplanation, global_explanation

        run = self.ctx.repo(Run).get(run_id)
        path = self._run_dir(run) / EVALUATION_DIR / "explanation.json"
        if path.is_file():
            return GlobalExplanation.model_validate_json(path.read_text(encoding="utf-8"))
        dataset_dir = self.view(self.dataset(run.dataset_version_id)).root
        result = global_explanation(self._run_dir(run), dataset_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def explain_row(self, run_id: str, row: dict[str, Any]) -> Any:
        from perceptron.evaluation.explain import local_tabular

        return local_tabular(self._run_dir(self.ctx.repo(Run).get(run_id)), row)

    def explain_image(self, run_id: str, image: Any) -> Any:
        from perceptron.evaluation.explain import local_image

        return local_image(self._run_dir(self.ctx.repo(Run).get(run_id)), image)

    def explain_text(self, run_id: str, text: str) -> Any:
        from perceptron.evaluation.explain import local_text

        return local_text(self._run_dir(self.ctx.repo(Run).get(run_id)), text)

    def explain_audio(self, run_id: str, path: Path) -> Any:
        from perceptron.evaluation.explain import local_audio

        return local_audio(self._run_dir(self.ctx.repo(Run).get(run_id)), path)

    def robustness(self, run_id: str) -> Any:
        """Degradación ante perturbaciones (RF-EVL-05), cacheada junto a la evaluación."""
        from perceptron.evaluation.robustness import RobustnessReport, robustness_report

        run = self.ctx.repo(Run).get(run_id)
        path = self._run_dir(run) / EVALUATION_DIR / "robustness.json"
        if path.is_file():
            return RobustnessReport.model_validate_json(path.read_text(encoding="utf-8"))
        dataset_dir = self.view(self.dataset(run.dataset_version_id)).root
        result = robustness_report(self._run_dir(run), dataset_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def report_document(self, run_id: str, fmt: str) -> tuple[bytes, str]:
        """Informe final con marca (RF-EVL-06): html, pdf o md, con lo que ya se analizó."""
        from perceptron.evaluation.report_render import (
            ReportInputs,
            render_html,
            render_markdown,
            render_pdf,
        )
        from perceptron.services.llm_roles import REPORT_DIR

        run = self.ctx.repo(Run).get(run_id)
        run_dir = self._run_dir(run)
        stored = run_dir / REPORT_DIR / "report.json"
        report = (
            Report.model_validate_json(stored.read_text(encoding="utf-8"))
            if stored.is_file()
            else self.roles.report(run_id, mode="rules")  # sin LLM ni costo si no hay uno
        )
        eval_dir = run_dir / EVALUATION_DIR

        def cached(name: str) -> Any:
            p = eval_dir / name
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

        inputs = ReportInputs(
            report=report,
            evaluation=self.evaluation_report(run_id),
            run_id=run_id,
            project=self.project(run.project_id).name,
            explanation=cached("explanation.json"),
            fairness=cached("fairness.json") or [],
            robustness=cached("robustness.json"),
        )
        if fmt == "pdf":
            return render_pdf(inputs), "application/pdf"
        if fmt == "md":
            return render_markdown(inputs).encode("utf-8"), "text/markdown; charset=utf-8"
        return render_html(inputs).encode("utf-8"), "text/html; charset=utf-8"

    def evaluation_report(self, run_id: str) -> EvaluationReport:
        run = self.ctx.repo(Run).get(run_id)
        path = self._run_dir(run) / EVALUATION_DIR / EVALUATION_FILE
        if not path.is_file():
            raise NotFoundError(f"el run {run_id} no fue evaluado")
        return EvaluationReport.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def baseline(
        self, dataset_version_id: str, pipeline_id: str, project_id: str
    ) -> dict[str, Any] | None:
        """LightGBM de referencia (§5.1): mismo pipeline y test sellado; se registra en MLflow."""
        from perceptron.evaluation.baseline import lightgbm_baseline

        dv = self.dataset(dataset_version_id)
        try:
            result = lightgbm_baseline(self.view(dv), self.fitted_pipeline(pipeline_id, dv.id))
        except (ImportError, ValueError) as e:
            logger.warning("baseline no disponible", extra={"error": str(e)})
            return None
        rid = self.tracker.start_run(
            project_id, "baseline-lightgbm", {"perceptron.baseline": "lightgbm"}
        )
        self.tracker.log_metrics(rid, {f"test.{k}": v for k, v in result["metrics"].items()})
        self.tracker.end_run(rid, "FINISHED")
        return result

    def register(self, run_id: str) -> ModelVersion:
        run = self.ctx.repo(Run).get(run_id)
        dv = self.dataset(run.dataset_version_id)
        report = self.evaluation_report(run_id)
        mv = build_model_version(
            run.project_id, run.id, self._run_dir(run), report, dataset_hash=dv.content_hash
        )
        from perceptron.tracking.registry import RegistryMirror

        if isinstance(self.tracker, MlflowTracker):
            version = RegistryMirror(self.tracker.client).register(mv, run.mlflow_run_id)
            mv = mv.model_copy(update={"mlflow_version": version})
        self.ctx.repo(ModelVersion).add(mv)
        if run.mlflow_run_id:
            self.tracker.set_tags(run.mlflow_run_id, {"perceptron.model_version": mv.id})
        return mv


# ---------------------------------------------------------------------- quickstart


@dataclass
class QuickstartResult:
    project: Project
    dataset: DatasetVersion
    card: ProfileCard
    pipeline: Pipeline
    archspec: ArchSpecRecord
    arch_rationale: str
    strategy: HPOStrategy
    study: Study
    study_result: StudyResult
    evaluation: EvaluationReport | None
    model_version: ModelVersion | None
    baseline: dict[str, Any] | None = None
    arch_origin: Origin = Origin.RULES
    llm_fallback: str | None = None
    report: Report | None = None
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    tournament: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        best = self.study_result.best_trial
        return {
            "project_id": self.project.id,
            "dataset_version_id": self.dataset.id,
            "modality": self.dataset.modality.value if self.dataset.modality else None,
            "target": self.dataset.target,
            "samples": self.dataset.num_samples,
            "alerts": [a.code.value for a in self.card.alerts],
            "architecture": self.archspec.name,
            "architecture_rationale": self.arch_rationale,
            "hpo_strategy": self.strategy.strategy,
            "hpo_pruner": self.strategy.pruner,
            "trials": len(self.study_result.trials),
            "trial_states": {
                s: sum(t.state == s for t in self.study_result.trials)
                for s in ("complete", "pruned", "fail")
            },
            "stop_reason": self.study_result.stop_reason,
            "best_run_id": best.run_id if best else None,
            "best_params": best.params if best else None,
            "test_metrics": self.evaluation.metrics if self.evaluation else None,
            "test_per_class": {
                c.label: {
                    "recall": round(c.recall, 4),
                    "precision": round(c.precision, 4),
                    "support": c.support,
                }
                for c in self.evaluation.classification.per_class
            }
            if self.evaluation and self.evaluation.classification
            else None,
            "test_confusion": self.evaluation.classification.confusion_matrix
            if self.evaluation and self.evaluation.classification
            else None,
            "model_version_id": self.model_version.id if self.model_version else None,
            "baseline": self.baseline,
            "architecture_origin": self.arch_origin.value,
            "hpo_origin": self.strategy.origin.value,
            "llm_fallback": self.llm_fallback,
            "llm_calls": self.llm_calls,
            "llm_cost_usd": round(self.llm_cost_usd, 6),
            "report_origin": self.report.origin if self.report else None,
            "tournament": self.tournament,
        }


def quickstart(
    ctx: EngineContext,
    source: Path,
    *,
    name: str | None = None,
    target: str | None = None,
    modality: Modality | None = None,
    series_overrides: dict[str, Any] | None = None,
    task: TaskType | None = None,
    trials: int = 10,
    max_epochs: int | None = None,
    max_time_s: float | None = None,
    device: Device | None = None,
    on_event: Callable[[RunEvent], None] | None = None,
    tracker: Tracker | None = None,
    llm: bool = False,
    goal: str | None = None,
) -> QuickstartResult:
    """Todo el flujo sin escribir código. Con `llm`, arquitectura, HPO e informe los propone
    el LLM (si está disponible; si no, reglas). Camino de aceptación de las Capas 1 y 2."""
    wf = Workflow(ctx, tracker)
    project = ctx.projects.add(
        Project(name=name or source.stem, goal=goal or ("" if llm else "quickstart"))
    )
    ctx.files.init_project(project)
    dv = wf.ingest(
        project.id,
        source,
        target=target,
        modality=modality,
        series_overrides=series_overrides,
        task=task,
    )
    card = wf.profile(dv.id)
    pipeline = wf.propose_pipeline(dv.id)
    budget = Budget(max_trials=trials, max_epochs_per_trial=max_epochs, max_time_s=max_time_s)
    arch_origin, fallback = Origin.RULES, None
    tournament: list[dict[str, Any]] = []
    if llm:
        proposals = wf.roles.propose_architectures(dv.id, pipeline.id, mode="auto")
        chosen = proposals.options[0]
        if len(proposals.options) > 1:
            from perceptron.services.autodesign import (
                TOURNAMENT_FRACTION,
                TOURNAMENT_MIN_EPOCHS,
                common_metric,
                tournament_subset,
            )
            from perceptron.services.tournament import mini_tournament

            # Como el diseño guiado (ADR-0041): métrica que todas registran y, con pocos
            # datos, todo train en cada época.
            specs = [ArchSpec.model_validate(o.record.spec) for o in proposals.options]
            n_train = int(card.split_counts.get("train") or card.num_samples)
            t = mini_tournament(
                wf,
                project.id,
                dv.id,
                pipeline.id,
                [o.record.id for o in proposals.options],
                device=device,
                fraction=TOURNAMENT_FRACTION,
                subset=tournament_subset(n_train),
                metric=common_metric(specs),
                min_epochs=TOURNAMENT_MIN_EPOCHS,
            )
            tournament = [
                {"architecture": e.name, "metric": e.metric, "status": e.status} for e in t.entries
            ]
            chosen = next((o for o in proposals.options if o.record.id == t.winner), chosen)
        archspec, why = chosen.record, chosen.rationale
        arch_origin, fallback = proposals.origin, proposals.fallback_reason
    else:
        archspec, why = wf.propose_architecture(dv.id, pipeline.id)
    strategy = wf.hpo_strategy(
        archspec.id, budget, mode="auto" if llm else "rules", dataset_version_id=dv.id
    )
    study, result = wf.run_study(
        project.id, dv.id, pipeline.id, archspec.id, strategy, device=device, on_event=on_event
    )
    evaluation = mv = None
    if result.best_trial is not None:
        _, evaluation = wf.evaluate(result.best_trial.run_id)
        mv = wf.register(result.best_trial.run_id)
    baseline = None
    if dv.modality is Modality.TABULAR and evaluation is not None:
        baseline = wf.baseline(dv.id, pipeline.id, project.id)
    report = None
    if llm and result.best_trial is not None:
        report = wf.roles.report(result.best_trial.run_id, mode="auto")
    calls = list(ctx.repo(LLMCall).list(filters={"project_id": project.id}, limit=1000))
    return QuickstartResult(
        project=ctx.projects.get(project.id),
        dataset=dv,
        card=card,
        pipeline=pipeline,
        archspec=archspec,
        arch_rationale=why,
        strategy=strategy,
        study=study,
        study_result=result,
        evaluation=evaluation,
        model_version=mv,
        baseline=baseline,
        arch_origin=arch_origin,
        llm_fallback=fallback,
        report=report,
        llm_calls=len(calls),
        llm_cost_usd=sum(c.cost_usd for c in calls),
        tournament=tournament,
    )


def new_study_id() -> str:
    return new_id(IdPrefix.STUDY)
