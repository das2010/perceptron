"""Guided image journey (RF-WIZ-01/03/04, RF-ING-08, RF-EXP-02).

The LLM extracts evidence; deterministic guards own all side effects. Documents use
atomic compare-and-swap, including SQLite, before launching any background work.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import update
from sqlalchemy.engine import CursorResult

from perceptron.api.jobs import JobContext
from perceptron.core.errors import ConflictError, ValidationError
from perceptron.domain.enums import LLMPurpose, Modality, SplitStrategy
from perceptron.domain.models import JourneyRecord, ModelVersion, utcnow
from perceptron.hpo.strategy import Budget
from perceptron.llm.privacy import LLMContext
from perceptron.services.workflow import Workflow
from perceptron.storage.db import EntityRow

State = Literal[
    "interview",
    "awaiting_data",
    "reviewing_data",
    "ready_to_train",
    "training",
    "ready_to_validate",
    "needs_improvement",
    "ready_to_use",
    "interrupted",
    "failed",
    "unsupported",
]
ACTIVE = {"reviewing_data", "training"}
PROCESS_STARTED = psutil.Process().create_time()


class Facts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modality: Literal["image", "other"] | None = None
    task: Literal["classification", "detection", "segmentation", "other"] | None = None
    classes: list[str] | None = Field(default=None, min_length=2, max_length=100)
    positive_class: str | None = Field(default=None, max_length=100)
    grouped: bool | None = None
    labels_available: bool | None = None
    success_percent: float | None = Field(default=None, ge=1, le=100)
    error_costs: Literal["misses", "false_alarms", "equal"] | None = None


class Extraction(BaseModel):
    facts: Facts = Field(default_factory=Facts)
    evidence: dict[str, str] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    question: str | None = Field(default=None, max_length=1000)
    question_field: (
        Literal[
            "modality",
            "task",
            "classes",
            "positive_class",
            "grouped",
            "labels_available",
            "success_percent",
            "error_costs",
        ]
        | None
    ) = None


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class Journey(BaseModel):
    schema_version: int = 1
    version: int = 1
    state: State = "interview"
    facts: Facts = Field(default_factory=Facts)
    messages: list[Turn] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    llm_failed: bool = False
    confirmed: bool = False
    dataset_id: str | None = None
    job_id: str | None = None
    child_job_id: str | None = None
    study_id: str | None = None
    owner_pid: int | None = None
    owner_started: float | None = None
    stage: str = "interview"
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    run_id: str | None = None
    model_id: str | None = None
    contract: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    validations: list[dict[str, Any]] = Field(default_factory=list)
    notification: str | None = None
    error: str | None = None
    stop_requested: bool = False


def missing(f: Facts) -> list[str]:
    needed = [k for k, v in f.model_dump().items() if v is None]
    if f.classes and (
        len(set(f.classes)) != len(f.classes) or any(not c.strip() for c in f.classes)
    ):
        needed.append("classes")
    if f.positive_class and f.classes and f.positive_class not in f.classes:
        needed.append("positive_class")
    return list(dict.fromkeys(needed))


class Coordinator:
    def __init__(self, wf: Workflow):
        self.wf, self.ctx = wf, wf.ctx
        self.repo = self.ctx.repo(JourneyRecord)

    def get(self, pid: str) -> Journey:
        self.wf.project(pid)
        record = self.repo.find(pid)
        if record is None:
            return Journey(pending=missing(Facts()))
        j = Journey.model_validate(record.payload).model_copy(update={"version": record.version})
        if j.state in ACTIVE and j.owner_pid:
            try:
                alive = psutil.Process(j.owner_pid).create_time() == j.owner_started
            except psutil.Error:
                alive = False
            if alive and j.owner_pid == os.getpid() and j.job_id:
                alive = self.ctx.jobs.get(j.job_id) is not None
            if not alive:
                # The durable study reference also covers a crash between enqueue and
                # saving child_job_id. Never retry while a restored remote worker is active.
                child = self.ctx.jobs.get(j.child_job_id) if j.child_job_id else None
                if child is None and j.study_id:
                    child = next(
                        (
                            job
                            for job in self.ctx.jobs.list()
                            if job.refs.get("study_id") == j.study_id
                        ),
                        None,
                    )
                if child and child.finished_at is None:
                    self.ctx.jobs.cancel(child.id)
                    j.stop_requested, j.stage = True, "stopping"
                    return j
                if j.study_id:
                    from perceptron.services.study_control import mark_interrupted

                    mark_interrupted(self.ctx, j.study_id)
                j.state, j.stage = "interrupted", "interrupted"
                j.notification = "interrupted"
                j.error = "El motor se interrumpió. Podés reintentar conservando tus datos."
                return self.save(pid, j)
        return j

    def revise(self, pid: str, version: int) -> Journey:
        j = self.get(pid)
        if j.version != version or j.state in ACTIVE:
            raise ConflictError("Esperá a que termine el trabajo antes de corregir el escenario.")
        return self.save(
            pid,
            Journey(
                version=j.version,
                facts=j.facts,
                messages=j.messages,
                assumptions=j.assumptions,
                pending=missing(j.facts),
            ),
        )

    def known_images(self, dataset_id: str) -> dict[str, str]:
        from perceptron.data.view import MANIFEST_FILE

        view = self.wf.view(self.wf.dataset(dataset_id))
        manifest = json.loads((view.root / MANIFEST_FILE).read_text(encoding="utf-8"))
        return {
            e["path"]: e["sha256"] for e in manifest["entries"] if e["path"].startswith("files/")
        }

    def save(self, pid: str, j: Journey) -> Journey:
        record = self.repo.find(pid)
        if record is None:
            self.repo.add(JourneyRecord(id=pid, project_id=pid, payload=j.model_dump(mode="json")))
            return j
        next_j = j.model_copy(update={"version": j.version + 1})
        entity = record.model_copy(
            update={
                "version": next_j.version,
                "updated_at": utcnow(),
                "payload": next_j.model_dump(mode="json"),
            }
        )
        with self.ctx.db.session() as session:
            result = session.execute(
                update(EntityRow)
                .where(
                    EntityRow.kind == "JourneyRecord",
                    EntityRow.id == pid,
                    EntityRow.version == j.version,
                )
                .values(
                    version=entity.version,
                    updated_at=entity.updated_at,
                    data=entity.model_dump(mode="json"),
                )
            )
            if not isinstance(result, CursorResult) or result.rowcount != 1:
                raise ConflictError("El recorrido cambió. Recargá para continuar.")
        return next_j

    def checked(self, pid: str, version: int, states: set[str]) -> Journey:
        j = self.get(pid)
        if j.version != version:
            raise ConflictError("El recorrido cambió. Recargá para continuar.")
        if j.state not in states:
            raise ValidationError("Esta acción no está disponible en la etapa actual.")
        return j

    def interview(
        self, pid: str, version: int, message: str, facts: Facts | None, language: str = "es"
    ) -> Journey:
        j = self.checked(pid, version, {"interview", "unsupported"})
        if message:
            j.messages.append(Turn(role="user", text=message))
        if facts is not None:
            values = {**j.facts.model_dump(), **facts.model_dump(exclude_none=True)}
            j.facts = Facts.model_validate(values)
        j.confirmed = False
        j.pending = missing(j.facts)
        # Commit user input before touching the network. Failure cannot lose the answer.
        j = self.save(pid, j)
        if message and facts is None:
            try:
                out = self.ctx.llm.structured(
                    LLMPurpose.COPILOT,
                    Extraction,
                    LLMContext(
                        goal=self.wf.project(pid).goal,
                        constraints={
                            "facts": j.facts.model_dump(),
                            "language": language,
                            "conversation": [m.model_dump() for m in j.messages[-24:]],
                            "pending": j.pending,
                        },
                    ),
                    project=self.wf.project(pid),
                    prompt="journey",
                ).value
                values = j.facts.model_dump()
                for key, value in out.facts.model_dump(exclude_none=True).items():
                    # A literal quote is necessary to auto-record an explicit answer.
                    quote = out.evidence.get(key, "")
                    if quote.strip() and quote.casefold() in message.casefold():
                        values[key] = value
                j.facts = Facts.model_validate(values)
                j.assumptions = out.assumptions
                j.llm_failed = False
                if out.question and out.question_field in missing(j.facts):
                    j.messages.append(Turn(role="assistant", text=out.question))
            except Exception:
                j.llm_failed = True
        j.pending = missing(j.facts)
        j.state = (
            "unsupported"
            if j.facts.modality == "other" or j.facts.task in {"detection", "segmentation", "other"}
            else "interview"
        )
        return self.save(pid, j)

    def confirm(self, pid: str, version: int) -> Journey:
        j = self.checked(pid, version, {"interview"})
        if missing(j.facts):
            raise ValidationError("Completá los datos esenciales antes de confirmar.")
        if j.facts.modality != "image" or j.facts.task != "classification":
            raise ValidationError("Este recorrido admite clasificación de imágenes.")
        j.confirmed, j.state, j.stage = True, "awaiting_data", "awaiting_data"
        return self.save(pid, j)

    def _change(self, pid: str, **values: Any) -> Journey:
        for _ in range(8):
            j = self.get(pid)
            try:
                return self.save(pid, j.model_copy(update=values))
            except ConflictError:
                continue
        raise ConflictError("No se pudo actualizar el recorrido.")

    def _launch(self, pid: str, j: Journey, work: Any) -> Journey:
        j.owner_pid, j.owner_started = os.getpid(), PROCESS_STARTED
        j.job_id, j.child_job_id = None, None
        j.study_id = None
        j.error, j.notification, j.stop_requested = None, None, False
        j = self.save(pid, j)  # durable reservation BEFORE submit; CAS rejects double clicks
        gate = threading.Event()

        def run(job: JobContext) -> None:
            gate.wait()

            def cancel() -> None:
                self._change(pid, stop_requested=True)

            job.cancel_callback = cancel
            try:
                work(job)
            except Exception as exc:
                self._change(
                    pid,
                    state="failed",
                    stage="failed",
                    notification="failed",
                    error=str(exc)[:1000],
                )
                raise

        try:
            job = self.ctx.jobs.submit("journey", run, refs={"project_id": pid}, dedicated=True)
            j = self._change(pid, job_id=job.id)
        except Exception:
            self._change(pid, state="failed", notification="failed")
            raise
        finally:
            gate.set()
        return j

    def prepare(self, pid: str, version: int, source_id: str) -> Journey:
        from perceptron.api.routers.data import _source_path
        from perceptron.data.splits import SplitRequest
        from perceptron.domain.models import DataSource

        j = self.checked(
            pid,
            version,
            {"awaiting_data", "ready_to_train", "needs_improvement", "failed", "interrupted"},
        )
        if not j.confirmed:
            raise ValidationError("Confirmá primero el resumen.")
        source = self.ctx.repo(DataSource).get(source_id)
        if source.project_id != pid:
            raise ValidationError("La fuente pertenece a otro proyecto.")
        path = _source_path(self.ctx, source)
        j.state, j.stage = "reviewing_data", "reviewing_data"
        j.dataset_id, j.model_id, j.run_id = None, None, None
        j.contract, j.result, j.validations = None, None, []
        j.blockers, j.warnings = [], []

        def work(job: JobContext) -> None:
            split = (
                SplitRequest(strategy=SplitStrategy.GROUP, group_column="__image_group__")
                if j.facts.grouped
                else SplitRequest()
            )
            dv = self.wf.ingest(
                pid, path, split=split, source_record=source, modality=Modality.IMAGE
            )
            # Ingest already checked file integrity for every image. Inspect only that
            # metadata (never test labels or pixels) before profiling train/val.
            import polars as pl

            view = self.wf.view(dv)
            if dv.modality is Modality.IMAGE:
                index = pl.scan_parquet(view.data_file)
                if "corrupt" in index.collect_schema().names():
                    unreadable = (
                        index.filter(pl.col("corrupt")).select("path").collect()["path"].to_list()
                    )
                    if unreadable:
                        self._change(
                            pid,
                            dataset_id=dv.id,
                            state="awaiting_data",
                            stage="awaiting_data",
                            blockers=[
                                "Reemplazá las imágenes ilegibles y cargá otro ZIP: "
                                + ", ".join(unreadable[:10])
                            ],
                        )
                        return
            card = self.wf.profile(dv.id)
            if self.get(pid).stop_requested:
                self._change(
                    pid, state="interrupted", stage="interrupted", notification="interrupted"
                )
                return
            blockers, warnings = [], []
            if card.modality != Modality.IMAGE or card.vision_task:
                blockers.append(
                    "Usá imágenes con una categoría por imagen, sin máscaras ni recuadros."
                )
            classes = {str(c.value) for c in (card.target.classes or [])} if card.target else set()
            view = self.wf.view(dv)
            if dv.target:
                for part in ("train", "val"):
                    actual_classes = set(view.read(part)[dv.target].drop_nulls().to_list())
                    if actual_classes != set(j.facts.classes or []):
                        blockers.append(
                            "Agregá más ejemplos o grupos de cada categoría: "
                            "algunas no están representadas en aprendizaje y validación."
                        )
                        break
            hashes = self.known_images(dv.id)
            if len(set(hashes.values())) != len(hashes):
                blockers.append(
                    "Hay archivos idénticos. Conservá una copia por imagen en el ZIP "
                    "para no repetir ejemplos entre aprendizaje y evaluación."
                )
            if classes != set(j.facts.classes or []):
                blockers.append(
                    "Las carpetas deben coincidir con las categorías confirmadas: "
                    + ", ".join(j.facts.classes or [])
                )
            if any(not card.split_counts.get(s) for s in ("train", "val", "test")):
                blockers.append("Agregá ejemplos o grupos para separar aprendizaje y evaluación.")
            if card.images and card.images.corrupt:
                blockers.append(
                    f"Reemplazá {card.images.corrupt} imágenes ilegibles y volvé a cargar."
                )
            if card.target and any(c.count < 5 for c in card.target.classes or []):
                blockers.append("Agregá al menos cinco ejemplos utilizables por categoría.")
            if any(c.null_fraction > 0 for c in card.columns if c.name == dv.target):
                blockers.append("Poné cada imagen dentro de la carpeta de su categoría.")
            if card.images and card.images.near_duplicate_pairs:
                warnings.append(
                    "Hay imágenes similares. Agrupalas por objeto o sesión; "
                    "no son evidencia independiente."
                )
            if card.num_samples < 100:
                warnings.append(
                    "Hay pocos ejemplos: el resultado es exploratorio. "
                    "Validá con más objetos y condiciones nuevas."
                )
            self._change(
                pid,
                model_id=None,
                run_id=None,
                contract=None,
                result=None,
                validations=[],
                dataset_id=dv.id,
                blockers=blockers,
                warnings=warnings,
                state="awaiting_data" if blockers else "ready_to_train",
                stage="awaiting_data" if blockers else "ready_to_train",
            )

        return self._launch(pid, j, work)

    def train(self, pid: str, version: int) -> Journey:
        from perceptron.api.jobs import TERMINAL
        from perceptron.export.formats import ExportRequest
        from perceptron.hpo.study import StudyResult
        from perceptron.services.studies import new_study

        j = self.checked(pid, version, {"ready_to_train", "failed", "interrupted"})
        if not j.dataset_id or j.blockers or not j.confirmed:
            raise ValidationError("Revisá los datos antes de entrenar.")
        dataset_id = j.dataset_id
        j.state, j.stage = "training", "preparing"

        def work(job: JobContext) -> None:
            def stopped() -> bool:
                if self.get(pid).stop_requested:
                    self._change(
                        pid, state="interrupted", stage="interrupted", notification="interrupted"
                    )
                    return True
                return False

            if stopped():
                return
            pipeline = self.wf.propose_pipeline(dataset_id, pretrained=False)
            arch, _ = self.wf.propose_architecture(dataset_id, pipeline.id)
            budget = Budget(max_trials=2, max_epochs_per_trial=12, max_time_s=600)
            strategy = self.wf.hpo_strategy(arch.id, budget, dataset_version_id=j.dataset_id)
            study = new_study(self.ctx, pid, dataset_id, pipeline.id, arch.id, strategy)
            self._change(pid, study_id=study.id)
            if stopped():
                return
            child = self.ctx.launch_study(study)  # preserves remote quotas and worker selection
            self._change(pid, child_job_id=child.id, stage="queued")
            deadline = time.monotonic() + 660
            timed_out = False
            while True:
                current = self.ctx.jobs.get(child.id)
                if current is None:
                    raise ValidationError("Se perdió el trabajo de entrenamiento. Reintentá.")
                stop = self.get(pid).stop_requested
                if current.status == "running" and self.get(pid).stage == "queued":
                    self._change(pid, stage="learning")
                timed_out = timed_out or time.monotonic() > deadline
                if stop or timed_out:
                    self.ctx.jobs.cancel(child.id)
                if current.status in TERMINAL and (
                    current.status != "cancelled" or current.finished_at is not None
                ):
                    break
                time.sleep(0.5)
            if stopped():
                return
            if timed_out or current.status != "succeeded":
                raise ValidationError(
                    "El entrenamiento no terminó. Revisá los datos o "
                    "los recursos disponibles y reintentá."
                )
            result = StudyResult.model_validate(current.result)
            if result.best_trial is None:
                raise ValidationError(
                    "No se obtuvo un modelo válido. Revisá los datos y reintentá."
                )
            rid = result.best_trial.run_id
            self._change(pid, stage="evaluating", run_id=rid)
            _, report = self.wf.evaluate(rid)
            if stopped():
                return
            model = self.wf.register(rid)
            self._change(pid, stage="preparing_prediction")
            exported = self.wf.export(rid, ExportRequest())
            if not exported.ok:
                raise ValidationError(
                    "El modelo entrenó, pero no pudo prepararse para predicciones."
                )
            contract = model.signature
            if (
                contract.get("inputs", {}).get("kind") != "image"
                or contract.get("outputs", {}).get("task") != "classification"
            ):
                raise ValidationError("El contrato del modelo no es clasificación de imágenes.")
            c = report.classification
            if c is None:
                raise ValidationError("No hay una evaluación de clasificación.")
            matrix = c.confusion_matrix
            total = sum(sum(row) for row in matrix)
            correct = sum(matrix[i][i] for i in range(len(matrix)))
            positive = c.labels.index(j.facts.positive_class or "")
            misses = sum(matrix[positive]) - matrix[positive][positive]
            alarms = sum(row[positive] for row in matrix) - matrix[positive][positive]
            meets = bool(total and correct / total * 100 >= (j.facts.success_percent or 100))
            if stopped():
                return
            self._change(
                pid,
                state="ready_to_validate" if meets else "needs_improvement",
                stage="complete",
                model_id=model.id,
                contract=contract,
                notification="training_finished",
                result={
                    "total": total,
                    "correct": correct,
                    "misses": misses,
                    "false_alarms": alarms,
                    "meets_target": meets,
                },
            )

        return self._launch(pid, j, work)

    def stop(self, pid: str, version: int) -> Journey:
        j = self.checked(pid, version, ACTIVE)
        j.stop_requested = True
        if j.child_job_id:
            self.ctx.jobs.cancel(j.child_job_id)
        return self.save(pid, j)

    def contract(self, pid: str) -> Journey:
        j = self.get(pid)
        if (
            j.state not in {"ready_to_validate", "ready_to_use", "needs_improvement"}
            or not j.model_id
        ):
            raise ValidationError("Todavía no hay un modelo preparado para probar.")
        model = self.ctx.repo(ModelVersion).get(j.model_id)
        if model.project_id != pid or model.run_id != j.run_id or model.signature != j.contract:
            raise ValidationError("El contrato no corresponde al modelo de este recorrido.")
        if model.signature.get("inputs", {}).get("kind") != "image":
            raise ValidationError("El modelo no admite esta entrada.")
        return j
