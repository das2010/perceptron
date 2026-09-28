"""Roles del LLM sobre el flujo del proyecto (SPEC §7.7.4), siempre con fallback por reglas.

Todo lo que devuelven es una **propuesta** (ArchSpec guardada con origin=llm, estrategia
de HPO, diagnóstico, informe, etiquetas) que el usuario acepta, edita o descarta: nada se
aplica solo. Si el LLM no está disponible (L0, sin clave, presupuesto) o su salida no valida
tras los reintentos, se usa la regla equivalente de Capa 1 (RF-WIZ-03, RF-ARC-04).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from perceptron.archspec.schema import ArchSpec, Provenance
from perceptron.archspec.validate import ValidationReport, offline_mode, validate_archspec
from perceptron.catalog.registry import HF_TEXT_MODELS, TIMM_WEIGHTS, blocks_for
from perceptron.catalog.rules import recommend
from perceptron.core.errors import NotFoundError, ValidationError
from perceptron.data.profiling.card import ProfileCard
from perceptron.domain.enums import LabelKind, LLMPurpose, Modality, Origin, PrivacyLevel
from perceptron.domain.models import ArchSpecRecord, LabelSet, ModelVersion, Project, Run
from perceptron.hpo.recommend import recommend_strategy
from perceptron.hpo.strategy import (
    NOT_TUNED,
    Budget,
    HPOStrategy,
    SearchParam,
    default_search_space,
)
from perceptron.llm.errors import (
    LLMBudgetExceededError,
    LLMOutputInvalidError,
    LLMProviderError,
    LLMUnavailableError,
)
from perceptron.llm.privacy import LLMContext, RunSummary
from perceptron.llm.schemas import (
    ArchCandidate,
    ArchProposalSet,
    ClassGuide,
    Diagnosis,
    HPOProposal,
    LabelingGuide,
    ModelCard,
    PrelabelBatch,
    Report,
)
from perceptron.services.estimate import estimate_epoch_time
from perceptron.training.config import RESULT_FILE, RunResult
from perceptron.training.diagnostics import rules_diagnosis
from perceptron.training.hardware import HardwareReport, detect_hardware

if TYPE_CHECKING:
    from perceptron.llm.gateway import Gateway
    from perceptron.services.workflow import Workflow

logger = logging.getLogger(__name__)

Mode = Literal["auto", "llm", "rules"]
FALLBACK = (LLMUnavailableError, LLMBudgetExceededError, LLMProviderError, LLMOutputInvalidError)
MAX_HISTORY_POINTS = 60
PRELABEL_BATCH = 20
REPORT_DIR = "report"


def _reason(e: Exception) -> str:
    code = getattr(e, "code", type(e).__name__)
    return f"{code}: {getattr(e, 'message', str(e))}"


def hardware_summary(hw: HardwareReport) -> dict[str, Any]:
    return {
        "recommended_device": hw.recommended_device.value,
        "cpu_cores": hw.cpu.logical_cores,
        "ram_gb": round(hw.ram_total_gb, 1),
        "gpus": [{"name": g.name, "vram_gb": round(g.vram_total_gb, 1)} for g in hw.gpus],
    }


def _compact(block: dict[str, Any]) -> dict[str, Any]:
    """Bloque del catálogo sin textos largos: menos tokens por llamada (modelos locales)."""
    params = {
        name: {k: v for k, v in p.items() if k in ("type", "default", "low", "high", "choices")}
        for name, p in (block.get("params") or {}).items()
    }
    keep = ("key", "consumes", "produces", "multi_input")
    return {**{k: block[k] for k in keep if k in block}, "params": params}


def catalog_for(card: ProfileCard, task: Any, *, compact: bool = False) -> list[dict[str, Any]]:
    blocks = [b.public() for b in blocks_for(card.modality, task)]
    if compact:
        blocks = [_compact(b) for b in blocks]
    if card.modality in (Modality.IMAGE, Modality.AUDIO):
        weights = [w.model_dump() for w in TIMM_WEIGHTS.values() if w.commercial_ok]
        blocks.append({"key": "_pretrained_weights_timm", "weights": weights})
    if card.modality is Modality.TEXT:
        models = [m.model_dump() for m in HF_TEXT_MODELS.values() if m.commercial_ok]
        blocks.append({"key": "_pretrained_text_encoders", "models": models})
    return blocks


def downsample(
    history: list[dict[str, float]], n: int = MAX_HISTORY_POINTS
) -> list[dict[str, float]]:
    if len(history) <= n:
        return history
    step = len(history) / n
    picked = [history[int(i * step)] for i in range(n - 1)]
    return [*picked, history[-1]]


# ---------------------------------------------------------------------- resultados


@dataclass
class ArchOption:
    record: ArchSpecRecord
    title: str
    rationale: str
    validation: ValidationReport
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    confidence: float | None = None
    estimates: dict[str, float | None] = field(default_factory=dict)


@dataclass
class ArchProposals:
    options: list[ArchOption]
    origin: Origin
    llm_call_id: str | None = None
    fallback_reason: str | None = None


class LLMRoles:
    def __init__(self, wf: Workflow) -> None:
        self.wf = wf
        self.ctx = wf.ctx

    @property
    def gateway(self) -> Gateway:
        return self.ctx.llm

    def _skip(self, mode: Mode, purpose: LLMPurpose, project: Project) -> str | None:
        """Motivo para no usar el LLM, o None. En modo `llm` la indisponibilidad es un error."""
        if mode == "rules":
            return "modo reglas"
        try:
            self.gateway.resolve(purpose, project)
        except LLMUnavailableError as e:
            if mode == "llm":
                raise
            return _reason(e)
        return None

    # ------------------------------------------------------------------ arquitecto

    def propose_architectures(
        self,
        dataset_version_id: str,
        pipeline_id: str,
        *,
        mode: Mode = "auto",
        n: int = 3,
        device: str | None = None,
    ) -> ArchProposals:
        """2–4 propuestas validadas del LLM (RF-ARC-01/02) o la de reglas (RF-ARC-04)."""
        wf = self.wf
        dv = wf.dataset(dataset_version_id)
        project = wf.project(dv.project_id)
        card = wf.profile_card(dv.id)
        fitted = wf.fitted_pipeline(pipeline_id, dv.id)
        hw = detect_hardware(self.ctx.settings.workspace_dir)
        rec = recommend(card, fitted, hw)
        device = device or hw.recommended_device.value
        memory_gb = (
            max((g.vram_total_gb for g in hw.gpus), default=None)
            if device != "cpu"
            else hw.ram_available_gb
        )

        def rules(reason: str | None) -> ArchProposals:
            record = wf.save_archspec(project.id, rec.spec, origin=Origin.RULES)
            report = validate_archspec(rec.spec)
            option = ArchOption(
                record=record,
                title=rec.spec.name,
                rationale=rec.rationale,
                validation=report,
                estimates=self._estimates(rec.spec, report, card, device),
            )
            return ArchProposals([option], Origin.RULES, fallback_reason=reason)

        skip = self._skip(mode, LLMPurpose.ARCHITECT, project)
        if skip:
            return rules(skip)
        n = min(max(n, 2), 4)
        compact = self.gateway.compact(LLMPurpose.ARCHITECT, project)
        if compact:
            n = 2  # modelos locales: generar ArchSpecs completas es lo más caro

        def fix(c: ArchCandidate) -> ArchSpec:
            # input y task salen de los datos, no son decisión de diseño.
            return c.archspec.model_copy(update={"input": rec.spec.input, "task": rec.spec.task})

        def check(c: ArchCandidate) -> ValidationReport:
            return validate_archspec(fix(c), device_memory_gb=memory_gb)

        def validator(value: ArchProposalSet) -> str | None:
            errors = [
                f"Propuesta {i + 1} «{c.title}»:\n{r.feedback()}"
                for i, c in enumerate(value.proposals)
                if not (r := check(c)).valid
            ]
            if len(value.proposals) < 2:
                errors.append("Se piden al menos 2 propuestas distintas.")
            return "\n\n".join(errors) or None

        llm_ctx = LLMContext(
            goal=project.goal or None,
            card=card,
            hardware=hardware_summary(hw),
            pipeline=fitted.spec.model_dump(mode="json"),
            catalog=catalog_for(card, rec.spec.task.type, compact=compact),
            constraints={
                "base_archspec": rec.spec.model_dump(mode="json", exclude={"provenance"}),
                "allow_pretrained": not offline_mode(),
                "commercial_use": True,
                "device": device,
                "n_train": card.split_counts.get("train"),
            },
        )
        call_id: str | None = None
        reason: str | None = None
        try:
            result = self.gateway.structured(
                LLMPurpose.ARCHITECT,
                ArchProposalSet,
                llm_ctx,
                project=project,
                validator=validator,
                prompt_vars={"n_min": 2, "n_max": n},
            )
            candidates, call_id = result.value.proposals, result.call_id
        except LLMOutputInvalidError as e:
            last = e.last_value
            candidates = last.proposals if isinstance(last, ArchProposalSet) else []
            call_id, reason = e.details.get("call_id"), _reason(e)
        except FALLBACK as e:
            return rules(_reason(e))

        options: list[ArchOption] = []
        for c in candidates:
            spec = fix(c)
            report = validate_archspec(spec, device_memory_gb=memory_gb)
            if not report.valid:
                continue
            spec = spec.model_copy(
                update={
                    "provenance": Provenance(
                        origin=Origin.LLM, llm_call_id=call_id, rationale=c.rationale
                    )
                }
            )
            options.append(
                ArchOption(
                    record=wf.save_archspec(project.id, spec, origin=Origin.LLM),
                    title=c.title,
                    rationale=c.rationale,
                    validation=report,
                    pros=c.pros,
                    cons=c.cons,
                    risks=c.risks,
                    confidence=c.confidence,
                    estimates=self._estimates(spec, report, card, device),
                )
            )
        if not options:
            return rules(reason or "ninguna propuesta del LLM validó")
        return ArchProposals(options, Origin.LLM, llm_call_id=call_id, fallback_reason=reason)

    @staticmethod
    def _estimates(
        spec: ArchSpec, report: ValidationReport, card: ProfileCard, device: str
    ) -> dict[str, float | None]:
        n_train = int(card.split_counts.get("train") or card.num_samples)
        return {
            "num_params": float(report.num_params) if report.num_params is not None else None,
            "memory_mb": report.estimated_memory_mb,
            "epoch_time_s": estimate_epoch_time(spec, n_train, device=device),
        }

    # ------------------------------------------------------------------ estratega de HPO

    def hpo_strategy(
        self,
        archspec_id: str,
        budget: Budget,
        *,
        mode: Mode = "auto",
        dataset_version_id: str | None = None,
    ) -> HPOStrategy:
        """HPOStrategy del LLM validada contra la ArchSpec y el presupuesto (RF-HPO-02)."""
        record = self.ctx.repo(ArchSpecRecord).get(archspec_id)
        spec = ArchSpec.model_validate(record.spec)
        base = recommend_strategy(spec, budget)
        project = self.wf.project(record.project_id)
        skip = self._skip(mode, LLMPurpose.HPO_STRATEGIST, project)
        if skip:
            return base
        space = HPOSpace.of(spec)
        tunable, metrics = space.tunable, space.metrics
        card = self.wf.profile_card(dataset_version_id) if dataset_version_id else None

        def validator(p: HPOProposal) -> str | None:
            return "\n".join(space.errors(space.repair(p, budget)[0], budget)) or None

        llm_ctx = LLMContext(
            goal=project.goal or None,
            card=card,
            archspec=spec.model_dump(mode="json", exclude={"provenance"}),
            hardware=hardware_summary(detect_hardware(self.ctx.settings.workspace_dir)),
            constraints={
                "tunable": [p.model_dump(mode="json") for p in tunable.values()],
                "not_tuned": sorted(NOT_TUNED),
                "budget": budget.model_dump(mode="json"),
                "objective_metrics": sorted(metrics),
                "rules_strategy": base.model_dump(mode="json", exclude={"search_space"}),
            },
        )
        try:
            result = self.gateway.structured(
                LLMPurpose.HPO_STRATEGIST,
                HPOProposal,
                llm_ctx,
                project=project,
                validator=validator,
            )
        except FALLBACK as e:
            logger.info("estrategia de HPO por reglas", extra={"reason": _reason(e)})
            return base
        repaired, _ = space.repair(result.value, budget)
        strategy = space.build(repaired, budget, origin=Origin.LLM)
        return strategy.model_copy(update={"llm_call_id": result.call_id})

    # ------------------------------------------------------------------ diagnosticador

    def diagnose(self, run_id: str, *, mode: Mode = "auto", store: bool = True) -> Diagnosis:
        run = self.ctx.repo(Run).get(run_id)
        run_dir = self.ctx.settings.paths.project(run.project_id).run(run.id)
        path = run_dir / RESULT_FILE
        if not path.is_file():
            raise NotFoundError(f"el run {run_id} no tiene resultado todavía")
        result = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
        record = self.ctx.repo(ArchSpecRecord).get(run.archspec_id)
        spec = ArchSpec.model_validate(record.spec)
        card = self.wf.profile_card(run.dataset_version_id)
        imbalance = card.target.imbalance_ratio if card.target else None
        hps = {**spec.hyperparameters(), **run.hyperparams}
        max_epochs = int(hps.get("epochs") or 0) or None
        diagnosis = rules_diagnosis(
            result.history, max_epochs=max_epochs, imbalance_ratio=imbalance, hyperparameters=hps
        )
        project = self.wf.project(run.project_id)
        skip = self._skip(mode, LLMPurpose.DIAGNOSTICIAN, project)
        if not skip:
            tunable = set(spec.hyperparameters())

            def validator(d: Diagnosis) -> str | None:
                bad = [
                    a.target
                    for a in d.actions
                    if a.kind == "change_hparam" and a.target and a.target not in tunable
                ]
                return (
                    f"hiperparámetros inexistentes: {bad} (válidos: {sorted(tunable)})"
                    if bad
                    else None
                )

            llm_ctx = LLMContext(
                goal=project.goal or None,
                card=card,
                archspec=spec.model_dump(mode="json", exclude={"provenance"}),
                runs=[
                    RunSummary(
                        run_id=run.id,
                        status=result.status,
                        architecture=spec.name,
                        hyperparams=hps,
                        metrics=result.best_metrics,
                        history=downsample(result.history),
                    )
                ],
                evidence=[p.model_dump(mode="json") for p in diagnosis.problems],
                constraints={"hyperparameters": sorted(tunable), "max_epochs": max_epochs},
            )
            try:
                out = self.gateway.structured(
                    LLMPurpose.DIAGNOSTICIAN,
                    Diagnosis,
                    llm_ctx,
                    project=project,
                    validator=validator,
                )
                diagnosis = out.value.model_copy(
                    update={"origin": "llm", "llm_call_id": out.call_id}
                )
            except FALLBACK as e:
                logger.info("diagnóstico por reglas", extra={"reason": _reason(e)})
        if store:
            run.diagnosis = diagnosis.model_dump(mode="json")
            self.ctx.repo(Run).update(run)
        return diagnosis

    # ------------------------------------------------------------------ informante

    def report(self, run_id: str, *, mode: Mode = "auto", language: str = "español") -> Report:
        """Informe final explicado + model card (RF-EVL-06). Requiere la evaluación en test."""
        run = self.ctx.repo(Run).get(run_id)
        evaluation = self.wf.evaluation_report(run_id)
        record = self.ctx.repo(ArchSpecRecord).get(run.archspec_id)
        spec = ArchSpec.model_validate(record.spec)
        card = self.wf.profile_card(run.dataset_version_id)
        project = self.wf.project(run.project_id)
        report = rules_report(project, card, spec, evaluation.model_dump(mode="json"))
        skip = self._skip(mode, LLMPurpose.REPORTER, project)
        if not skip:
            llm_ctx = LLMContext(
                goal=project.goal or None,
                card=card,
                archspec=spec.model_dump(mode="json", exclude={"provenance"}),
                evaluation=evaluation.model_dump(mode="json", exclude={"curves"}),
                runs=[
                    RunSummary(
                        run_id=run.id,
                        status=run.status.value,
                        architecture=spec.name,
                        hyperparams=run.hyperparams,
                        metrics=run.metrics,
                    )
                ],
            )
            try:
                out = self.gateway.structured(
                    LLMPurpose.REPORTER,
                    Report,
                    llm_ctx,
                    project=project,
                    prompt_vars={"language": language},
                    validator=report_validator(language),
                )
                metrics = {k: float(v) for k, v in evaluation.metrics.items()}
                card_out = out.value.model_card.model_copy(update={"metrics": metrics})
                report = out.value.model_copy(
                    update={"origin": "llm", "llm_call_id": out.call_id, "model_card": card_out}
                )
            except FALLBACK as e:
                logger.info("informe por reglas", extra={"reason": _reason(e)})
        self._save_report(run, report)
        return report

    def _save_report(self, run: Run, report: Report) -> Path:
        folder = self.ctx.settings.paths.project(run.project_id).run(run.id) / REPORT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.md").write_text(report.markdown, encoding="utf-8")
        (folder / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        for mv in self.ctx.repo(ModelVersion).list(filters={"run_id": run.id}):
            mv.model_card = {**mv.model_card, **report.model_card.model_dump(mode="json")}
            self.ctx.repo(ModelVersion).update(mv)
        if run.mlflow_run_id:
            try:
                self.wf.tracker.log_artifact(run.mlflow_run_id, folder / "report.md", REPORT_DIR)
            except Exception:
                logger.warning("no se pudo subir el informe a MLflow", exc_info=True)
        return folder

    # ------------------------------------------------------------------ etiquetador

    def labeling_guide(
        self, project_id: str, classes: dict[str, str], *, mode: Mode = "auto"
    ) -> LabelingGuide:
        """Guía de etiquetado (RF-LBL-04). Sin LLM: la guía son las definiciones del usuario."""
        if not classes:
            raise ValidationError("hace falta al menos una clase con su descripción")
        project = self.wf.project(project_id)
        base = LabelingGuide(classes=[ClassGuide(name=k, definition=v) for k, v in classes.items()])
        if self._skip(mode, LLMPurpose.LABELER, project):
            return base

        def validator(g: LabelingGuide) -> str | None:
            names = {c.name for c in g.classes}
            missing = sorted(set(classes) - names)
            extra = sorted(names - set(classes))
            if missing or extra:
                return f"la guía debe cubrir exactamente estas clases: {sorted(classes)}"
            return None

        llm_ctx = LLMContext(goal=project.goal or None, system={"classes": classes})
        try:
            return self.gateway.structured(
                LLMPurpose.LABELER,
                LabelingGuide,
                llm_ctx,
                project=project,
                validator=validator,
                prompt="guide",
            ).value
        except FALLBACK as e:
            logger.info("guía de etiquetado sin LLM", extra={"reason": _reason(e)})
            return base

    def prelabel(
        self,
        dataset_version_id: str,
        guide: LabelingGuide,
        *,
        column: str | None = None,
        limit: int = 100,
        split: str = "train",
    ) -> LabelSet:
        """Pre-etiquetado de texto con el LLM (RF-LBL-02). Necesita ver muestras: L2 o L3."""
        wf = self.wf
        dv = wf.dataset(dataset_version_id)
        project = wf.project(dv.project_id)
        view = wf.view(dv)
        column = column or view.text_column
        if column is None:
            raise ValidationError("el pre-etiquetado con LLM necesita una columna de texto")
        res = self.gateway.resolve(LLMPurpose.LABELER, project)
        level = self.gateway.effective_level(project, local=res.provider.is_local)
        if level.rank < PrivacyLevel.L2.rank:
            raise LLMUnavailableError(
                "El pre-etiquetado envía muestras: requiere privacidad L2 o L3",
                details={"reason": "privacy", "level": level.value},
            )
        if level is PrivacyLevel.L2 and not self.gateway.policy.pii.handles_free_text:
            raise LLMUnavailableError(
                "En L2 el texto libre solo sale con un motor NER de PII configurado",
                details={"reason": "no_ner", "level": level.value},
            )
        frame = view.read(split).head(limit)
        texts = [str(t) for t in frame.get_column(column).to_list()]
        classes = [c.name for c in guide.classes]
        labels: list[dict[str, Any]] = []
        for start in range(0, len(texts), PRELABEL_BATCH):
            batch = {
                f"{split}:{i}": texts[i]
                for i in range(start, min(start + PRELABEL_BATCH, len(texts)))
            }
            labels += self._prelabel_batch(project, guide, classes, batch, column, level)
        labelset = LabelSet(dataset_version_id=dv.id, kind=LabelKind.CLASS, classes=classes)
        rel = Path("labels") / f"{labelset.id}.jsonl"
        path = self.ctx.settings.paths.project(project.id).root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in labels) + "\n", encoding="utf-8"
        )
        labelset.path = rel.as_posix()
        return self.ctx.repo(LabelSet).add(labelset)

    def _prelabel_batch(
        self,
        project: Project,
        guide: LabelingGuide,
        classes: list[str],
        batch: dict[str, str],
        column: str,
        level: PrivacyLevel,
    ) -> list[dict[str, Any]]:
        def validator(out: PrelabelBatch) -> str | None:
            errors = [
                f"'{p.label}' no es una clase de la guía"
                for p in out.labels
                if p.label not in classes
            ]
            ids = {p.sample_id for p in out.labels}
            if ids != set(batch):
                errors.append(f"hay que etiquetar exactamente estos sample_id: {sorted(batch)}")
            return "\n".join(errors) or None

        llm_ctx = LLMContext(
            samples=[{"sample_id": k, column: v} for k, v in batch.items()],
            text_fields=[column] if level is PrivacyLevel.L2 else [],
            system={"classes": classes, "guide": guide.model_dump(mode="json")},
        )
        out = self.gateway.structured(
            LLMPurpose.LABELER,
            PrelabelBatch,
            llm_ctx,
            project=project,
            validator=validator,
            prompt="prelabel",
        )
        return [
            {**p.model_dump(mode="json"), "origin": "llm", "llm_call_id": out.call_id}
            for p in out.value.labels
        ]


@dataclass
class HPOSpace:
    """Qué puede ajustar un estratega (LLM o agente) sobre una ArchSpec, y cómo se valida."""

    tunable: dict[str, SearchParam]
    metrics: set[str]

    @classmethod
    def of(cls, spec: ArchSpec) -> HPOSpace:
        metrics = {"val_loss", *(f"val_{m}" for m in spec.metrics)}
        if spec.training.early_stopping:
            metrics.add(spec.training.early_stopping.monitor)
        return cls({p.name: p for p in default_search_space(spec)}, metrics)

    def build(self, p: HPOProposal, budget: Budget, *, origin: Origin) -> HPOStrategy:
        """Estrategia final: defaults de la plantilla (trial 0) y presupuesto acotado."""
        space = [
            sp.model_copy(update={"default": self.tunable[sp.name].default})
            for sp in p.search_space
        ]
        epochs = p.max_epochs_per_trial or budget.max_epochs_per_trial
        if budget.max_epochs_per_trial and epochs:
            epochs = min(epochs, budget.max_epochs_per_trial)
        return HPOStrategy(
            strategy=p.strategy,
            pruner=p.pruner,
            search_space=space,
            objectives=p.objectives,
            budget=budget.model_copy(
                update={
                    "max_trials": min(p.max_trials, budget.max_trials),
                    "max_epochs_per_trial": epochs,
                }
            ),
            pruner_warmup_epochs=p.pruner_warmup_epochs,
            rationale=p.rationale,
            origin=origin,
        )

    def repair(self, p: HPOProposal, budget: Budget) -> tuple[HPOProposal, list[str]]:
        """Arregla lo que el sistema puede acotar sin decidir por el LLM (límites del catálogo).

        Rangos fuera de los límites se recortan, opciones inválidas se filtran, parámetros
        inexistentes se descartan, los trials se acotan al presupuesto y un objetivo
        desconocido pasa a `val_loss`. Lo que no se puede reparar lo informa `errors`.
        """
        notes: list[str] = []
        space: list[SearchParam] = []
        for sp in p.search_space:
            ref = self.tunable.get(sp.name)
            if ref is None:
                notes.append(f"se descartó '{sp.name}' (no es ajustable)")
                continue
            fixed = _clamp_param(sp, ref)
            if fixed != sp:
                notes.append(f"'{sp.name}' se acotó a los límites del catálogo")
            space.append(fixed)
        if not space:
            space = list(p.search_space)
        objectives = [
            o
            if o.metric in self.metrics or o.metric == "num_params"
            else o.model_copy(update={"metric": "val_loss", "direction": "minimize"})
            for o in p.objectives
        ]
        if objectives != p.objectives:
            notes.append("objetivo desconocido reemplazado por val_loss")
        trials = min(p.max_trials, budget.max_trials)
        if trials != p.max_trials:
            notes.append(f"max_trials acotado a {trials}")
        rationale = p.rationale + (f" [Sistema: {'; '.join(notes)}]" if notes else "")
        repaired = p.model_copy(
            update={
                "search_space": space,
                "objectives": objectives,
                "max_trials": trials,
                "rationale": rationale,
            }
        )
        return repaired, notes

    def errors(self, p: HPOProposal, budget: Budget) -> list[str]:
        errors: list[str] = []
        for sp in p.search_space:
            ref = self.tunable.get(sp.name)
            if ref is None:
                errors.append(f"'{sp.name}' no es un hiperparámetro ajustable de la ArchSpec")
                continue
            errors += _range_errors(sp, ref)
        errors += [
            f"objetivo '{o.metric}' desconocido (válidos: {sorted(self.metrics)})"
            for o in p.objectives
            if o.metric not in self.metrics and o.metric != "num_params"
        ]
        if p.max_trials > budget.max_trials:
            errors.append(f"max_trials {p.max_trials} supera el presupuesto ({budget.max_trials})")
        if not errors:
            try:
                self.build(p, budget, origin=Origin.LLM)
            except ValueError as e:
                errors.append(str(e))
        return errors


def _clamp_param(sp: SearchParam, ref: SearchParam) -> SearchParam:
    """Lleva un parámetro propuesto dentro del espacio del catálogo."""
    if ref.type == "categorical":
        choices = [c for c in (sp.choices or []) if c in (ref.choices or [])]
        return ref.model_copy(update={"choices": choices or ref.choices})
    if sp.type == "categorical":
        choices = [c for c in (sp.choices or []) if isinstance(c, int | float) and ref.accepts(c)]
        return sp.model_copy(update={"choices": choices}) if choices else ref
    lo, hi = float(ref.low or 0), float(ref.high or 0)
    low = min(max(float(sp.low if sp.low is not None else lo), lo), hi)
    high = min(max(float(sp.high if sp.high is not None else hi), lo), hi)
    if low > high or (ref.log and low <= 0):
        low, high = lo, hi
    return sp.model_copy(update={"type": ref.type, "low": low, "high": high, "log": ref.log})


def _range_errors(sp: SearchParam, ref: SearchParam) -> list[str]:
    errors: list[str] = []
    if ref.type == "categorical":
        extra = [c for c in (sp.choices or []) if c not in (ref.choices or [])]
        if sp.type != "categorical" or extra:
            errors.append(f"'{sp.name}': opciones válidas {ref.choices}")
        return errors
    if sp.type == "categorical":
        bad = [
            c for c in (sp.choices or []) if not isinstance(c, int | float) or not ref.accepts(c)
        ]
        if bad:
            errors.append(f"'{sp.name}': {bad} fuera de [{ref.low}, {ref.high}]")
        return errors
    lo, hi = float(ref.low or 0), float(ref.high or 0)
    if float(sp.low or 0) < lo or float(sp.high or 0) > hi:
        errors.append(f"'{sp.name}': el rango debe estar dentro de [{lo}, {hi}]")
    return errors


REPORT_SECTIONS = {
    "español": ["Resumen", "Datos", "Modelo", "Resultados", "Errores y límites", "Recomendaciones"],
    "english": ["Summary", "Data", "Model", "Results", "Errors and limitations", "Recommendations"],
}


def report_validator(language: str) -> Callable[[Report], str | None]:
    """El informe debe tener las secciones como encabezados Markdown (`## …`)."""
    sections = REPORT_SECTIONS.get(language, REPORT_SECTIONS["español"])

    def check(r: Report) -> str | None:
        missing = [
            name
            for name in sections
            if not re.search(
                rf"^#{{1,3}}\s*{re.escape(name)}", r.markdown, re.MULTILINE | re.IGNORECASE
            )
        ]
        if missing:
            return "El campo markdown debe usar encabezados Markdown: faltan " + ", ".join(
                f"'## {m}'" for m in missing
            )
        return None

    return check


def rules_report(
    project: Project, card: ProfileCard, spec: ArchSpec, evaluation: dict[str, Any]
) -> Report:
    """Informe sin LLM: plantilla con los números de la evaluación."""
    metrics = {k: float(v) for k, v in (evaluation.get("metrics") or {}).items()}
    rows = "\n".join(f"| {k} | {v:.4g} |" for k, v in sorted(metrics.items()))
    splits = ", ".join(f"{k}: {v}" for k, v in card.split_counts.items())
    goal = project.goal or "sin objetivo declarado"
    md = (
        f"# Informe — {project.name}\n\n## Resumen\n\nObjetivo: {goal}. Se entrenó "
        f"`{spec.name}` ({spec.task.type.value}) sobre {card.num_samples} muestras.\n\n"
        f"## Datos\n\nModalidad {card.modality.value}; particiones {splits}.\n\n"
        f"## Modelo\n\nArquitectura `{spec.name}` con {len(spec.nodes)} bloques.\n\n"
        f"## Resultados (test sellado)\n\n| Métrica | Valor |\n|---|---|\n{rows}\n\n"
        "## Errores y límites\n\nRevisá la matriz de confusión y las métricas por clase; "
        "el test es una muestra: el rendimiento en producción puede variar.\n"
    )
    return Report(
        title=f"Informe — {project.name}",
        summary=f"{spec.name}: " + ", ".join(f"{k}={v:.3g}" for k, v in list(metrics.items())[:4]),
        markdown=md,
        model_card=ModelCard(
            intended_use=goal,
            data=f"{card.num_samples} muestras ({card.modality.value}); {splits}",
            training=f"{spec.name}, optimizador {spec.optimizer.type}",
            metrics=metrics,
            limitations=["Evaluado en un único test sellado; validar con datos nuevos."],
        ),
    )
