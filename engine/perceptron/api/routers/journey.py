"""API del asistente guiado. La autenticación y el RBAC del servidor aplican como al resto.

Las predicciones usan los endpoints del Playground con `journey.run_id` (`predictRows`,
`predictTexts`, `predictFile`); acá quedan las acciones del recorrido y la validación con casos
nuevos, que compara contra el resultado real sin que ese dato entre al modelo.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.api.security import MB, read_limited
from perceptron.core.errors import ValidationError
from perceptron.services.journey import Coordinator, Journey, State, validation_key
from perceptron.services.journey_guide import Facts
from perceptron.services.workflow import Workflow

router = APIRouter(tags=["journey"])
Ctx = Annotated[EngineContext, Depends(get_context)]
BATCH_MAX = 500


class JourneyAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=1)
    action: Literal[
        "message", "files", "choose", "confirm", "prepare", "train", "stop", "acknowledge", "revise"
    ]
    message: str = Field(default="", max_length=4000)
    facts: Facts | None = None
    source_id: str | None = None
    goal_id: str | None = Field(default=None, max_length=20, description="Objetivo elegido")
    target: str | None = Field(default=None, description="Columna con el resultado (tablas)")
    language: Literal["es", "en"] = "es"


@router.get("/projects/{project_id}/journey", operation_id="getJourney")
def get_journey(project_id: str, ctx: Ctx) -> Journey:
    return Coordinator(Workflow(ctx)).get(project_id)


@router.post("/projects/{project_id}/journey", operation_id="actJourney")
def act_journey(project_id: str, body: JourneyAction, ctx: Ctx) -> Journey:
    c = Coordinator(Workflow(ctx))
    if body.action == "revise":
        return c.revise(project_id, body.version)
    if body.action == "message":
        if not body.message.strip() and body.facts is None:
            raise ValidationError("Escribí una respuesta o completá el dato que falta.")
        if c.get(project_id).state in ("interview", "unsupported"):
            return c.interview(project_id, body.version, body.message, body.facts, body.language)
        # Después de la entrevista se puede seguir conversando en cualquier etapa.
        if not body.message.strip():
            raise ValidationError("Escribí tu mensaje.")
        return c.ask(project_id, body.version, body.message, body.language)
    if body.action == "files":
        # Archivos antes de contar el caso: se leen y se proponen objetivos posibles.
        if not body.source_id:
            raise ValidationError("Subí primero los archivos.")
        return c.read_files(project_id, body.version, body.source_id, body.message, body.language)
    if body.action == "choose":
        if not body.goal_id:
            raise ValidationError("Elegí uno de los objetivos propuestos.")
        return c.choose(project_id, body.version, body.goal_id, body.language)
    if body.action == "confirm":
        return c.confirm(project_id, body.version)
    if body.action == "prepare":
        if not body.source_id:
            raise ValidationError("Subí primero los datos.")
        return c.prepare(project_id, body.version, body.source_id, body.target)
    if body.action == "train":
        return c.train(project_id, body.version)
    if body.action == "stop":
        return c.stop(project_id, body.version)
    j = c.checked(project_id, body.version, set(get_args(State)))
    j.notification = None
    return c.save(project_id, j)


class ValidationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row: dict[str, Any] | None = Field(default=None, description="Campos del caso (tablas)")
    text: str | None = Field(default=None, max_length=20_000)
    actual: str | float = Field(description="Resultado real, solo para comparar")


class GuidedValidation(BaseModel):
    prediction: Any
    actual: Any
    correct: bool
    checked: int = Field(default=1, description="Casos comparados en esta llamada")
    journey: Journey


@router.post("/projects/{project_id}/journey/validate", operation_id="validateJourney")
def validate_journey(project_id: str, body: ValidationCase, ctx: Ctx) -> GuidedValidation:
    """Un caso nuevo (fila o texto) con su resultado real: se predice sin el real y se compara."""
    c = Coordinator(Workflow(ctx))
    j = c.contract(project_id)
    if c.seen_case(j, row=body.row, digest=None):
        raise ValidationError(
            "Ese caso ya estaba en los datos que subiste: validá con un caso nuevo."
        )
    pred = c.predict(j, row=body.row, text=body.text)
    key = validation_key(row=body.row, text=body.text, data=None)
    j = c.record_validation(project_id, key, body.actual, pred.prediction, j.model_id)
    last = j.validations[-1]
    return GuidedValidation(
        prediction=last["prediction"], actual=last["actual"], correct=last["correct"], journey=j
    )


@router.post("/projects/{project_id}/journey/validate/file", operation_id="validateJourneyFile")
async def validate_journey_file(
    project_id: str,
    ctx: Ctx,
    file: Annotated[UploadFile, File()],
    actual: Annotated[str | None, Form()] = None,
) -> GuidedValidation:
    """Una imagen o un audio con su resultado real, o un CSV de casos nuevos con la columna del
    resultado (tablas y textos): se predice sin esa columna y se compara caso por caso."""
    c = Coordinator(Workflow(ctx))
    j = c.contract(project_id)
    data = await read_limited(file, 50 * MB)
    suffix = Path(file.filename or "").suffix.lower()
    kind = str((j.contract or {}).get("inputs", {}).get("kind"))
    if kind in ("tabular", "tokens"):
        return _validate_batch(c, project_id, j, data, kind)
    if kind == "sequence":
        return _validate_series(c, project_id, j, data, suffix, actual)
    if actual is None:
        raise ValidationError("Indicá el resultado real de este caso.")
    digest = validation_key(row=None, text=None, data=data)
    if c.seen_case(j, row=None, digest=digest):
        raise ValidationError(
            "Ese archivo ya estaba en los datos que subiste: validá con uno nuevo."
        )
    pred = c.predict(j, file=(data, suffix))
    j = c.record_validation(project_id, digest, actual, pred.prediction, j.model_id)
    last = j.validations[-1]
    return GuidedValidation(
        prediction=last["prediction"], actual=last["actual"], correct=last["correct"], journey=j
    )


def _validate_batch(
    c: Coordinator, project_id: str, j: Journey, data: bytes, kind: str
) -> GuidedValidation:
    target = j.target
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))[:BATCH_MAX]
    if not rows:
        raise ValidationError("El archivo no tiene casos.")
    if not target or target not in rows[0]:
        raise ValidationError(
            f"El archivo tiene que incluir la columna «{target}» con el resultado real."
        )
    text_col = next((k for k in rows[0] if k != target), None)
    last: dict[str, Any] = {}
    checked = 0
    for raw in rows:
        actual = raw.pop(target, None)
        if actual in (None, ""):
            continue
        if kind == "tokens":
            text = raw.get(text_col or "", "")
            pred = c.predict(j, text=text)
            key = validation_key(row=None, text=text, data=None)
        else:
            if c.seen_case(j, row=raw, digest=None):
                continue
            pred = c.predict(j, row=raw)
            key = validation_key(row=raw, text=None, data=None)
        j = c.record_validation(project_id, key, actual, pred.prediction, j.model_id)
        last = j.validations[-1]
        checked += 1
    if not checked:
        raise ValidationError(
            "No quedaron casos nuevos para comparar (o les falta el resultado real)."
        )
    return GuidedValidation(
        prediction=last["prediction"],
        actual=last["actual"],
        correct=last["correct"],
        checked=checked,
        journey=j,
    )


def _table_rows(data: bytes, suffix: str) -> list[dict[str, Any]]:
    """Una planilla subida (CSV o Excel) como filas con tipos (fechas y números)."""
    import tempfile

    from perceptron.data.sources.files import scan_table

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"datos{suffix or '.csv'}"
        path.write_bytes(data)
        try:
            df = scan_table(path).collect()
        except Exception as exc:
            raise ValidationError(
                "No se pudo leer la planilla. Probá con un CSV o un Excel."
            ) from exc
    if df.height == 0:
        raise ValidationError("La planilla no tiene filas.")
    return df.head(50_000).to_dicts()


def _validate_series(
    c: Coordinator, project_id: str, j: Journey, data: bytes, suffix: str, actual: str | None
) -> GuidedValidation:
    """Pronóstico: los últimos valores de cada serie quedan como «lo que pasó después» y se
    pronostican con lo anterior. Casos raros: la persona dice si lo marcado está bien."""
    rows = _table_rows(data, suffix)
    cfg = c.series_config(j)
    digest = validation_key(row=None, text=None, data=data)
    if _task_of(j) != "forecasting":
        if actual is None:
            raise ValidationError("Contame si los momentos que marcó como raros están bien.")
        preds = c.predict(j, rows=rows)
        found = sum(int(p.prediction or 0) for p in preds)
        j = c.record_validation(project_id, digest, actual, found, j.model_id)
        last = j.validations[-1]
        return GuidedValidation(
            prediction=found, actual=last["actual"], correct=last["correct"], journey=j
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(cfg.series_id)) if cfg.series_id else "", []).append(row)
    checked, last = 0, {}
    for name, items in groups.items():
        items.sort(key=lambda r: r.get(cfg.time_column) or 0)
        if len(items) <= cfg.horizon:
            continue
        history, future = items[: -cfg.horizon], items[-cfg.horizon :]
        preds = c.predict(j, rows=history)
        pred = next(
            (p for p in preds if str((p.details or {}).get("series") or "") == name), preds[0]
        )
        real = [r.get(cfg.target) for r in future]
        j = c.record_validation(project_id, f"{digest}:{name}", real, pred.prediction, j.model_id)
        last = j.validations[-1]
        checked += 1
    if not checked:
        raise ValidationError(
            f"Hacen falta más de {cfg.horizon} fechas por serie: las últimas se usan para "
            "comparar con lo que pronostica."
        )
    return GuidedValidation(
        prediction=last["prediction"],
        actual=last["actual"],
        correct=last["correct"],
        checked=checked,
        journey=j,
    )


def _task_of(j: Journey) -> str:
    return str((j.contract or {}).get("outputs", {}).get("task") or "")


class SeriesForecast(BaseModel):
    predictions: list[dict[str, Any]]
    task: str


@router.post("/projects/{project_id}/journey/predict/series", operation_id="predictJourneySeries")
async def predict_journey_series(
    project_id: str, ctx: Ctx, file: Annotated[UploadFile, File()]
) -> SeriesForecast:
    """La historia reciente (CSV o Excel con las mismas columnas que los datos): el pronóstico
    de los próximos valores o los momentos raros, por serie."""
    c = Coordinator(Workflow(ctx))
    j = c.contract(project_id)
    data = await read_limited(file, 50 * MB)
    rows = _table_rows(data, Path(file.filename or "").suffix.lower())
    preds = c.predict(j, rows=rows)
    return SeriesForecast(predictions=[p.to_dict() for p in preds], task=_task_of(j))
