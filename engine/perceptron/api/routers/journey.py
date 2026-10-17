"""Guided journey API. Router-wide authentication/RBAC applies as for the advanced API."""

from __future__ import annotations

import io
from typing import Annotated, Literal, get_args

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.api.security import MB, read_limited
from perceptron.core.errors import ValidationError
from perceptron.services.journey import Coordinator, Facts, Journey, State
from perceptron.services.workflow import Workflow

router = APIRouter(tags=["journey"])
Ctx = Annotated[EngineContext, Depends(get_context)]


class JourneyAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=1)
    action: Literal["message", "confirm", "prepare", "train", "stop", "acknowledge", "revise"]
    message: str = Field(default="", max_length=4000)
    facts: Facts | None = None
    source_id: str | None = None
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
            raise ValidationError("Escribí una respuesta o completá un dato esencial.")
        return c.interview(project_id, body.version, body.message, body.facts, body.language)
    if body.action == "confirm":
        return c.confirm(project_id, body.version)
    if body.action == "prepare":
        if not body.source_id:
            raise ValidationError("Cargá primero las imágenes.")
        return c.prepare(project_id, body.version, body.source_id)
    if body.action == "train":
        return c.train(project_id, body.version)
    if body.action == "stop":
        return c.stop(project_id, body.version)
    j = c.checked(project_id, body.version, set(get_args(State)))
    j.notification = None
    return c.save(project_id, j)


class GuidedPrediction(BaseModel):
    prediction: str
    actual: str | None = None
    correct: bool | None = None
    journey: Journey


@router.post("/projects/{project_id}/journey/predict", operation_id="predictJourney")
async def predict_journey(
    project_id: str,
    ctx: Ctx,
    file: Annotated[UploadFile, File()],
) -> GuidedPrediction:
    return await _predict_journey(project_id, ctx, file)


@router.post("/projects/{project_id}/journey/validate", operation_id="validateJourney")
async def validate_journey(
    project_id: str,
    ctx: Ctx,
    file: Annotated[UploadFile, File()],
    actual: Annotated[str, Form()],
) -> GuidedPrediction:
    return await _predict_journey(project_id, ctx, file, actual)


async def _predict_journey(
    project_id: str,
    ctx: Ctx,
    file: Annotated[UploadFile, File()],
    actual: Annotated[str | None, Form()] = None,
) -> GuidedPrediction:
    from PIL import Image, UnidentifiedImageError

    c = Coordinator(Workflow(ctx))
    j = c.contract(project_id)
    if not j.run_id:
        raise ValidationError("No hay un modelo asociado al recorrido.")
    prediction_model_id = j.model_id
    classes = (j.contract or {}).get("outputs", {}).get("classes", [])
    if actual is not None and actual not in classes:
        raise ValidationError("El resultado real debe ser una categoría del modelo.")
    data = await read_limited(file, 20 * MB)
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    if actual is not None and j.dataset_id and digest in c.known_images(j.dataset_id).values():
        raise ValidationError(
            "Esta imagen ya estaba en los datos originales. "
            "Validá con una imagen nueva de otro objeto o sesión."
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            # Ground truth NEVER enters inference, preprocessing, or the LLM.
            model = c.wf.inference_model(j.run_id)
            if any(model.signature.get(key) != value for key, value in (j.contract or {}).items()):
                raise ValidationError("La firma del archivo no coincide con el modelo registrado.")
            prediction = model.predict_images([image])[0]
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValidationError("No se pudo leer la imagen. Probá un PNG o JPG válido.") from exc
    label = str(prediction.prediction)
    if label not in classes:
        raise ValidationError("La predicción no corresponde a las categorías del modelo.")
    if actual is not None:
        # One vote per image, even after reconnect/retry. No original image is retained.
        j = c.contract(project_id)
        if j.model_id != prediction_model_id:
            from perceptron.core.errors import ConflictError

            raise ConflictError("El modelo cambió durante la predicción. Probá la imagen de nuevo.")
        j.validations = [v for v in j.validations if v["image_hash"] != digest]
        j.validations.append(
            {
                "image_hash": digest,
                "actual": actual,
                "prediction": label,
                "correct": actual == label,
            }
        )
        total = len(j.validations)
        correct = sum(v["correct"] for v in j.validations)
        covered = {v["actual"] for v in j.validations} == set(classes)
        if total >= 10 and covered:
            meets = correct / total * 100 >= (j.facts.success_percent or 100)
            j.state = (
                "ready_to_use"
                if meets and (j.result or {}).get("meets_target")
                else "needs_improvement"
            )
        j = c.save(project_id, j)
    return GuidedPrediction(
        prediction=label,
        actual=actual,
        correct=actual == label if actual is not None else None,
        journey=j,
    )
