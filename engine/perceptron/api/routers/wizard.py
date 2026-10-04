"""Wizard y copiloto (SPEC §10: `GET/PATCH /projects/{id}/draft`, `WS /projects/{id}/copilot`)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.api.streaming import WS_UNAUTHORIZED, ws_authorized
from perceptron.core.errors import PerceptronError
from perceptron.domain.models import ProjectDraft
from perceptron.llm.types import Message
from perceptron.services.brief import STEPS, BriefPatch, WizardPlan
from perceptron.services.wizard import DraftValues, Wizard
from perceptron.services.workflow import Workflow

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wizard"])
Ctx = Annotated[EngineContext, Depends(get_context)]


class DraftView(BaseModel):
    draft: ProjectDraft
    steps: list[str]
    values: DraftValues
    plan: WizardPlan = Field(description="Pasos, defaults y chequeos para este caso (ADR-0040)")


class IntakeTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class IntakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    history: list[IntakeTurn] = Field(default_factory=list, max_length=40)


class IntakeReply(BaseModel):
    patch: BriefPatch
    llm_call_id: str


class DraftUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1, description="Versión conocida (bloqueo optimista)")
    step: str | None = None
    values: dict[str, Any] = Field(default_factory=dict, description="Campos de DraftValues")
    origin: Literal["user", "copilot"] = "user"


def _view(wizard: Wizard, draft: ProjectDraft) -> DraftView:
    return DraftView(
        draft=draft,
        steps=list(STEPS),
        values=DraftValues.model_validate(draft.values),
        plan=wizard.plan(draft),
    )


@router.get("/projects/{project_id}/draft", operation_id="getDraft")
def get_draft(project_id: str, ctx: Ctx) -> DraftView:
    wizard = Wizard(Workflow(ctx))
    return _view(wizard, wizard.get(project_id))


@router.patch("/projects/{project_id}/draft", operation_id="updateDraft")
def update_draft(project_id: str, body: DraftUpdate, ctx: Ctx) -> DraftView:
    wizard = Wizard(Workflow(ctx))
    draft = wizard.update(
        project_id, version=body.version, values=body.values, step=body.step, origin=body.origin
    )
    return _view(wizard, draft)


@router.post("/projects/{project_id}/draft/intake", operation_id="draftIntake")
def draft_intake(project_id: str, body: IntakeRequest, ctx: Ctx) -> IntakeReply:
    """Entrevista del paso Objetivo (ADR-0040): cambios propuestos a la ficha y próxima pregunta.

    No modifica el borrador: la persona acepta los cambios con `PATCH /draft` (`brief`).
    Sin LLM utilizable responde 503 `llm_unavailable` y la ficha se completa a mano.
    """
    history = [Message(role=t.role, content=t.content) for t in body.history]
    patch, call_id = Wizard(Workflow(ctx)).intake(project_id, body.message, history)
    return IntakeReply(patch=patch, llm_call_id=call_id)


def _produce(
    wizard: Wizard,
    project_id: str,
    question: str,
    history: list[Message],
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any]],
) -> None:
    """Corre en un hilo: el Gateway es síncrono; los eventos vuelven al loop del WebSocket."""

    def put(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    pieces: list[str] = []
    try:
        for piece in wizard.chat(project_id, question, history):
            pieces.append(piece)
            put({"type": "token", "text": piece})
        answer = "".join(pieces)
        put({"type": "done", "text": answer})
        patch, call_id = wizard.suggest_patch(project_id, question, answer)
        if patch.changes:
            put({"type": "patch", "patch": patch.model_dump(mode="json"), "llm_call_id": call_id})
    except PerceptronError as e:
        put({"type": "error", "code": e.code, "message": e.message})
    except Exception:  # el copiloto nunca tira abajo la conexión
        logger.exception("el copiloto falló", extra={"project_id": project_id})
        put({"type": "error", "code": "internal_error", "message": "error interno del copiloto"})
    finally:
        put({"type": "_end"})


@router.websocket("/projects/{project_id}/copilot")
async def copilot(ws: WebSocket, project_id: str) -> None:
    """Copiloto del wizard (RF-LLM-05).

    Cliente → `{"type": "message", "text": "..."}`. Servidor → `token` (fragmentos de la
    respuesta), `done` (respuesta completa), `patch` (cambios sugeridos al borrador,
    aceptables/rechazables) o `error` (`code` estable, p. ej. `llm_unavailable`).
    """
    ctx: EngineContext = ws.app.state.ctx
    token = ctx.settings.api.token
    if not ws_authorized(ws, token.get_secret_value() if token else None):
        await ws.close(code=WS_UNAUTHORIZED)
        return
    await ws.accept()
    wizard = Wizard(Workflow(ctx))
    loop = asyncio.get_running_loop()
    history: list[Message] = []
    try:
        while True:
            msg = await ws.receive_json()
            if not isinstance(msg, dict) or msg.get("type") != "message":
                continue
            question = str(msg.get("text", "")).strip()
            if not question:
                continue
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            threading.Thread(
                target=_produce,
                args=(wizard, project_id, question, list(history), loop, queue),
                daemon=True,
            ).start()
            answer = ""
            while True:
                event = await queue.get()
                if event["type"] == "_end":
                    break
                if event["type"] == "done":
                    answer = str(event["text"])
                await ws.send_json(event)
            if answer:
                history += [
                    Message(role="user", content=question),
                    Message(role="assistant", content=answer),
                ]
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(RuntimeError):
            await ws.close()


@router.post("/projects/{project_id}/draft/reconcile", operation_id="draftReconcile")
def draft_reconcile(project_id: str, ctx: Ctx) -> IntakeReply:
    """Compara la ficha con el perfil de los datos (ADR-0040, fase 2).

    Propone correcciones o preguntas cuando lo declarado no coincide con lo que se mide; no
    modifica el borrador (se acepta con `PATCH /draft`). Sin datos elegidos, 422; sin LLM, 503.
    """
    patch, call_id = Wizard(Workflow(ctx)).reconcile(project_id)
    return IntakeReply(patch=patch, llm_call_id=call_id)
