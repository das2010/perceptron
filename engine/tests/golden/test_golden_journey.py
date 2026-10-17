"""Golden tests del asistente guiado con proveedores reales (ADR-0042, `llm.yml`).

Personas simuladas responden en lenguaje natural, como lo haría alguien sin conocimientos de
ML. Se verifica que la conversación llegue a un caso completo y correcto en pocos turnos y sin
jerga. Cada conversación se guarda en `PERCEPTRON_TRANSCRIPTS` para leerla.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from perceptron.api.context import EngineContext
from perceptron.domain.models import Project
from perceptron.llm.gateway import Gateway
from perceptron.services.journey import Coordinator
from perceptron.services.workflow import Workflow
from perceptron.tracking.tracker import MemoryTracker

pytestmark = [pytest.mark.golden, pytest.mark.timeout(1800)]

JARGON = re.compile(
    r"\b(features?|epochs?|épocas?|dataset|arquitectura|hiperpar\w*|target|label|"
    r"overfitting|regresi[oó]n|clasificaci[oó]n|clase positiva|falsos? negativos?)\b",
    re.IGNORECASE,
)
MAX_TURNS = 10


@pytest.fixture
def coordinator(ctx: EngineContext, monkeypatch: pytest.MonkeyPatch) -> Coordinator:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    settings = ctx.settings.llm.model_copy(update={"enabled": True, "cache": False})
    ctx.use_llm(Gateway.from_settings(ctx.settings.model_copy(update={"llm": settings}), ctx.db))
    return Coordinator(Workflow(ctx, MemoryTracker()))


Answer = Callable[[str | None, str], str]


def converse(c: Coordinator, name: str, first: str, answer: Answer) -> tuple[Any, list[str]]:
    """La persona escribe `first` y después responde cada pregunta con `answer(campo, texto)`."""
    project = c.ctx.projects.add(Project(name=name))
    j = c.interview(project.id, 1, first)
    log = [f"PERSONA: {first}"]
    for _ in range(MAX_TURNS):
        said = [m for m in j.messages if m.role == "assistant"]
        if said:
            log.append(f"ASISTENTE: {said[-1].text}")
        log.append(
            f"   datos: {json.dumps(j.facts.model_dump(exclude_none=True), ensure_ascii=False)}"
        )
        if not j.pending or j.state != "interview":
            break
        reply = answer(j.pending[0], said[-1].text if said else "")
        log.append(f"PERSONA: {reply}")
        j = c.interview(project.id, j.version, reply)
    folder = os.environ.get("PERCEPTRON_TRANSCRIPTS")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        provider = os.environ.get("PERCEPTRON_LLM__PROFILE", "llm")
        Path(folder, f"{provider}-{name}.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    return j, log


def assistant_texts(j: Any) -> list[str]:
    return [m.text for m in j.messages if m.role == "assistant"]


def test_churn_with_a_spreadsheet(coordinator: Coordinator) -> None:
    answers = {
        "data_kind": "Es un Excel que exportamos del sistema",
        "problem": "Saber si se va o no",
        "prediction": "Si el cliente se va a dar de baja el mes que viene",
        "labels_available": "Sí, tengo una columna que dice si se fue",
        "error_costs": "Prefiero que me avise de más; perder un cliente sin enterarme es peor",
        "positive_class": "Los que se van",
        "success_target": "Con que acierte 8 de cada 10 me sirve",
    }
    j, log = converse(
        coordinator,
        "churn",
        "Hola, tengo los datos de mis clientes y quiero saber cuáles se van a ir",
        lambda field, _q: answers.get(field or "", "No sé, lo que te parezca mejor"),
    )
    text = "\n".join(log)
    assert not j.pending, text
    assert j.facts.data_kind == "table" and j.facts.problem == "category", text
    assert j.facts.error_costs == "false_negative_worse", text
    assert all(not JARGON.search(t) for t in assistant_texts(j)), text


def test_house_prices_said_all_at_once(coordinator: Coordinator) -> None:
    """Quien cuenta todo de entrada no debería tener que repetirlo."""
    answers = {
        "extrapolate": "No, son casas parecidas a las que ya tengo",
        "success_target": "Si le erra por un 10 % está bien",
        "labels_available": "Sí, son ventas que ya se hicieron",
    }
    j, log = converse(
        coordinator,
        "casas",
        "Tengo una planilla con 3000 casas vendidas, con superficie, ambientes, barrio y el "
        "precio de venta. Quiero estimar a cuánto se va a vender una casa nueva.",
        lambda field, _q: answers.get(field or "", "No sé"),
    )
    text = "\n".join(log)
    assert not j.pending, text
    assert j.facts.data_kind == "table" and j.facts.problem == "value", text
    assert len(assistant_texts(j)) <= 4, text  # no vuelve a preguntar lo que ya se dijo


def test_cracks_in_photos_with_a_question_back(coordinator: Coordinator) -> None:
    """La persona hace una pregunta en vez de responder: el asistente la contesta y sigue."""
    answers = {
        "problem": "Que me diga si la pieza tiene fisura o no",
        "prediction": "si la pieza tiene fisura",
        "labels_available": "Sí, ya las separé en dos carpetas",
        "classes": "con fisura y sin fisura",
        "error_costs": "¿Qué querés decir con eso? No entiendo",
        "positive_class": "con fisura",
        "grouped": "Sí, saco 3 fotos de cada pieza",
        "success_target": "95 %",
    }
    asked: list[str] = []

    def answer(field: str | None, _q: str) -> str:
        if field == "error_costs" and asked:
            return "Ah, entiendo. Es peor que se escape una pieza fisurada"
        if field == "error_costs":
            asked.append(field)
        return answers.get(field or "", "No sé")

    j, log = converse(coordinator, "fisuras", "Quiero revisar fotos de piezas", answer)
    text = "\n".join(log)
    assert not j.pending, text
    assert j.facts.data_kind == "images" and j.facts.grouped is True, text
    assert j.facts.error_costs == "false_negative_worse", text
