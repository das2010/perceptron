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
            extra = f"  [pregunta: {said[-1].field}; opciones: {said[-1].options}]"
            log.append(f"ASISTENTE: {said[-1].text}" + (extra if said[-1].field else ""))
        log.append(
            f"   datos: {json.dumps(j.facts.model_dump(exclude_none=True), ensure_ascii=False)}"
        )
        if not j.pending or j.state != "interview":
            break
        # Como una persona: responde lo que le preguntaron, no lo que falta en la lista.
        asked = said[-1].field if said and said[-1].field else j.pending[0]
        reply = answer(asked, said[-1].text if said else "")
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


def test_questions_after_the_interview(coordinator: Coordinator) -> None:
    """Con el caso confirmado, la persona pregunta y pide cosas: respuestas con el estado real,
    sin jerga, y solo acciones permitidas (sin datos no se entrena)."""
    from perceptron.services.journey_guide import Facts

    project = coordinator.ctx.projects.add(Project(name="chat"))
    coordinator.ctx.files.init_project(project)
    facts = Facts(
        data_kind="table",
        problem="category",
        prediction="si el cliente se va a dar de baja",
        labels_available=True,
        error_costs="false_negative_worse",
        positive_class="se va",
        success_target=80,
    )
    j = coordinator.interview(project.id, 1, "Quiero saber qué clientes se van", facts)
    j = coordinator.confirm(project.id, j.version)
    log = [f"ASISTENTE: {j.messages[-1].text}"]
    for message in (
        "¿Puedo subir el Excel tal como sale del sistema, con varias hojas?",
        "¿Para qué necesitás tantos ejemplos?",
        "Dale, entrená ya",
    ):
        j = coordinator.ask(project.id, j.version, message)
        log += [
            f"PERSONA: {message}",
            f"ASISTENTE: {j.messages[-1].text} [acción: {j.messages[-1].action}]",
        ]
    folder = os.environ.get("PERCEPTRON_TRANSCRIPTS")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        provider = os.environ.get("PERCEPTRON_LLM__PROFILE", "llm")
        Path(folder, f"{provider}-chat.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    text = "\n".join(log)
    replies = [m for m in j.messages if m.role == "assistant"][-3:]
    assert all(m.text and not m.text.startswith("No pude consultar") for m in replies), text
    assert all(not JARGON.search(m.text) for m in replies), text
    assert j.state == "awaiting_data" and replies[-1].action is None, text


def test_wear_over_time_is_framed_not_rejected(coordinator: Coordinator) -> None:
    """Caso real: «desgaste en el tiempo». El asistente aclara si es estimar o pronosticar y
    nunca termina en «no se puede»."""
    answers = {
        "data_kind": "Una planilla con las mediciones de cada herramienta",
        "problem": "Quiero saber cuánto se van a gastar en las próximas semanas, tengo la "
        "historia de mediciones semanales",
        "prediction": "el desgaste semanal de cada herramienta",
        "labels_available": "Sí, tengo las mediciones de desgaste",
        "horizon": "las próximas 4 semanas",
        "extrapolate": "No, valores parecidos",
        "success_target": "Que no le erre por más de un 10 %",
    }
    j, log = converse(
        coordinator,
        "desgaste",
        "Quiero predecir cómo va a cambiar el desgaste en el tiempo",
        lambda field, _q: answers.get(field or "", "No sé, lo que recomiendes"),
    )
    text = "\n".join(log)
    assert j.state == "interview" and not j.unsupported, text
    assert j.facts.problem in ("forecast", "value"), text
    assert not j.pending, text


def test_photos_to_locate_defects(coordinator: Coordinator) -> None:
    answers = {
        "problem": "Que me marque dónde está cada fisura en la foto",
        "prediction": "dónde están las fisuras",
        "labels_available": "Sí, ya las marqué con recuadros en CVAT",
        "success_target": "Que encuentre 8 de cada 10",
    }
    j, log = converse(
        coordinator,
        "ubicar",
        "Tengo fotos de piezas y quiero encontrar las fisuras",
        lambda field, _q: answers.get(field or "", "No sé"),
    )
    text = "\n".join(log)
    assert not j.unsupported and not j.pending, text
    assert j.facts.data_kind == "images" and j.facts.problem in ("locate", "segment"), text


def test_files_first_the_assistant_proposes_the_goal(
    coordinator: Coordinator, fixtures_dir: Path
) -> None:
    """La persona sube la planilla antes de contar el caso: el asistente cuenta qué vio,
    propone primero lo que encaja con lo que dijo y la conversación sigue desde ahí."""
    from perceptron.domain.models import DataSource

    c = coordinator
    project = c.ctx.projects.add(Project(name="archivos"))
    path = fixtures_dir / "uc01_churn" / "churn.csv"
    src = c.ctx.repo(DataSource).add(
        DataSource(type="file", project_id=project.id, name=path.name, config={"path": str(path)})
    )
    j = c.read_files(project.id, 1, src.id, "Quiero saber qué clientes se van a ir")
    said = j.messages[-1]
    log = ["PERSONA: Quiero saber qué clientes se van a ir [sube churn.csv]"]
    titles = {g.id: g.title for g in j.goals}
    log.append(f"ASISTENTE: {said.text}  [objetivos: {[titles[i] for i in said.goals]}]")
    first = next(g for g in j.goals if g.id == said.goals[0])
    answers = {
        "error_costs": "Que se me escape uno que se va es peor",
        "positive_class": "Los que se van",
        "success_target": "8 de cada 10",
    }
    if first.target == "churn":
        log.append(f"PERSONA: [elige «{first.title}»]")
        j = c.choose(project.id, j.version, first.id)
        for _ in range(MAX_TURNS):
            if not j.pending:
                break
            asked = j.messages[-1].field or j.pending[0]
            log.append(f"ASISTENTE: {j.messages[-1].text}")
            reply = answers.get(asked, "No sé, lo que te parezca")
            log.append(f"PERSONA: {reply}")
            j = c.interview(project.id, j.version, reply)
        log.append(f"ASISTENTE: {j.messages[-1].text}")
    folder = os.environ.get("PERCEPTRON_TRANSCRIPTS")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        provider = os.environ.get("PERCEPTRON_LLM__PROFILE", "llm")
        Path(folder, f"{provider}-archivos.txt").write_text("\n".join(log) + "\n", "utf-8")
    text = "\n".join(log)
    assert len(said.goals) >= 2 and first.target == "churn", text
    assert "churn" in said.text or "cliente" in said.text.casefold(), text
    assert not JARGON.search(said.text), text
    assert not j.pending and j.facts.target_name == "churn" and j.data_ready, text
