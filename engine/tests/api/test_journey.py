"""Asistente guiado de punta a punta (todas las modalidades del recorrido)."""

from __future__ import annotations

import csv
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from perceptron.core.errors import ConflictError, ValidationError
from perceptron.domain.models import DataSource, Project, ProjectDraft
from perceptron.services.journey import Coordinator, FactChange, apply_changes
from perceptron.services.journey_guide import Facts, data_guide, missing, unsupported_reason
from perceptron.services.workflow import Workflow

CHURN = Facts(
    data_kind="table",
    problem="category",
    prediction="si el cliente se va a dar de baja",
    labels_available=True,
    error_costs="false_negative_worse",
    positive_class="1",
    success_target=50,
    target_name="churn",
)


def start(ctx: Any, facts: Facts = CHURN, name: str = "Bajas") -> tuple[str, Coordinator, Any]:
    p = ctx.projects.add(Project(name=name))
    ctx.files.init_project(p)
    c = Coordinator(Workflow(ctx))
    j = c.interview(p.id, 1, "Quiero anticipar qué clientes se van a dar de baja", facts)
    return p.id, c, c.confirm(p.id, j.version)


def wait(c: Coordinator, pid: str, timeout: float = 300) -> Any:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        j = c.get(pid)
        if j.state not in {"reviewing_data", "training"}:
            return j
        time.sleep(0.2)
    raise AssertionError("el recorrido no terminó")


# ---------------------------------------------------------------------- reglas puras


def test_what_to_ask_depends_on_the_case() -> None:
    assert missing(Facts()) == ["data_kind", "problem"]
    table_value = Facts(data_kind="table", problem="value")
    assert missing(table_value) == [
        "prediction",
        "labels_available",
        "extrapolate",
        "success_target",
    ]
    photos = Facts(data_kind="images", problem="category")
    assert {"classes", "error_costs", "grouped"} <= set(missing(photos))
    worse = photos.model_copy(update={"error_costs": "false_positive_worse", "classes": ["a", "b"]})
    assert "positive_class" in missing(worse)
    assert not missing(CHURN)


def test_unsupported_cases_point_to_advanced_mode() -> None:
    assert unsupported_reason(Facts(data_kind="table", problem="forecast"))
    assert unsupported_reason(Facts(data_kind="images", problem="value"))
    assert unsupported_reason(Facts(data_kind="other", problem="category"))
    assert "Etiquetado" in (
        unsupported_reason(CHURN.model_copy(update={"labels_available": False})) or ""
    )
    assert unsupported_reason(CHURN) is None


def test_data_guide_uses_the_words_of_the_case() -> None:
    house = Facts(
        data_kind="table",
        problem="value",
        target_name="precio",
        input_names=["superficie", "ambientes", "barrio"],
        extrapolate=True,
    )
    g = data_guide(house)
    assert g.layout == "table" and ".csv" in g.accept
    assert g.example_columns == ["superficie", "ambientes", "barrio", "precio"]
    assert g.template_csv and g.template_csv.splitlines()[0] == "superficie,ambientes,barrio,precio"
    assert any("rango" in t for t in g.tips)  # extrapolar: cubrir el rango
    photos = data_guide(
        Facts(data_kind="images", problem="category", classes=["fisura", "sano"], grouped=True)
    )
    assert photos.layout == "folders" and photos.accept == [".zip"]
    assert photos.example_tree[0] == "fisura/objeto_01/foto_01.jpg"


def test_llm_facts_need_a_literal_quote() -> None:
    said = "Tengo una planilla y quiero saber el precio de cada casa"
    facts = apply_changes(
        Facts(),
        [
            FactChange(field="data_kind", value="table", evidence="Tengo una planilla"),
            FactChange(field="problem", value="value", evidence="el precio de cada casa"),
            FactChange(field="success_target", value=10, evidence="un error del 10 %"),  # no dicho
            FactChange(
                field="error_costs", value="nada", evidence="Tengo una planilla"
            ),  # inválido
        ],
        said,
    )
    assert facts.data_kind == "table" and facts.problem == "value"
    assert facts.success_target is None and facts.error_costs is None


# ---------------------------------------------------------------------- API y entrevista


def test_interview_without_llm_keeps_the_answer_and_offers_controls(client: Any, ctx: Any) -> None:
    p = ctx.projects.add(Project(name="Prueba"))
    url = f"/api/v1/projects/{p.id}/journey"
    r = client.post(url, json={"action": "message", "version": 1, "message": "Tengo fotos"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["llm_failed"] and j["messages"][0]["text"] == "Tengo fotos"
    assert j["pending"][0] == "data_kind"  # la UI muestra el control de ese dato
    assert client.get(url).json() == j
    assert client.post(url, json={"action": "confirm", "version": j["version"]}).status_code == 422
    stale = client.post(url, json={"action": "message", "version": 1, "message": "otro"})
    assert stale.status_code == 409


def test_llm_interview_iterates_and_confirm_fills_the_wizard_brief(ctx: Any, fake_llm: Any) -> None:
    p = ctx.projects.add(Project(name="Casas"))
    c = Coordinator(Workflow(ctx))
    fake_llm.script(
        "copilot",
        {
            "changes": [
                {"field": "data_kind", "value": "table", "evidence": "tengo una planilla"},
                {"field": "problem", "value": "value", "evidence": "estimar el precio"},
                {"field": "prediction", "value": "precio de venta", "evidence": "el precio"},
            ],
            "question": "¿Ya tenés el precio real de cada casa en la planilla?",
            "question_field": "labels_available",
        },
    )
    j = c.interview(p.id, 1, "Hola, tengo una planilla de casas y quiero estimar el precio")
    assert j.facts.data_kind == "table" and j.facts.problem == "value"
    assert j.messages[-1].role == "assistant" and "precio real" in j.messages[-1].text
    assert j.pending[0] == "labels_available"
    # Los controles completan lo que falta (sin LLM) y aparece el resumen para confirmar.
    j = c.interview(
        p.id,
        j.version,
        "Sí, tengo los precios. Un 10 % de error está bien y no voy a extrapolar.",
        Facts(labels_available=True, extrapolate=False, success_target=10, target_name="precio"),
    )
    assert not j.pending and j.messages[-1].role == "assistant"
    assert "querés estimar esto: precio de venta" in j.messages[-1].text
    assert j.messages[-1].action == "confirm"
    j = c.confirm(p.id, j.version)
    assert j.state == "awaiting_data" and j.guide and j.guide.template_csv
    [draft] = ctx.repo(ProjectDraft).list(filters={"project_id": p.id})
    brief = draft.values["brief"]
    assert brief["problem"] == "value" and brief["extrapolate"] is False
    assert brief["prediction"] == "precio de venta"
    assert ctx.projects.get(p.id).goal.startswith("Hola, tengo una planilla")


def test_without_llm_the_assistant_always_answers_in_words(ctx: Any) -> None:
    """Sin proveedor de IA la conversación sigue: preguntas redactadas y acuse de cada respuesta."""
    p = ctx.projects.add(Project(name="Bajas"))
    c = Coordinator(Workflow(ctx))
    j = c.interview(p.id, 1, "Quiero saber qué clientes se van a ir")
    first = j.messages[-1]
    assert first.role == "assistant" and first.field == "data_kind" and first.step == "case"
    assert first.text.startswith("¡Hola!") and "planilla" in first.text
    # Respuesta con el control: el asistente acusa y pregunta lo siguiente.
    j = c.interview(p.id, j.version, "Una tabla (planilla, CSV, Excel)", Facts(data_kind="table"))
    last = j.messages[-1]
    assert [m.role for m in j.messages] == ["user", "assistant", "user", "assistant"]
    assert last.field == "problem" and last.text.split()[0] in {
        "Perfecto.",
        "Entendido.",
        "Bien.",
        "Anotado.",
    }
    j = c.interview(p.id, j.version, "", Facts(problem="forecast"))
    assert j.state == "unsupported" and j.messages[-1].action == "advanced"


def test_llm_reply_answers_doubts_and_offers_quick_answers(ctx: Any, fake_llm: Any) -> None:
    p = ctx.projects.add(Project(name="Fisuras"))
    c = Coordinator(Workflow(ctx))
    fake_llm.script(
        "copilot",
        {
            "reply": "Buena pregunta: un modelo aprende de tus fotos a reconocer fisuras.",
            "changes": [{"field": "data_kind", "value": "images", "evidence": "fotos"}],
            "question": "¿Qué querés que te diga de cada foto?",
            "question_field": "problem",
            "options": ["Si tiene fisura o no", "Qué tipo de defecto tiene"],
        },
        {  # pregunta por algo ya respondido: se usa la pregunta redactada del dato que falta
            "reply": "Entendido.",
            "changes": [{"field": "problem", "value": "category", "evidence": "si tiene fisura"}],
            "question": "¿Tenés fotos?",
            "question_field": "data_kind",
        },
    )
    j = c.interview(p.id, 1, "Tengo fotos de piezas, ¿qué es un modelo?")
    last = j.messages[-1]
    assert last.text.startswith("Buena pregunta") and last.text.endswith("de cada foto?")
    assert last.field == "problem" and last.options == [
        "Si tiene fisura o no",
        "Qué tipo de defecto tiene",
    ]
    j = c.interview(p.id, j.version, "Quiero saber si tiene fisura")
    last = j.messages[-1]
    assert last.field == j.pending[0] == "prediction" and last.text.startswith("Entendido.")
    assert "Contame con tus palabras" in last.text and not last.options


def test_conversation_continues_after_the_interview(ctx: Any, fake_llm: Any) -> None:
    pid, c, j = start(ctx)
    assert j.messages[-1].step == "data" and "necesito tus datos" in j.messages[-1].text
    # Sin respuesta del LLM: dice qué se puede hacer ahora, sin cambiar el estado.
    j = c.ask(pid, j.version, "¿Puedo subir un Excel?")
    assert j.state == "awaiting_data" and j.messages[-2].text == "¿Puedo subir un Excel?"
    assert j.messages[-1].text.startswith("No pude consultar a la IA")
    assert j.messages[-1].chat and j.messages[-2].chat and not j.messages[-3].chat
    # El LLM ofrece una acción: solo si está permitida en la etapa (sin datos no se entrena).
    fake_llm.script("copilot", {"reply": "Primero subí tus datos.", "action": "train"})
    j = c.ask(pid, j.version, "Entrená ya")
    assert j.messages[-1].text == "Primero subí tus datos." and j.messages[-1].action is None
    fake_llm.script("copilot", {"reply": "Volvamos a contar el caso.", "action": "revise"})
    j = c.ask(pid, j.version, "Me equivoqué en lo que te conté")
    assert j.messages[-1].action == "revise" and j.state == "awaiting_data"


def test_unsupported_case_cannot_be_confirmed(ctx: Any) -> None:
    p = ctx.projects.add(Project(name="Pronóstico"))
    c = Coordinator(Workflow(ctx))
    j = c.interview(p.id, 1, "", Facts(data_kind="table", problem="forecast"))
    assert j.state == "unsupported" and j.unsupported
    with pytest.raises(ValidationError):
        c.confirm(p.id, j.version)


# ---------------------------------------------------------------------- robustez


def test_cross_project_source_is_rejected(ctx: Any, tmp_path: Path) -> None:
    pid, c, j = start(ctx)
    other = ctx.projects.add(Project(name="Otro"))
    src = ctx.repo(DataSource).add(
        DataSource(type="folder", project_id=other.id, name="datos", config={"path": str(tmp_path)})
    )
    with pytest.raises(ValidationError):
        c.prepare(pid, j.version, src.id)


def test_atomic_reservation_rejects_double_start(ctx: Any) -> None:
    pid, c, j = start(ctx)
    candidates = [j.model_copy(update={"state": "training"}) for _ in range(2)]

    def reserve(candidate: Any) -> bool:
        try:
            c.save(pid, candidate)
            return True
        except ConflictError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(reserve, candidates)) == [False, True]


def test_recovery_preserves_data_and_notifies(ctx: Any) -> None:
    pid, c, j = start(ctx)
    j.state, j.owner_pid, j.owner_started = "training", 2_000_000_000, 0
    j.dataset_id = "dataset-guardado"
    c.save(pid, j)
    restored = Coordinator(Workflow(ctx)).get(pid)
    assert restored.state == "interrupted" and restored.notification == "interrupted"
    assert restored.dataset_id == "dataset-guardado" and restored.facts == CHURN


def test_recovery_waits_for_remote_cancellation_before_retry(ctx: Any) -> None:
    from perceptron.domain.models import Study

    pid, c, j = start(ctx)
    study = ctx.repo(Study).add(Study(project_id=pid, name="pendiente"))
    child = ctx.jobs.track("study", refs={"study_id": study.id, "project_id": pid}, runner="remote")
    j.state, j.owner_pid, j.owner_started = "training", 2_000_000_000, 0
    j.study_id = study.id  # caída entre encolar y guardar child_job_id
    c.save(pid, j)
    recovering = c.get(pid)
    assert recovering.state == "training" and recovering.stage == "stopping"
    assert ctx.jobs.get(child.id).status == "cancelled"
    with pytest.raises(ValidationError):
        c.train(pid, recovering.version)
    ctx.jobs.apply(child.id, "finished", {"status": "cancelled"})
    assert c.get(pid).state == "interrupted"


def test_legacy_image_journeys_restart_the_interview(ctx: Any) -> None:
    from perceptron.domain.models import JourneyRecord

    p = ctx.projects.add(Project(name="Viejo"))
    ctx.repo(JourneyRecord).add(
        JourneyRecord(
            id=p.id,
            project_id=p.id,
            payload={
                "schema_version": 1,
                "state": "ready_to_train",
                "facts": {"modality": "image"},
                "messages": [{"role": "user", "text": "fotos"}],
            },
        )
    )
    j = Coordinator(Workflow(ctx)).get(p.id)
    assert j.state == "interview" and j.messages[0].text == "fotos"


# ---------------------------------------------------------------------- datos


def images(tmp_path: Path, labels: tuple[str, ...] = ("fisura", "sano")) -> Path:
    root = tmp_path / "Imágenes con espacios"
    rng = np.random.default_rng(42)
    for k, label in enumerate(labels):
        folder = root / label
        folder.mkdir(parents=True)
        for i in range(30):
            arr = rng.integers(0, 100, (32, 32, 3), dtype=np.uint8) + 100 * k
            Image.fromarray(arr.astype(np.uint8)).save(folder / f"{i}.png")
    return root


def test_folders_must_match_the_confirmed_categories(ctx: Any, tmp_path: Path) -> None:
    facts = Facts(
        data_kind="images",
        problem="category",
        prediction="si la pieza tiene una fisura",
        labels_available=True,
        classes=["fisura", "sano"],
        error_costs="symmetric",
        grouped=False,
        success_target=60,
    )
    pid, c, j = start(ctx, facts, name="Fisuras")
    root = images(tmp_path, ("fisura", "otra"))
    src = ctx.repo(DataSource).add(
        DataSource(type="folder", project_id=pid, name="fotos", config={"path": str(root)})
    )
    c.prepare(pid, j.version, src.id)
    j = wait(c, pid)
    assert j.state == "awaiting_data"
    assert any("coincidir con las categorías" in b for b in j.blockers)


def test_wrong_target_column_asks_to_choose_it(ctx: Any, fixtures_dir: Path) -> None:
    pid, c, j = start(ctx, CHURN.model_copy(update={"target_name": "abandono"}))
    src = ctx.repo(DataSource).add(
        DataSource(
            type="file",
            project_id=pid,
            name="churn",
            config={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        )
    )
    c.prepare(pid, j.version, src.id)
    j = wait(c, pid)
    assert "churn" in j.columns
    if j.state == "awaiting_data":
        assert any("Elegí cuál columna" in b for b in j.blockers)
        c.prepare(pid, j.version, src.id, target="churn")
        j = wait(c, pid)
    assert j.state == "ready_to_train" and j.target == "churn", j.blockers


# ---------------------------------------------------------------------- de punta a punta


def test_table_journey_trains_validates_and_predicts(
    client: Any, ctx: Any, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from perceptron.hpo.plan import plan_budget
    from perceptron.services import autodesign

    # Presupuesto mínimo para el test: el recorrido es el mismo.
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    monkeypatch.setattr(autodesign, "TOURNAMENT_BUDGET_S", 0)

    def tiny(spec: Any, **kw: Any) -> Any:
        return plan_budget(spec, **kw).model_copy(
            update={"max_trials": 1, "max_epochs_per_trial": 2}
        )

    monkeypatch.setattr(autodesign, "plan_budget", tiny)
    pid, c, j = start(ctx)
    src = ctx.repo(DataSource).add(
        DataSource(
            type="file",
            project_id=pid,
            name="churn",
            config={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        )
    )
    c.prepare(pid, j.version, src.id)
    j = wait(c, pid)
    assert j.state == "ready_to_train", j.blockers
    assert j.data_summary["target"] == "churn" and j.data_summary["rows"] > 100
    said = j.messages[-1]
    assert said.role == "assistant" and said.step == "review" and said.action == "train"
    assert "Revisé tus datos" in said.text and any(m.text == "Te paso «churn»." for m in j.messages)
    started = c.train(pid, j.version)
    with pytest.raises((ConflictError, ValidationError)):
        c.train(pid, j.version)  # doble clic: no lanza otro entrenamiento
    j = wait(c, pid, 600)
    assert j.state in {"ready_to_validate", "needs_improvement"}, (j.error, j.stage)
    assert j.notification == "training_finished" and started.job_id
    steps = [m.step for m in j.messages if m.role == "assistant"]
    assert steps[-2:] == ["training", "validate"]  # arranque y resultado en el mismo hilo
    assert j.result["kind"] == "classification" and j.result["total"] > 0
    assert j.result["explanation"] and j.result["next_steps"]
    assert "positive" in j.result  # omisiones y falsas alarmas de la categoría importante
    # El borrador del wizard quedó con el diseño elegido (se ve en las opciones avanzadas).
    [draft] = ctx.repo(ProjectDraft).list(filters={"project_id": pid})
    assert draft.values["archspec_id"] and draft.values["dataset_version_id"] == j.dataset_id
    # Formulario con solo lo que el modelo necesita, con tipos y opciones.
    names = {f.name: f for f in j.form.fields}
    assert "churn" not in names and "customer_id" not in names
    assert names["edad"].kind == "number" and names["region"].kind == "category"
    assert names["region"].options
    row = {
        "edad": 30,
        "region": "AMBA",
        "plan": "premium",
        "antiguedad_meses": 12,
        "cargo_mensual": 50.0,
        "tickets_90d": 1,
    }
    pred = client.post(f"/api/v1/runs/{j.run_id}/predict", json={"rows": [row]})
    assert pred.status_code == 200, pred.text  # el Playground sirve al recorrido
    url = f"/api/v1/projects/{pid}/journey/validate"
    r = client.post(url, json={"row": row, "actual": "1"})
    assert r.status_code == 200, r.text
    r = client.post(url, json={"row": row, "actual": "1"})
    assert len(r.json()["journey"]["validations"]) == 1  # repetir un caso no suma
    assert client.post(url, json={"row": row, "actual": "quizás"}).status_code == 422
    # En lote: un CSV de casos nuevos con la columna del resultado.
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[*row, "churn"])
    writer.writeheader()
    for k in range(12):
        writer.writerow({**row, "edad": 40 + k, "churn": k % 2})
    r = client.post(
        f"{url}/file", files={"file": ("nuevos.csv", buf.getvalue().encode(), "text/csv")}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["checked"] == 12 and len(body["journey"]["validations"]) == 13
    assert body["journey"]["state"] in {"ready_to_use", "needs_improvement", "ready_to_validate"}


def test_without_llm_written_answers_to_the_last_question_are_understood(ctx: Any) -> None:
    from perceptron.services.journey_talk import understand

    assert understand("classes", "con fisura y sin fisura") == {
        "classes": ["con fisura", "sin fisura"]
    }
    assert understand("success_target", "8 de cada 10") == {"success_target": 80.0}
    assert understand("success_target", "un 15 %") == {"success_target": 15.0}
    assert understand("grouped", "Sí, saco 3 fotos de cada pieza") == {"grouped": True}
    assert understand("labels_available", "no") == {"labels_available": False}
    assert understand("error_costs", "no sé") is None  # elegir entre opciones: botones
    p = ctx.projects.add(Project(name="Fisuras"))
    c = Coordinator(Workflow(ctx))
    j = c.interview(p.id, 1, "", Facts(data_kind="images", problem="category"))
    assert j.messages[-1].field == "prediction"
    j = c.interview(p.id, j.version, "si la pieza tiene una fisura")
    assert j.facts.prediction == "si la pieza tiene una fisura"
    assert j.messages[-1].field == "labels_available" and "carpeta" in j.messages[-1].text
