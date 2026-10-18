"""El asistente lee los archivos de la persona y le propone qué puede aprender la red (ADR-0042):
planillas sueltas, varias planillas que se juntan o se unen por una columna, fotos con marcas o
por carpeta y grabaciones."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Any

import polars as pl

from perceptron.domain.models import DatasetVersion, DataSource, Project
from perceptron.services.journey import Coordinator
from perceptron.services.journey_files import consistent, mentioned, read
from perceptron.services.journey_guide import Facts
from perceptron.services.workflow import Workflow

API = "/api/v1"


def _titles(path: Path) -> list[str]:
    return [g.title for g in read(path).goals]


def test_one_spreadsheet_proposes_what_can_be_predicted(fixtures_dir: Path) -> None:
    r = read(fixtures_dir / "uc01_churn" / "churn.csv")
    assert r.files[0].kind == "table" and r.files[0].rows and len(r.files[0].columns) == 8
    first = r.goals[0]
    assert first.title == "Predecir «churn»" and first.target == "churn"
    assert first.facts.problem == "category" and first.facts.labels_available is True
    assert "customer_id" not in (first.facts.input_names or [])  # un id no es un dato
    assert all(g.target != "customer_id" for g in r.goals)
    # Un número de la tabla también se puede estimar.
    assert any(g.facts.problem == "value" and g.target == "cargo_mensual" for g in r.goals)
    # Sin valores de las filas: solo nombres, tipos y cantidades.
    dumped = r.files[0].model_dump_json()
    assert "C00000" not in dumped and "AMBA" not in dumped


def test_measurements_in_time_propose_forecasts_and_unusual_moments(fixtures_dir: Path) -> None:
    demand = read(fixtures_dir / "uc07_demand" / "demanda.csv").goals
    assert demand[0].facts.problem == "forecast" and "para cada «sku»" in demand[0].detail
    unmarked = next(g for g in demand if g.facts.problem == "anomaly")
    assert unmarked.facts.labels_available is False
    telemetry = read(fixtures_dir / "uc08_telemetry" / "telemetria.csv").goals
    assert telemetry[0].title == "Detectar momentos raros" and telemetry[0].target == "anomalia"
    assert telemetry[0].facts.labels_available is True


def test_photos_and_recordings_by_folder(fixtures_dir: Path, tmp_path: Path) -> None:
    # Fotos con recuadros y además separadas por carpeta: se puede ubicar o clasificar.
    defects = read(fixtures_dir / "uc04_defects").goals
    assert [g.facts.problem for g in defects] == ["locate", "category"]
    assert defects[1].facts.classes == ["defect", "ok"]
    assert [g.facts.problem for g in read(fixtures_dir / "uc05_masks").goals] == ["segment"]
    assert [g.facts.problem for g in read(fixtures_dir / "uc06_ocr").goals] == ["read_text"]
    audio = read(fixtures_dir / "uc09_motor_audio").goals
    assert audio[0].facts.data_kind == "audio" and len(audio[0].facts.classes or []) == 4
    # Un zip de fotos se lee sin extraerlo.
    zipped = tmp_path / "fotos.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        for folder in ("defect", "ok"):
            for p in sorted((fixtures_dir / "uc04_defects" / folder).glob("*.png"))[:10]:
                zf.write(p, f"fotos/{folder}/{p.name}")
    assert [g.facts.problem for g in read(zipped).goals] == ["category"]
    # Fotos sin separar: no hay qué aprender todavía, y se explica cómo prepararlas.
    loose = tmp_path / "sueltas"
    loose.mkdir()
    for p in sorted((fixtures_dir / "uc04_defects").rglob("*.png"))[:5]:
        (loose / p.name).write_bytes(p.read_bytes())
    r = read(loose)
    assert not r.goals and r.note and "una carpeta por respuesta" in r.note


def _customers(fixtures_dir: Path, folder: Path) -> pl.DataFrame:
    churn = pl.read_csv(fixtures_dir / "uc01_churn" / "churn.csv")
    folder.mkdir(parents=True, exist_ok=True)
    churn.select("customer_id", "edad", "region", "plan", "churn").write_csv(
        folder / "clientes.csv"
    )
    payments = churn.select("customer_id", "cargo_mensual", "tickets_90d").sample(
        n=churn.height * 2, with_replacement=True, seed=7
    )
    payments.write_csv(folder / "pagos.csv")
    return payments


def test_several_spreadsheets_are_joined_or_stacked(fixtures_dir: Path, tmp_path: Path) -> None:
    _customers(fixtures_dir, tmp_path / "unir")
    r = read(tmp_path / "unir")
    assert r.combined == "«pagos.csv» unido con «clientes.csv» por «customer_id»"
    churn = next(g for g in r.goals if g.target == "churn")
    assert churn.parts == ["pagos.csv", "clientes.csv"] and churn.keys == ["customer_id"]
    assert churn.combine == "join" and "customer_id" not in (churn.facts.input_names or [])
    assert {"cargo_mensual", "edad"} <= set(churn.facts.input_names or [])
    # Mismas columnas en varios archivos: son partes de una misma tabla.
    data = pl.read_csv(fixtures_dir / "uc01_churn" / "churn.csv")
    (tmp_path / "meses").mkdir()
    data.head(250).write_csv(tmp_path / "meses" / "enero.csv")
    data.tail(250).write_csv(tmp_path / "meses" / "febrero.csv")
    stacked = read(tmp_path / "meses")
    assert stacked.combined == "los 2 archivos juntos" and stacked.goals[0].combine == "stack"
    # Un archivo que no se puede leer se cuenta y no frena a los demás.
    (tmp_path / "meses" / "roto.parquet").write_bytes(b"no es parquet")
    mixed = read(tmp_path / "meses")
    assert any(f.error for f in mixed.files) and mixed.goals


def test_other_formats_are_explained(tmp_path: Path) -> None:
    pdf = tmp_path / "informe.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    r = read(pdf)
    assert not r.goals and r.note and "planillas" in r.note
    english = read(pdf, "en")
    assert english.note and "spreadsheets" in english.note


def test_the_person_names_a_goal_in_their_own_words(fixtures_dir: Path) -> None:
    goals = read(fixtures_dir / "uc07_demand" / "demanda.csv").goals
    forecast = mentioned("quiero el pronóstico de unidades", goals)
    assert forecast and forecast.facts.problem == "forecast"
    estimate = mentioned("estimar las unidades", goals)
    assert estimate and estimate.facts.problem == "value"
    assert mentioned("no sé bien", goals) is None
    assert [g.facts.problem for g in consistent(Facts(problem="forecast"), goals)] == ["forecast"]


def _wait(c: Coordinator, pid: str, timeout: float = 300) -> Any:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        j = c.get(pid)
        if j.state not in {"reviewing_data", "training"}:
            return j
        time.sleep(0.2)
    raise AssertionError("el recorrido no terminó")


def test_files_first_then_goal_then_review_without_llm(ctx: Any, fixtures_dir: Path) -> None:
    """Sin IA: subir, elegir el objetivo con un clic, completar lo que falta y revisar los datos
    sin volver a subirlos."""
    p = ctx.projects.add(Project(name="Bajas"))
    ctx.files.init_project(p)
    path = fixtures_dir / "uc01_churn" / "churn.csv"
    src = ctx.repo(DataSource).add(
        DataSource(type="file", project_id=p.id, name=path.name, config={"path": str(path)})
    )
    c = Coordinator(Workflow(ctx))
    j = c.read_files(p.id, 1, src.id, "Tengo esta planilla de clientes")
    assert [m.role for m in j.messages] == ["user", "user", "assistant"]
    assert j.messages[1].text == "Te paso «churn.csv»." and j.messages[1].action == "files"
    said = j.messages[-1]
    assert said.text.startswith("Leí «churn.csv» (") and "te propongo estos objetivos" in said.text
    assert said.goals[0] == "g1" and len(said.goals) == 4
    j = c.choose(p.id, j.version, "g1")
    assert j.messages[-2].text == "Predecir «churn»" and j.messages[-2].action == "goal"
    assert j.messages[-1].text.startswith("Perfecto: predecir «churn»")
    assert j.facts.problem == "category" and j.facts.target_name == "churn"
    assert j.data_ready and j.target == "churn" and j.source_id == src.id
    assert "data_kind" not in j.pending and "problem" not in j.pending
    j = c.interview(
        p.id,
        j.version,
        "Los dos errores pesan igual y 8 de cada 10 está bien",
        Facts(error_costs="symmetric", success_target=80),
    )
    assert not j.pending and j.messages[-1].action == "confirm"
    j = c.confirm(p.id, j.version)
    assert "tengo tus datos" in next(m.text for m in reversed(j.messages) if m.step == "data")
    j = _wait(c, p.id)
    assert j.state == "ready_to_train", j.blockers
    assert j.target == "churn" and j.data_summary and j.data_summary["target"] == "churn"
    assert ctx.projects.get(p.id).goal == "Tengo esta planilla de clientes"


def test_two_spreadsheets_by_upload_are_joined_and_reviewed(
    client: Any, ctx: Any, fixtures_dir: Path, tmp_path: Path
) -> None:
    payments = _customers(fixtures_dir, tmp_path)
    pid = client.post(f"{API}/projects", json={"name": "Clientes"}).json()["id"]
    files = [
        ("files", (name, (tmp_path / name).read_bytes(), "text/csv"))
        for name in ("clientes.csv", "pagos.csv")
    ]
    src = client.post(f"{API}/projects/{pid}/uploads", files=files)
    assert src.status_code == 201, src.text
    url = f"{API}/projects/{pid}/journey"
    j = client.post(
        url, json={"action": "files", "version": 1, "source_id": src.json()["id"]}
    ).json()
    assert j["messages"][0]["text"] == "Te paso «clientes.csv» y «pagos.csv»."
    assert "Los puedo usar combinados" in j["messages"][-1]["text"]
    churn = next(g for g in j["goals"] if g["target"] == "churn")
    r = client.post(url, json={"action": "choose", "version": j["version"], "goal_id": churn["id"]})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["source_id"] != src.json()["id"]  # la tabla unida es una fuente nueva
    combined = ctx.repo(DataSource).get(j["source_id"])
    assert Path(combined.config["path"]).suffix == ".parquet"
    assert pl.read_parquet(combined.config["path"]).height == payments.height
    j = client.post(
        url,
        json={
            "action": "message",
            "version": j["version"],
            "message": "igual, 7 de cada 10",
            "facts": {"error_costs": "symmetric", "success_target": 70},
        },
    ).json()
    j = client.post(url, json={"action": "confirm", "version": j["version"]}).json()
    c = Coordinator(Workflow(ctx))
    done = _wait(c, pid)
    assert done.state == "ready_to_train", done.blockers
    assert done.data_summary and done.data_summary["rows"] == payments.height
    # Cada cliente tiene varias filas: se separan por cliente y su id no es un dato.
    assert "customer_id" not in done.data_summary["inputs"]
    dv = ctx.repo(DatasetVersion).get(done.dataset_id)
    assert dv.split.strategy == "group" and dv.split.group_column == "customer_id"
    bad = client.post(url, json={"action": "choose", "version": done.version, "goal_id": "g1"})
    assert bad.status_code == 422  # ya no se está contando el caso


def test_written_answer_picks_the_goal_without_llm(ctx: Any, fixtures_dir: Path) -> None:
    p = ctx.projects.add(Project(name="Demanda"))
    path = fixtures_dir / "uc07_demand" / "demanda.csv"
    src = ctx.repo(DataSource).add(
        DataSource(type="file", project_id=p.id, name=path.name, config={"path": str(path)})
    )
    c = Coordinator(Workflow(ctx))
    j = c.read_files(p.id, 1, src.id)
    j = c.interview(p.id, j.version, "Quiero el pronóstico de las unidades")
    assert j.goal_id and j.facts.problem == "forecast" and j.facts.target_name == "unidades"
    assert j.messages[-1].field == "horizon"
    assert j.messages[-1].text.startswith("Perfecto: pronosticar «unidades»")
    # Sin elegir, el asistente vuelve a ofrecer los objetivos.
    other = ctx.projects.add(Project(name="Demanda 2"))
    src2 = ctx.repo(DataSource).add(
        DataSource(type="file", project_id=other.id, name=path.name, config={"path": str(path)})
    )
    j2 = c.read_files(other.id, 1, src2.id)
    j2 = c.interview(other.id, j2.version, "mmm no sé")
    assert j2.messages[-1].goals and "objetivos" in j2.messages[-1].text and not j2.goal_id


def test_llm_orders_and_explains_the_proposals(ctx: Any, fake_llm: Any, fixtures_dir: Path) -> None:
    p = ctx.projects.add(Project(name="Bajas", privacy_level="L1"))
    path = fixtures_dir / "uc01_churn" / "churn.csv"
    src = ctx.repo(DataSource).add(
        DataSource(type="file", project_id=p.id, name=path.name, config={"path": str(path)})
    )
    fake_llm.script(
        "copilot",
        {
            "reply": "Tenés clientes con su plan y si se dieron de baja. Empezaría por ahí.",
            "picks": ["g1", "inventado", "g3"],
        },
    )
    c = Coordinator(Workflow(ctx))
    j = c.read_files(p.id, 1, src.id)
    said = j.messages[-1]
    assert said.text.startswith("Tenés clientes") and said.goals[:2] == ["g1", "g3"]
    sent = str(fake_llm.calls("copilot")[-1])
    assert "churn" in sent and "C00000" not in sent  # columnas sí, valores de filas no
    # En la entrevista, el LLM señala el objetivo que la persona describe: se ofrece primero.
    fake_llm.script("copilot", {"reply": "Entiendo.", "goal": "g3", "changes": []})
    j = c.interview(p.id, j.version, "me interesa lo que paga cada uno")
    assert j.messages[-1].goals[0] == "g3" and not j.goal_id


def test_files_upload_is_read_from_a_zip_of_spreadsheets(
    ctx: Any, fixtures_dir: Path, tmp_path: Path
) -> None:
    _customers(fixtures_dir, tmp_path / "datos")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in ("clientes.csv", "pagos.csv"):
            zf.write(tmp_path / "datos" / name, name)
    zipped = tmp_path / "subida" / "tablas.zip"
    zipped.parent.mkdir()
    zipped.write_bytes(buf.getvalue())
    r = read(zipped)
    assert r.combined and any(g.target == "churn" for g in r.goals)
