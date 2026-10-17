from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from perceptron.core.errors import ConflictError, ValidationError
from perceptron.domain.models import DataSource, Project
from perceptron.hpo.strategy import Budget
from perceptron.services.journey import Coordinator, Facts, missing
from perceptron.services.workflow import Workflow


def setup_journey(ctx):
    p = ctx.projects.add(Project(name="Fisuras", goal="Clasificar imágenes con y sin fisuras"))
    ctx.files.init_project(p)
    c = Coordinator(Workflow(ctx))
    facts = Facts(
        modality="image",
        task="classification",
        classes=["fisura", "sano"],
        positive_class="fisura",
        grouped=False,
        labels_available=True,
        success_percent=50,
        error_costs="misses",
    )
    j = c.interview(p.id, 1, "", facts)
    return p.id, c, c.confirm(p.id, j.version)


def wait(c, pid, timeout=120):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        j = c.get(pid)
        if j.state not in {"reviewing_data", "training"}:
            return j
        time.sleep(0.1)
    raise AssertionError("Journey did not finish")


def test_rules_require_explicit_task_and_consistent_classes():
    assert "task" in missing(Facts(modality="image"))
    assert "positive_class" in missing(Facts(classes=["a", "b"], positive_class="c"))
    assert "classes" in missing(Facts(classes=["a", "a"]))


def test_persistence_and_llm_failure(client, ctx):
    p = ctx.projects.add(Project(name="Prueba"))
    url = f"/api/v1/projects/{p.id}/journey"
    r = client.post(url, json={"action": "message", "version": 1, "message": "Tengo fotos"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["llm_failed"]
    assert j["messages"][0]["text"] == "Tengo fotos"
    assert client.get(url).json() == j
    assert client.post(url, json={"action": "confirm", "version": j["version"]}).status_code == 422
    assert client.post(url, json={"action": "train", "version": j["version"]}).status_code == 422
    assert (
        client.post(url, json={"action": "message", "version": 1, "message": "otro"}).status_code
        == 409
    )


def test_unsupported_and_cross_project_source(ctx, tmp_path):
    pid, c, j = setup_journey(ctx)
    other = ctx.projects.add(Project(name="Otro"))
    src = ctx.repo(DataSource).add(
        DataSource(type="folder", project_id=other.id, name="datos", config={"path": str(tmp_path)})
    )
    with pytest.raises(ValidationError):
        c.prepare(pid, j.version, src.id)
    p = ctx.projects.add(Project(name="Localizar"))
    unsupported = c.interview(p.id, 1, "", Facts(modality="image", task="detection"))
    assert unsupported.state == "unsupported"
    with pytest.raises(ValidationError):
        c.confirm(p.id, unsupported.version)


def test_atomic_reservation_rejects_double_start(ctx):
    pid, c, j = setup_journey(ctx)
    candidates = [j.model_copy(update={"state": "training"}) for _ in range(2)]

    def reserve(candidate):
        try:
            c.save(pid, candidate)
            return True
        except ConflictError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(reserve, candidates)) == [False, True]


def test_recovery_preserves_data_and_notifies(ctx):
    pid, c, j = setup_journey(ctx)
    j.state, j.owner_pid, j.owner_started = "training", 2_000_000_000, 0
    j.dataset_id = "saved-dataset"
    c.save(pid, j)
    restored = Coordinator(Workflow(ctx)).get(pid)
    assert restored.state == "interrupted"
    assert restored.notification == "interrupted"
    assert restored.dataset_id == "saved-dataset"
    assert restored.facts.classes == ["fisura", "sano"]


def images(tmp_path: Path):
    root = tmp_path / "Imágenes con espacios"
    rng = np.random.default_rng(42)
    for label, offset in [("fisura", 0), ("sano", 100)]:
        folder = root / label
        folder.mkdir(parents=True)
        for i in range(40):
            arr = rng.integers(0, 100, (32, 32, 3), dtype=np.uint8) + offset
            Image.fromarray(arr).save(folder / f"{i}.png")
    return root


def test_prepare_real_images_and_reconnect(ctx, tmp_path):
    pid, c, j = setup_journey(ctx)
    root = images(tmp_path)
    src = ctx.repo(DataSource).add(
        DataSource(type="folder", project_id=pid, name="datos", config={"path": str(root)})
    )
    c.prepare(pid, j.version, src.id)
    j = wait(c, pid)
    assert j.state == "ready_to_train", j.model_dump()
    assert len(list(root.rglob("*.png"))) == 80
    assert Coordinator(Workflow(ctx)).get(pid).dataset_id == j.dataset_id


@pytest.mark.timeout(300)
def test_complete_real_image_journey(ctx, client, tmp_path, monkeypatch):
    import perceptron.services.journey as module

    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    monkeypatch.setattr(
        module, "Budget", lambda **_: Budget(max_trials=1, max_epochs_per_trial=1, max_time_s=90)
    )
    pid, c, j = setup_journey(ctx)
    root = images(tmp_path)
    src = ctx.repo(DataSource).add(
        DataSource(type="folder", project_id=pid, name="datos", config={"path": str(root)})
    )
    c.prepare(pid, j.version, src.id)
    j = wait(c, pid)
    assert j.state == "ready_to_train", j.model_dump()
    started = c.train(pid, j.version)
    with pytest.raises((ConflictError, ValidationError)):
        c.train(pid, j.version)
    j = wait(c, pid, 240)
    assert j.state in {"ready_to_validate", "needs_improvement"}, j.model_dump()
    assert j.notification == "training_finished"
    assert j.contract["inputs"]["kind"] == "image"
    assert j.result["total"] > 0
    url = f"/api/v1/projects/{pid}/journey/predict"
    import io

    buffer = io.BytesIO()
    Image.fromarray(np.random.default_rng(999).integers(0, 100, (32, 32, 3), dtype=np.uint8)).save(
        buffer, format="PNG"
    )
    data = buffer.getvalue()
    r = client.post(url, files={"file": ("image.png", data, "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["actual"] is None
    url = f"/api/v1/projects/{pid}/journey/validate"
    for _ in range(2):
        r = client.post(
            url, files={"file": ("image.png", data, "image/png")}, data={"actual": "fisura"}
        )
        assert r.status_code == 200, r.text
    assert len(c.get(pid).validations) == 1
    assert r.json()["correct"] == (r.json()["prediction"] == "fisura")
    assert (
        client.post(
            url, files={"file": ("image.png", data)}, data={"actual": "unknown"}
        ).status_code
        == 422
    )
    assert started.job_id


def test_llm_only_records_quoted_facts(ctx, fake_llm):
    p = ctx.projects.add(Project(name="Entrevista"))
    fake_llm.script(
        "copilot",
        {
            "facts": {"modality": "image", "task": "classification"},
            "evidence": {"modality": "fotos", "task": "clasificación"},
            "assumptions": ["Quizás busca una categoría por foto"],
            "question": "¿Necesitás una categoría o localizar el defecto?",
            "question_field": "task",
        },
    )
    c = Coordinator(Workflow(ctx))
    j = c.interview(p.id, 1, "Tengo fotos con fisuras", None)
    assert not j.llm_failed
    assert j.facts.modality == "image"
    assert j.facts.task is None  # no matching explicit evidence
    assert "task" in j.pending
    assert c.get(p.id).assumptions == ["Quizás busca una categoría por foto"]
    assert len(c.get(p.id).messages) == 2


def test_group_partitions_and_metadata_are_not_predictors(ctx, tmp_path):
    from perceptron.data.view import Purpose

    pid, c, j = setup_journey(ctx)
    j = c.revise(pid, j.version)
    j = c.interview(pid, j.version, "", Facts(grouped=True))
    j = c.confirm(pid, j.version)
    root = images(tmp_path)
    for label in ["fisura", "sano"]:
        for i in range(40):
            destination = root / label / f"grupo_{i // 2}"
            destination.mkdir(exist_ok=True)
            (root / label / f"{i}.png").rename(destination / f"{i}.png")
    src = ctx.repo(DataSource).add(
        DataSource(type="folder", project_id=pid, name="grupos", config={"path": str(root)})
    )
    c.prepare(pid, j.version, src.id)
    j = wait(c, pid)
    assert j.state == "ready_to_train", j.error or j.blockers
    view = c.wf.view(c.wf.dataset(j.dataset_id))
    # Test data inspected ONLY by this test assertion, never by coordinator/model selection.
    rows = view.read(purpose=Purpose.VERSIONING)
    assert (
        rows.group_by("__image_group__")
        .agg(__import__("polars").col("__split__").n_unique())["__split__"]
        .max()
        == 1
    )
    assert "__image_group__" not in [col.name for col in view.schema.feature_columns]


def test_duplicate_images_block_training(ctx, tmp_path):
    import shutil

    pid, c, j = setup_journey(ctx)
    root = images(tmp_path)
    shutil.copyfile(root / "fisura" / "0.png", root / "fisura" / "duplicate.png")
    src = ctx.repo(DataSource).add(
        DataSource(type="folder", project_id=pid, name="datos", config={"path": str(root)})
    )
    c.prepare(pid, j.version, src.id)
    j = wait(c, pid)
    assert j.state == "awaiting_data"
    assert any("idénticos" in b for b in j.blockers)
    with pytest.raises(ValidationError):
        c.train(pid, j.version)


def test_cancel_is_persistent_and_does_not_allow_early_retry(ctx):
    pid, c, j = setup_journey(ctx)
    # An active reservation with no job id models the interval before submission.
    j = c.save(pid, j.model_copy(update={"state": "training", "dataset_id": "dsv_saved"}))
    stopped = c.stop(pid, j.version)
    assert stopped.stop_requested
    assert stopped.state == "training"
    assert c.get(pid).stop_requested
    with pytest.raises(ValidationError):
        c.train(pid, stopped.version)


def test_wrong_model_contract_is_rejected(ctx):
    from perceptron.domain.models import ModelVersion

    pid, c, j = setup_journey(ctx)
    other = ctx.projects.add(Project(name="Otro"))
    model = ctx.repo(ModelVersion).add(ModelVersion(project_id=other.id, run_id="run_wrong"))
    c.save(
        pid,
        j.model_copy(
            update={
                "state": "ready_to_validate",
                "model_id": model.id,
                "run_id": "run_wrong",
                "contract": {"inputs": {"kind": "image"}},
            }
        ),
    )
    with pytest.raises(ValidationError):
        c.contract(pid)


def test_unreadable_images_are_actionable_without_touching_originals(ctx, tmp_path):
    pid, c, j = setup_journey(ctx)
    root = images(tmp_path)
    broken = root / "fisura" / "0.png"
    broken.write_bytes(b"not-an-image")
    src = ctx.repo(DataSource).add(
        DataSource(type="folder", project_id=pid, name="datos", config={"path": str(root)})
    )
    c.prepare(pid, j.version, src.id)
    j = wait(c, pid)
    assert j.state == "awaiting_data"
    assert any("fisura/0.png" in b for b in j.blockers)
    assert broken.read_bytes() == b"not-an-image"


def test_uploaded_zip_uses_existing_ingestion(client, ctx, tmp_path):
    import io
    import zipfile

    pid, c, j = setup_journey(ctx)
    root = images(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for image in root.rglob("*.png"):
            archive.write(image, image.relative_to(root).as_posix())
    response = client.post(
        f"/api/v1/projects/{pid}/uploads",
        files=[("files", ("imagenes.zip", buffer.getvalue(), "application/zip"))],
    )
    assert response.status_code == 201, response.text
    response = client.post(
        f"/api/v1/projects/{pid}/journey",
        json={
            "action": "prepare",
            "version": j.version,
            "source_id": response.json()["id"],
        },
    )
    assert response.status_code == 200, response.text
    assert wait(c, pid).state == "ready_to_train"


def test_recovery_waits_for_remote_cancellation_before_retry(ctx):
    from perceptron.domain.models import Study

    pid, c, j = setup_journey(ctx)
    study = ctx.repo(Study).add(Study(project_id=pid, name="pendiente"))
    child = ctx.jobs.track("study", refs={"study_id": study.id, "project_id": pid}, runner="remote")
    j.state, j.owner_pid, j.owner_started = "training", 2_000_000_000, 0
    j.study_id = study.id  # simulate crash after enqueue, before saving child_job_id
    c.save(pid, j)
    recovering = c.get(pid)
    assert recovering.state == "training"
    assert recovering.stage == "stopping"
    assert ctx.jobs.get(child.id).status == "cancelled"
    with pytest.raises(ValidationError):
        c.train(pid, recovering.version)
    ctx.jobs.apply(child.id, "finished", {"status": "cancelled"})
    recovered = c.get(pid)
    assert recovered.state == "interrupted"
    assert ctx.repo(Study).get(study.id).cancel_requested_at is not None
