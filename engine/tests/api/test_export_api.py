"""Export verificado del modelo (RF-EXP-01, RF-EXP-05; ADR-0027) por la API."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

API = "/api/v1"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _wait_job(client: TestClient, job_id: str, timeout: float = 600) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _ok(client.get(f"{API}/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("el job no terminó a tiempo")


def _trained_run(client: TestClient, fixtures_dir: Path) -> str:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Export"}), 201)["id"]
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    pipe = _ok(
        client.post(
            f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv["id"]}
        ),
        201,
    )
    prop = _ok(
        client.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"]},
        ),
        201,
    )["proposals"][0]
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv["id"],
                "pipeline_id": pipe["id"],
                "archspec_id": prop["archspec"]["id"],
                "budget": {"max_trials": 1, "max_epochs_per_trial": 2},
            },
        ),
        202,
    )
    job = _wait_job(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    run_id: str = job["result"]["best_trial"]["run_id"]
    return run_id


def test_export_all_formats_verified(client: TestClient, fixtures_dir: Path) -> None:
    run_id = _trained_run(client, fixtures_dir)
    assert client.get(f"{API}/runs/{run_id}/export").status_code == 404

    body = {"formats": ["onnx", "torch_export", "torchscript"], "fp16": True, "int8": True}
    launch = _ok(client.post(f"{API}/runs/{run_id}/export", json=body), 202)
    job = _wait_job(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]

    report = _ok(client.get(f"{API}/runs/{run_id}/export"))
    by_format = {a["format"]: a for a in report["artifacts"]}
    assert set(by_format) >= {"onnx", "onnx_fp16", "onnx_int8", "torch_export", "torchscript"}
    for fmt in ("onnx", "onnx_fp16", "torch_export", "torchscript"):
        art = by_format[fmt]
        assert art["error"] is None, (fmt, art["error"])
        assert art["verification"]["passed"], (fmt, art["verification"])
    # ONNX coincide con PyTorch (aceptación §14): 1e-4 en fp32 y 1e-2 en fp16.
    assert by_format["onnx"]["verification"]["tolerance"] == 1e-4
    assert by_format["onnx_fp16"]["verification"]["tolerance"] == 1e-2
    assert by_format["onnx_int8"]["verification"]["tolerance"] is None  # solo se informa
    assert by_format["torchscript"]["legacy"] is True
    assert [i["name"] for i in report["inputs"]] == ["x_num", "x_cat"]
    assert report["signature"]["inputs"]["kind"] == "tabular"
    assert report["signature"]["run_id"] == run_id and report["signature"]["perceptron_version"]

    onnx = client.get(f"{API}/runs/{run_id}/export/files/model.onnx")
    assert onnx.status_code == 200 and len(onnx.content) == by_format["onnx"]["size_bytes"]
    assert _ok(client.get(f"{API}/runs/{run_id}/export/files/signature.json"))["inputs"]
    # Solo se sirven los archivos del reporte: nada de rutas arbitrarias del run.
    bad = client.get(f"{API}/runs/{run_id}/export/files/run.json")
    assert bad.status_code == 404
    assert client.get(f"{API}/runs/{run_id}/export/files/..%2Frun.json").status_code == 404


def test_export_unknown_run(client: TestClient) -> None:
    assert client.post(f"{API}/runs/run_nope/export", json={}).status_code == 404


def _churn_rows(fixtures_dir: Path, n: int = 3) -> list[dict[str, Any]]:
    import csv

    with (fixtures_dir / "uc01_churn" / "churn.csv").open(encoding="utf-8") as f:
        rows = [dict(r) for _, r in zip(range(n), csv.DictReader(f), strict=False)]
    for r in rows:
        r.pop("churn")
    return rows


def test_playground_and_serving_bundle(
    client: TestClient, fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import json
    import subprocess
    import sys
    import zipfile

    run_id = _trained_run(client, fixtures_dir)
    rows = _churn_rows(fixtures_dir)
    # Sin export ONNX todavía: el playground lo pide.
    assert client.post(f"{API}/runs/{run_id}/predict", json={"rows": rows}).status_code == 404
    launch = _ok(client.post(f"{API}/runs/{run_id}/export", json={"formats": ["onnx"]}), 202)
    assert _wait_job(client, launch["job"]["id"])["status"] == "succeeded"

    # Playground (RF-EXP-02).
    res = _ok(client.post(f"{API}/runs/{run_id}/predict", json={"rows": rows}))
    assert res["task"] == "classification" and len(res["predictions"]) == len(rows)
    first = res["predictions"][0]
    assert first["prediction"] in {"0", "1"} and 0.5 <= first["confidence"] <= 1.0
    assert abs(sum(first["probabilities"].values()) - 1.0) < 1e-5
    partial = [{k: v for k, v in rows[0].items() if k != next(iter(rows[0]))}]
    bad = client.post(f"{API}/runs/{run_id}/predict", json={"rows": partial})
    assert bad.status_code == 422 and "faltan columnas" in bad.text

    # Paquete de serving (RF-EXP-03).
    zip_resp = client.get(f"{API}/runs/{run_id}/export/serving.zip")
    assert zip_resp.status_code == 200
    bundle = tmp_path / "bundle con espacio"
    archive = tmp_path / "serving.zip"
    archive.write_bytes(zip_resp.content)
    with zipfile.ZipFile(archive) as z:
        z.extractall(bundle)
    for rel in ("app/app.py", "app/model/model.onnx", "Dockerfile.cpu", "Dockerfile.cuda"):
        assert (bundle / rel).is_file(), rel
    reqs = (bundle / "requirements.txt").read_text(encoding="utf-8")
    assert "onnxruntime==" in reqs and "torch" not in reqs  # tabular: sin torch

    # El código vendorizado alcanza para predecir en un proceso aislado (-I: sin site del usuario).
    probe = (
        "import json, sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(bundle / 'vendor')!r}); "
        "import perceptron, perceptron.serving.runtime as r; "
        f"assert Path(perceptron.__file__).is_relative_to({str(bundle / 'vendor')!r}); "
        f"m = r.InferenceModel(Path({str(bundle / 'app' / 'model')!r})); "
        f"print(json.dumps([p.to_dict() for p in m.predict_rows({rows!r})]))"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    vendored = json.loads(out.stdout.strip().splitlines()[-1])
    assert [p["prediction"] for p in vendored] == [p["prediction"] for p in res["predictions"]]

    # El servidor generado: health, API key y predicción.
    monkeypatch.setenv("PERCEPTRON_MODEL_DIR", str(bundle / "app" / "model"))
    monkeypatch.setenv("PERCEPTRON_API_KEY", "clave")
    spec = importlib.util.spec_from_file_location("bundle_app", bundle / "app" / "app.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    served = TestClient(module.app)
    assert served.get("/health").json()["task"] == "classification"
    assert served.post("/predict", json={"rows": rows}).status_code == 401
    ok = served.post("/predict", json={"rows": rows}, headers={"X-API-Key": "clave"})
    assert ok.status_code == 200
    assert [p["prediction"] for p in ok.json()["predictions"]] == [
        p["prediction"] for p in res["predictions"]
    ]
    assert "predictions_total" in served.get("/metrics").text


def test_exportable_project_reproduces_and_infers(
    client: TestClient, fixtures_dir: Path, tmp_path: Path
) -> None:
    import json
    import subprocess
    import sys
    import zipfile

    run_id = _trained_run(client, fixtures_dir)
    launch = _ok(client.post(f"{API}/runs/{run_id}/export", json={"formats": ["onnx"]}), 202)
    assert _wait_job(client, launch["job"]["id"])["status"] == "succeeded"
    rows = _churn_rows(fixtures_dir, 5)
    onnx_preds = _ok(client.post(f"{API}/runs/{run_id}/predict", json={"rows": rows}))

    resp = client.get(f"{API}/runs/{run_id}/export/project.zip")
    assert resp.status_code == 200
    archive = tmp_path / "proyecto.zip"
    archive.write_bytes(resp.content)
    dest = tmp_path / "exportado ñ"
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    (root,) = [p for p in dest.iterdir() if p.is_dir()]
    package = next(p.name for p in (root / "src").iterdir() if p.is_dir())
    for rel in (
        "pyproject.toml",
        "config.yaml",
        "README.md",
        "LICENSES.md",
        "artifacts/model.pt",
        "data/train.parquet",
        "data/val.parquet",
        f"src/{package}/model.py",
        "vendor/perceptron/data/pipeline/pipeline.py",
        "tests/test_smoke.py",
    ):
        assert (root / rel).is_file(), rel
    assert "torch==" in (root / "pyproject.toml").read_text(encoding="utf-8")

    def run(code: str) -> str:
        prelude = (
            "import sys; "
            f"sys.path[:0] = [{str(root / 'src')!r}, {str(root / 'vendor')!r}]; "
            "import perceptron; from pathlib import Path; "
            f"assert Path(perceptron.__file__).is_relative_to({str(root / 'vendor')!r}); "
        )
        out = subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-c", prelude + code],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            cwd=root,
        )
        assert out.returncode == 0, out.stderr[-3000:]
        return out.stdout

    predict = (
        "import json, polars as pl; "
        f"from {package}.infer import predict; "
        f"print(json.dumps(predict(pl.DataFrame({rows!r})).to_dicts()))"
    )
    # Con los pesos que entrenó Perceptron, el código generado predice lo mismo que ONNX.
    exported = json.loads(run(predict).strip().splitlines()[-1])
    assert [p["prediction"] for p in exported] == [
        str(p["prediction"]) for p in onnx_preds["predictions"]
    ]

    # Reentrena desde cero (una época para el test) y vuelve a predecir (O5 en local).
    cfg = root / "config.yaml"
    cfg.write_text(
        "\n".join(
            "epochs: 1" if line.startswith("epochs:") else line
            for line in cfg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    trained = run(f"from {package}.train import main; main()")
    assert "best_val_loss" in trained
    again = json.loads(run(predict).strip().splitlines()[-1])
    assert len(again) == len(rows) and all("confidence" in p for p in again)
