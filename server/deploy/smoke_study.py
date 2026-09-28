"""Smoke de la cola del Team Server desplegado (Capa 5b): un estudio corre en un worker.

Solo stdlib (corre en el host, fuera de la imagen). Usa la fuente del servidor
`/sources/demanda.csv` (UC-07), lanza un estudio corto y espera que un worker lo termine con
el tracking en el MLflow server.

    PERCEPTRON_ADMIN_EMAIL=... PERCEPTRON_ADMIN_PASSWORD=... python smoke_study.py [URL]

Con `--expect-project NOMBRE` no entrena: verifica que el proyecto (y sus runs) existan,
p. ej. después de restaurar un backup (RF-SRV-08).
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import sys
import time
import urllib.request
from typing import Any

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
BASE = (ARGS[0] if ARGS else "http://localhost:8080") + "/api/v1"
EXPECT = (
    sys.argv[sys.argv.index("--expect-project") + 1] if "--expect-project" in sys.argv else None
)
jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def call(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    csrf = next((c.value for c in jar if c.name == "pt_csrf"), "")
    req = urllib.request.Request(  # noqa: S310 - URL del propio despliegue
        BASE + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "X-CSRF-Token": csrf},
    )
    with opener.open(req, timeout=120) as res:
        raw = res.read()
    return json.loads(raw) if raw else None


def wait_healthy(timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            call("GET", "/system/health")
            return
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(2)


def check_restored(name: str) -> None:
    projects = [p for p in call("GET", "/projects") if p["name"] == name]
    assert projects, f"no está el proyecto {name!r} después de restaurar"
    runs = call("GET", f"/projects/{projects[0]['id']}/runs")
    assert runs, "el proyecto restaurado no tiene runs"
    print(f"restauración OK: {name} con {len(runs)} run(s)")


def main() -> None:
    wait_healthy()
    call(
        "POST",
        "/auth/login",
        {
            "email": os.environ["PERCEPTRON_ADMIN_EMAIL"],
            "password": os.environ["PERCEPTRON_ADMIN_PASSWORD"],
        },
    )
    if EXPECT:
        check_restored(EXPECT)
        return
    pid = call("POST", "/projects", {"name": "Smoke cola"})["id"]
    src = call("POST", f"/projects/{pid}/sources", {"path": "/sources/demanda.csv"})
    dv = call("POST", f"/sources/{src['id']}/ingest", {"target": "unidades"})
    pipe = call("POST", f"/projects/{pid}/pipelines/propose", {"dataset_version_id": dv["id"]})
    arch = call(
        "POST",
        f"/projects/{pid}/arch/propose",
        {"dataset_version_id": dv["id"], "pipeline_id": pipe["id"]},
    )["proposals"][0]["archspec"]
    launch = call(
        "POST",
        f"/projects/{pid}/studies",
        {
            "dataset_version_id": dv["id"],
            "pipeline_id": pipe["id"],
            "archspec_id": arch["id"],
            "budget": {"max_trials": 1, "max_epochs_per_trial": 3},
        },
    )
    job = launch["job"]
    assert job["runner"] == "queue:cpu", job
    deadline = time.time() + 600
    while job["status"] not in ("succeeded", "failed", "cancelled"):
        assert time.time() < deadline, "el worker no terminó el estudio"
        time.sleep(2)
        job = call("GET", f"/jobs/{job['id']}")
    assert job["status"] == "succeeded", job.get("error")
    assert job["worker"] == "cpu-1", job
    run = call("GET", f"/runs/{job['result']['best_trial']['run_id']}")
    assert run["mlflow_run_id"], run
    queue = call("GET", "/server/queue")
    assert queue["mode"] == "queue" and [w["worker"] for w in queue["workers"]] == ["cpu-1"]
    print(f"smoke de la cola OK: {job['id']} en {job['worker']}, run {run['id']}")


if __name__ == "__main__":
    main()
