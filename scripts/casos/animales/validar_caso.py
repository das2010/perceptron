"""Corre el caso UC-11 («Identificación de animales en cámaras del campo») contra un Team
Server y califica los criterios de aceptación. Solo biblioteca estándar.

Requisitos:
- el caso armado con `preparar_caso.py` dentro de una fuente del servidor (p. ej. la carpeta
  `fixtures/` que el Compose monta como `/sources`);
- credenciales en variables de entorno: PERCEPTRON_EMAIL y PERCEPTRON_PASSWORD (no se imprimen).

Uso:
    python scripts/casos/animales/validar_caso.py --url http://localhost:8080 \\
        --caso fixtures/caso_animales --fuente /sources/caso_animales --salida resultado.json
"""

from __future__ import annotations

import argparse
import csv
import http.cookiejar
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

SPECIES = ["Cat", "Cow", "Deer", "Dog", "Goat", "Hen", "Rabbit", "Sheep"]
CRITERIA = {
    "test_accuracy": 0.85,  # accuracy en el test sellado
    "min_recall": 0.70,  # recall de la peor especie en test
    "prod_accuracy": 0.85,  # accuracy sobre las fotos de día que llegan al deployment
    "zero_shot_accuracy": 0.85,  # acierto del pre-etiquetado zero-shot sin entrenar
    "day_drift": "low",  # severidad máxima aceptable con fotos de día
    "night_drift": "medium",  # severidad mínima esperada con fotos nocturnas
    "llm_cost_usd": 1.0,  # gasto máximo del LLM en el caso
}
RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


class Client:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/") + "/api/v1"
        self.jar = http.cookiejar.CookieJar()
        self.http = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))

    def _csrf(self) -> str | None:
        return next((c.value for c in self.jar if c.name == "pt_csrf"), None)

    def call(self, method: str, path: str, body: Any = None, *, files: Any = None) -> Any:
        headers: dict[str, str] = {}
        data = None
        if files is not None:
            boundary = uuid.uuid4().hex
            parts = [
                f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
                f'filename="{name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
                + content
                + b"\r\n"
                for field, (name, content, ctype) in files
            ]
            data = b"".join(parts) + f"--{boundary}--\r\n".encode()
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if method != "GET" and self._csrf():
            headers["X-CSRF-Token"] = self._csrf() or ""
        # La URL la pasa quien corre el caso (http/https), no sale de datos externos.
        req = urllib.request.Request(  # noqa: S310
            self.base + path, data=data, method=method, headers=headers
        )
        # Las lecturas se reintentan si se corta la conexión (con el equipo cargado, el proxy
        # de puertos de Docker Desktop a veces corta una); las escrituras no, para no duplicar.
        for attempt in range(5 if method == "GET" else 1):
            try:
                with self.http.open(req, timeout=3600) as res:
                    raw, status = res.read(), res.status
                break
            except urllib.error.HTTPError as e:
                raw, status = e.read(), e.code
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == (4 if method == "GET" else 0):
                    raise
                time.sleep(5 * (attempt + 1))
        payload = json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode("utf-8", "replace")
        if status >= 400:
            raise RuntimeError(f"{method} {path} → {status}: {str(payload)[:500]}")
        return payload

    def wait_job(self, job_id: str, timeout: float = 7200) -> dict[str, Any]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            job: dict[str, Any] = self.call("GET", f"/jobs/{job_id}")
            if job["status"] in ("succeeded", "failed", "cancelled"):
                if job["status"] != "succeeded":
                    raise RuntimeError(f"job {job['status']}: {json.dumps(job.get('error'))[:500]}")
                return job
            time.sleep(5)
        raise RuntimeError(f"el job {job_id} no terminó en {timeout:.0f} s")


def answers(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["file_name"]: r["especie"] for r in csv.DictReader(fh)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--caso", type=Path, default=Path("fixtures/caso_animales"))
    ap.add_argument("--fuente", default="/sources/caso_animales", help="ruta en el servidor")
    ap.add_argument("--salida", type=Path, default=Path("resultado_caso_animales.json"))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--epocas", type=int, default=6)
    ap.add_argument(
        "--reanudar",
        action="store_true",
        help="retoma la corrida de --salida (proyecto, datos y estudio en curso)",
    )
    args = ap.parse_args(argv)

    c = Client(args.url)
    c.call(
        "POST",
        "/auth/login",
        {"email": os.environ["PERCEPTRON_EMAIL"], "password": os.environ["PERCEPTRON_PASSWORD"]},
    )
    report: dict[str, Any] = {
        "pasos": [],
        "criterios": {},
        "inicio": time.strftime("%Y-%m-%d %H:%M"),
    }
    state: dict[str, Any] = {}
    done: set[str] = set()
    if args.reanudar and args.salida.is_file():
        previous = json.loads(args.salida.read_text(encoding="utf-8"))
        state.update(previous.get("estado", {}))
        report["criterios"].update(previous.get("criterios", {}))
        report["pasos"] = [p for p in previous.get("pasos", []) if p.get("ok")]
        done = {p["paso"] for p in report["pasos"]}

    def step(name: str, fn: Any) -> Any:
        if name in done:
            print(f"= {name}: ya hecho", flush=True)
            return None
        t0 = time.time()
        try:
            info = fn()
            report["pasos"].append(
                {"paso": name, "ok": True, "s": round(time.time() - t0, 1), "info": info}
            )
            print(
                f"✔ {name} ({time.time() - t0:.0f} s): "
                f"{json.dumps(info, ensure_ascii=False)[:300]}",
                flush=True,
            )
            return info
        except Exception as exc:
            report["pasos"].append(
                {
                    "paso": name,
                    "ok": False,
                    "s": round(time.time() - t0, 1),
                    "error": str(exc)[:800],
                }
            )
            print(f"✘ {name}: {exc}", flush=True)
            return None
        finally:
            report["estado"] = state
            args.salida.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def project() -> Any:
        p = c.call(
            "POST",
            "/projects",
            {
                "name": f"UC-11 animales {time.strftime('%d/%m %H:%M')}",
                "goal": "Identificar la especie del animal que aparece en cada foto de las "
                "cámaras del campo (8 especies), para contar animales por corral y avisar si "
                "entra un perro o un gato donde no debe.",
            },
        )
        state["pid"] = p["id"]
        return {"proyecto": p["id"]}

    def data() -> Any:
        src = c.call(
            "POST", f"/projects/{state['pid']}/sources", {"path": f"{args.fuente}/entrenamiento"}
        )
        prev = c.call("POST", f"/sources/{src['id']}/preview")
        dv = c.call("POST", f"/sources/{src['id']}/ingest", {})
        state["dv"] = dv["id"]
        card = c.call("POST", f"/datasets/{dv['id']}/profile")
        return {
            "muestras": dv["num_samples"],
            "clases": prev.get("classes"),
            "alertas": [a.get("code") for a in card.get("alerts", [])],
        }

    def design() -> Any:
        pipe = c.call(
            "POST",
            f"/projects/{state['pid']}/pipelines/propose",
            {"dataset_version_id": state["dv"]},
        )
        state["pipe"] = pipe["id"]
        res = c.call(
            "POST",
            f"/projects/{state['pid']}/arch/propose",
            {"dataset_version_id": state["dv"], "pipeline_id": pipe["id"], "mode": "auto", "n": 3},
        )
        state["arch"] = res["proposals"][0]["archspec"]["id"]
        strat = c.call(
            "POST",
            f"/projects/{state['pid']}/hpo/strategy",
            {
                "archspec_id": state["arch"],
                "dataset_version_id": state["dv"],
                "mode": "auto",
                "budget": {"max_trials": args.trials, "max_epochs_per_trial": args.epocas},
            },
        )
        state["strategy"] = strat
        return {
            "origen": res.get("origin"),
            "arquitecturas": [p.get("title") for p in res["proposals"]],
            "estrategia": strat.get("strategy"),
        }

    def train() -> Any:
        budget = {"max_trials": args.trials, "max_epochs_per_trial": args.epocas}
        if "strategy" not in state:  # al reanudar sin la estrategia guardada
            state["strategy"] = c.call(
                "POST",
                f"/projects/{state['pid']}/hpo/strategy",
                {
                    "archspec_id": state["arch"],
                    "dataset_version_id": state["dv"],
                    "mode": "auto",
                    "budget": budget,
                },
            )
        if "job" not in state:  # al reanudar, se espera el estudio que ya estaba en curso
            launch = c.call(
                "POST",
                f"/projects/{state['pid']}/studies",
                {
                    "dataset_version_id": state["dv"],
                    "pipeline_id": state["pipe"],
                    "archspec_id": state["arch"],
                    "strategy": {**state["strategy"], "budget": budget},
                    "budget": budget,
                },
            )
            state["job"] = launch["job"]["id"]
            args.salida.write_text(
                json.dumps({**report, "estado": state}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        best = c.wait_job(state["job"])["result"]["best_trial"]
        state["run"] = best["run_id"]
        return {
            "mejor_run": best["run_id"],
            "params": best.get("params"),
            "val_loss": best.get("values"),
        }

    def evaluate() -> Any:
        ev = c.call("POST", f"/runs/{state['run']}/evaluate")
        m = ev.get("metrics", {})
        per_class = (ev.get("classification") or {}).get("per_class") or []
        recalls = {x.get("label"): x.get("recall") for x in per_class}
        report["criterios"]["test_accuracy"] = m.get("accuracy")
        report["criterios"]["min_recall"] = min(
            (r for r in recalls.values() if r is not None), default=None
        )
        return {
            "accuracy": m.get("accuracy"),
            "f1_macro": m.get("f1_macro"),
            "recall_por_especie": recalls,
        }

    def explain() -> Any:
        img = next((args.caso / "entrenamiento" / "Dog").glob("*.jpg"))
        res = c.call(
            "POST",
            f"/runs/{state['run']}/explain/image",
            files=[("file", (img.name, img.read_bytes(), "image/jpeg"))],
        )
        return {
            "metodo": res["method"],
            "prediccion": res["prediction"],
            "mapa_de_calor": bool(res.get("heatmap_png")),
        }

    def report_llm() -> Any:
        r = c.call("POST", f"/runs/{state['run']}/report", {"mode": "auto", "language": "español"})
        return {"origen": r.get("origin"), "caracteres": len(r.get("markdown") or "")}

    def deploy() -> Any:
        ex = c.call("POST", f"/runs/{state['run']}/export", {"formats": ["onnx"]})
        c.wait_job(ex["job"]["id"])
        mv = c.call("POST", f"/runs/{state['run']}/register")
        dep = c.call(
            "POST",
            f"/models/{mv['id']}/deployments",
            {"name": "camaras-campo", "monitoring": {"window": 100000, "min_labels": 30}},
        )
        state["dep"] = dep["id"]
        return {"deployment": dep["id"]}

    def predict(folder: Path) -> list[tuple[str, dict[str, Any]]]:
        files = sorted(folder.glob("*.jpg"))
        out: list[tuple[str, dict[str, Any]]] = []
        for i in range(0, len(files), 32):
            batch = files[i : i + 32]
            res = c.call(
                "POST",
                f"/deployments/{state['dep']}/predict/file",
                files=[("files", (f.name, f.read_bytes(), "image/jpeg")) for f in batch],
            )
            out += list(zip([f.name for f in batch], res["predictions"], strict=True))
        return out

    def day() -> Any:
        truth = answers(args.caso / "respuestas" / "produccion_dia.csv")
        preds = predict(args.caso / "produccion_dia")
        hits = sum(1 for name, p in preds if str(p["prediction"]) == truth[name])
        report["criterios"]["prod_accuracy"] = hits / len(preds)
        # La etiqueta real llega después (feedback) y permite medir la performance en uso.
        c.call(
            "POST",
            f"/deployments/{state['dep']}/feedback",
            {
                "items": [
                    {"prediction_id": p["prediction_id"], "label": truth[name]} for name, p in preds
                ]
            },
        )
        rep = c.call("POST", f"/deployments/{state['dep']}/check?last={len(preds)}")
        emb = rep["metrics"].get("embedding") or {}
        report["criterios"]["day_drift"] = rep["severity"]
        perf = rep["metrics"].get("performance") or {}
        return {
            "accuracy": round(hits / len(preds), 3),
            "drift": rep["severity"],
            "auc_dominio": emb.get("domain_auc"),
            "performance_feedback": perf.get("current"),
        }

    def night() -> Any:
        preds = predict(args.caso / "produccion_noche")
        rep = c.call("POST", f"/deployments/{state['dep']}/check?last={len(preds)}")
        emb = rep["metrics"].get("embedding") or {}
        report["criterios"]["night_drift"] = rep["severity"]
        alerts = c.call("GET", f"/projects/{state['pid']}/alerts")
        return {
            "drift": rep["severity"],
            "auc_dominio": emb.get("domain_auc"),
            "alertas": [(a["kind"], a["severity"]) for a in alerts],
        }

    def labeling() -> Any:
        src = c.call(
            "POST", f"/projects/{state['pid']}/sources", {"path": f"{args.fuente}/sin_etiquetar"}
        )
        dv = c.call("POST", f"/sources/{src['id']}/ingest", {})
        ls = c.call(
            "POST", f"/datasets/{dv['id']}/labelsets", {"kind": "class", "classes": SPECIES}
        )
        n = c.call(
            "POST", f"/labelsets/{ls['id']}/prelabel", {"method": "zero_shot", "limit": 200}
        )["count"]
        truth = answers(args.caso / "respuestas" / "sin_etiquetar.csv")
        queue = c.call("GET", f"/labelsets/{ls['id']}/queue?limit=200")
        judged = [(s["path"].split("/")[-1], s["item"]["label"]) for s in queue if s.get("item")]
        hits = sum(1 for name, label in judged if truth.get(name) == label)
        report["criterios"]["zero_shot_accuracy"] = hits / max(1, len(judged))
        accepted = c.call("POST", f"/labelsets/{ls['id']}/accept", {"min_confidence": 0.9})["count"]
        return {
            "sugeridas": n,
            "acierto": round(hits / max(1, len(judged)), 3),
            "aceptadas_auto": accepted,
        }

    def cost() -> Any:
        audit = c.call("GET", f"/llm/audit?project={state['pid']}")
        items = audit if isinstance(audit, list) else audit.get("items", [])
        total = sum(float(x.get("cost_usd") or 0) for x in items)
        report["criterios"]["llm_cost_usd"] = round(total, 4)
        return {"llamadas": len(items), "usd": round(total, 4)}

    # (paso, función, lo que necesita del estado): si falta, se saltea en vez de fallar en cadena.
    for name, fn, needs in [
        ("proyecto", project, ()),
        ("datos y perfil", data, ("pid",)),
        ("diseño con el LLM", design, ("dv",)),
        ("entrenamiento en el worker", train, ("arch",)),
        ("evaluación en test sellado", evaluate, ("run",)),
        ("explicación (mapa de calor)", explain, ("run",)),
        ("informe del LLM", report_llm, ("run",)),
        ("export, registro y deployment", deploy, ("run",)),
        ("producción de día + feedback", day, ("dep",)),
        ("producción de noche", night, ("dep",)),
        ("etiquetado asistido zero-shot", labeling, ("pid",)),
        ("gasto del LLM", cost, ("pid",)),
    ]:
        missing = [k for k in needs if k not in state]
        if missing:
            report["pasos"].append(
                {"paso": name, "ok": False, "error": f"salteado: falta {missing}"}
            )
            print(f"– {name}: salteado (falta un paso anterior)", flush=True)
            continue
        step(name, fn)

    got = report["criterios"]
    verdict = {
        "test_accuracy": (got.get("test_accuracy") or 0) >= CRITERIA["test_accuracy"],
        "min_recall": (got.get("min_recall") or 0) >= CRITERIA["min_recall"],
        "prod_accuracy": (got.get("prod_accuracy") or 0) >= CRITERIA["prod_accuracy"],
        "zero_shot_accuracy": (got.get("zero_shot_accuracy") or 0)
        >= CRITERIA["zero_shot_accuracy"],
        "day_drift": RANK.get(got.get("day_drift", "high"), 3) <= RANK[CRITERIA["day_drift"]],
        "night_drift": RANK.get(got.get("night_drift", "none"), 0) >= RANK[CRITERIA["night_drift"]],
        "llm_cost_usd": (got.get("llm_cost_usd") if got.get("llm_cost_usd") is not None else 99)
        <= CRITERIA["llm_cost_usd"],
    }
    report["veredicto"] = verdict
    report["aprobado"] = all(verdict.values()) and all(p["ok"] for p in report["pasos"])
    args.salida.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nCriterios:")
    for k, ok in verdict.items():
        print(f"  {'✔' if ok else '✘'} {k}: {got.get(k)} (objetivo {CRITERIA[k]})")
    print("CASO APROBADO" if report["aprobado"] else "CASO NO APROBADO")
    return 0 if report["aprobado"] else 1


if __name__ == "__main__":
    sys.exit(main())
