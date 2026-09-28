"""Retención de versiones de datos y export DVC (RF-MON-07)."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import polars as pl
from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.domain.models import Run

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _versions(client: TestClient, tmp_path: Path, fixtures_dir: Path) -> tuple[str, list[str]]:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Versiones"}), 201)["id"]
    base = pl.read_csv(fixtures_dir / "uc01_churn" / "churn.csv")
    ids = []
    for k in range(3):  # tres versiones distintas, de la más vieja a la más nueva
        path = tmp_path / f"churn v{k}.csv"
        base.head(300 + 20 * k).write_csv(path)
        src = _ok(client.post(f"{API}/projects/{pid}/sources", json={"path": str(path)}), 201)
        dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
        ids.append(dv["id"])
    return pid, ids


def test_retention_keeps_recent_and_in_use_versions(
    client: TestClient, ctx: EngineContext, tmp_path: Path, fixtures_dir: Path
) -> None:
    pid, (old, middle, new) = _versions(client, tmp_path, fixtures_dir)
    ctx.repo(Run).add(
        Run(project_id=pid, archspec_id="arc_x", pipeline_id="pip_x", dataset_version_id=old)
    )
    folder = ctx.settings.paths.project(pid).dataset(
        _ok(client.get(f"{API}/datasets/{middle}"))["content_hash"]
    )
    preview = _ok(client.post(f"{API}/projects/{pid}/datasets/retention", json={"keep_last": 1}))
    assert preview["dry_run"] and preview["kept"] == [new]
    assert preview["in_use"] == [old] and preview["deleted"] == [middle]
    assert preview["freed_bytes"] > 0 and folder.is_dir()  # la vista previa no borra
    done = _ok(
        client.post(
            f"{API}/projects/{pid}/datasets/retention", json={"keep_last": 1, "dry_run": False}
        )
    )
    assert done["deleted"] == [middle] and not folder.exists()
    assert client.get(f"{API}/datasets/{middle}").status_code == 404
    assert _ok(client.get(f"{API}/datasets/{old}"))["id"] == old


def test_dvc_export_has_files_and_a_dvc3_pointer(
    client: TestClient, ctx: EngineContext, tmp_path: Path, fixtures_dir: Path
) -> None:
    _, (_, _, new) = _versions(client, tmp_path, fixtures_dir)
    r = client.get(f"{API}/datasets/{new}/dvc.zip")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = z.namelist()
        pointer = next(n for n in names if n.endswith(".dvc"))
        text = z.read(pointer).decode("utf-8")
        name = pointer.removesuffix(".dvc")
        files = sorted(n for n in names if n.startswith(f"{name}/"))
        entries = sorted(
            (
                {
                    "md5": hashlib.md5(z.read(n), usedforsecurity=False).hexdigest(),
                    "relpath": n[len(name) + 1 :],
                }
                for n in files
            ),
            key=lambda e: e["relpath"],
        )
    raw = json.dumps(entries, sort_keys=True).encode("utf-8")
    assert f"md5: {hashlib.md5(raw, usedforsecurity=False).hexdigest()}.dir" in text
    assert f"nfiles: {len(files)}" in text and "hash: md5" in text and f"path: {name}" in text
