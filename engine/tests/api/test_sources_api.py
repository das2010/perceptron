"""Fuentes remotas por la API (RF-ING-03 bases SQL, RF-ING-04 Hugging Face y Kaggle)."""

from __future__ import annotations

import csv
import io
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from perceptron.data.sources import remote

API = "/api/v1"
# Almacén de secretos en memoria (el runner de CI no tiene llavero del sistema).
pytestmark = pytest.mark.usefixtures("fake_llm")


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _project(client: TestClient) -> str:
    pid: str = _ok(client.post(f"{API}/projects", json={"name": "Fuentes"}), 201)["id"]
    return pid


def _churn_db(fixtures_dir: Path, path: Path) -> int:
    with (fixtures_dir / "uc01_churn" / "churn.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    con = sqlite3.connect(path)
    cols = list(rows[0])
    con.execute(f"CREATE TABLE clientes ({', '.join(c + ' TEXT' for c in cols)})")
    con.executemany(
        f"INSERT INTO clientes VALUES ({', '.join('?' for _ in cols)})",  # noqa: S608 - columnas propias del test
        [tuple(r[c] for c in cols) for r in rows],
    )
    con.commit()
    con.close()
    return sum(1 for r in rows if r["plan"] == "premium")


def test_sqlite_source_preview_ingest_refresh(
    client: TestClient, fixtures_dir: Path, tmp_path: Path
) -> None:
    db = tmp_path / "crm con espacio ñ.sqlite"
    premium = _churn_db(fixtures_dir, db)
    pid = _project(client)
    body = {
        "name": "Clientes premium",
        "config": {
            "dialect": "sqlite",
            "database": str(db),
            "query": "SELECT * FROM clientes WHERE plan = 'premium'",
        },
        "password": "s3creto",
    }
    resp = client.post(f"{API}/projects/{pid}/sources/db", json=body)
    src = _ok(resp, 201)
    assert "s3creto" not in resp.text  # la clave va al almacén de secretos
    assert src["type"] == "db" and src["config"]["rows"] == premium
    assert src["secret_refs"] == {"password": f"source/{src['id']}/credentials"}

    preview = _ok(client.post(f"{API}/sources/{src['id']}/preview?limit=5"))
    assert preview["kind"] == "table" and "churn" in preview["columns"]
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    assert dv["num_samples"] == premium

    con = sqlite3.connect(db)
    con.execute("UPDATE clientes SET plan = 'premium' WHERE plan = 'básico'")
    con.commit()
    con.close()
    refreshed = _ok(client.post(f"{API}/sources/{src['id']}/refresh"))
    assert refreshed["config"]["rows"] > premium
    dv2 = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    assert dv2["id"] != dv["id"] and dv2["num_samples"] == refreshed["config"]["rows"]

    bad = {**body, "config": {**body["config"], "query": "SELECT * FROM no_existe"}}
    assert client.post(f"{API}/projects/{pid}/sources/db", json=bad).status_code == 422


def test_huggingface_source(
    client: TestClient, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PERCEPTRON_OFFLINE", raising=False)
    churn = fixtures_dir / "uc01_churn" / "churn.csv"
    seen: dict[str, Any] = {}

    def fake_list(repo_id: str, token: str | None) -> list[str]:
        seen["token"] = token
        return ["README.md", "data/train.csv", "data/test.csv", ".gitattributes"]

    def fake_download(repo_id: str, filename: str, token: str | None, dest: Path) -> Path:
        seen.setdefault("files", []).append(filename)
        out = dest / Path(filename).name
        out.write_bytes(churn.read_bytes())
        return out

    monkeypatch.setattr(remote, "_hf_list", fake_list)
    monkeypatch.setattr(remote, "_hf_download", fake_download)
    pid = _project(client)
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources/hub",
            json={
                "provider": "huggingface",
                "dataset": "org/churn",
                "split": "train",
                "token": "hf_x",
            },
        ),
        201,
    )
    assert seen == {"token": "hf_x", "files": ["data/train.csv"]}
    assert src["type"] == "hf" and "hf_x" not in str(src)
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    assert dv["num_samples"] > 0


def test_kaggle_source(
    client: TestClient, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PERCEPTRON_OFFLINE", raising=False)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as z:
        z.write(fixtures_dir / "uc01_churn" / "churn.csv", "churn.csv")

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/v1/datasets/download/owner/churn"
        assert req.headers["authorization"].startswith("Basic ")
        return httpx.Response(200, content=payload.getvalue())

    monkeypatch.setattr(
        remote, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    pid = _project(client)
    no_creds = client.post(
        f"{API}/projects/{pid}/sources/hub", json={"provider": "kaggle", "dataset": "owner/churn"}
    )
    assert no_creds.status_code == 422
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources/hub",
            json={"provider": "kaggle", "dataset": "owner/churn", "token": "yo:clave"},
        ),
        201,
    )
    assert src["type"] == "kaggle"
    preview = _ok(client.post(f"{API}/sources/{src['id']}/preview"))
    assert "churn" in preview["columns"]


def test_hf_file_selection() -> None:
    files = ["a/train-00000.parquet", "a/train-00001.parquet", "a/test.parquet", "b.csv"]
    assert remote.hf_data_files(files, "train") == [
        "a/train-00000.parquet",
        "a/train-00001.parquet",
    ]
    assert remote.hf_data_files(["x/test/data.csv"], "test") == ["x/test/data.csv"]
    assert remote.hf_data_files(["README.md"], None) == []


def test_secret_store_without_keychain_is_a_clear_error() -> None:
    from perceptron.core.errors import ValidationError
    from perceptron.llm.secrets import ChainSecrets, EnvSecrets

    class Broken:
        def get(self, name: str) -> str | None:
            return None

        def set(self, name: str, value: str) -> None:
            raise RuntimeError("sin backend")

        def delete(self, name: str) -> None:
            return None

    with pytest.raises(ValidationError, match="llavero"):
        ChainSecrets([Broken(), EnvSecrets()]).set("x", "y")
