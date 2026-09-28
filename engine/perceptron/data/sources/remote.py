"""Fuentes remotas (RF-ING-03, RF-ING-04): bases de datos SQL, Hugging Face Datasets y Kaggle.

Cada fuente se **materializa** en un archivo local dentro del proyecto (Parquet para SQL; los
archivos del dataset para HF/Kaggle). Así la vista previa, el perfil y la ingesta son los de
cualquier archivo, y cada ingesta queda fija en una versión inmutable. Las credenciales van
al almacén de secretos (keychain o archivo cifrado), nunca a la configuración de la fuente.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Literal

import httpx
import polars as pl
from pydantic import BaseModel, Field

from perceptron.core.errors import ValidationError

Dialect = Literal["sqlite", "postgresql", "mysql", "mssql"]
BATCH_ROWS = 50_000
MAX_DOWNLOAD_BYTES = 10 * 1024**3
_DRIVERS = {
    "postgresql": "postgresql+pg8000",
    "mysql": "mysql+pymysql",
    "mssql": "mssql+pyodbc",
}
_DATA_SUFFIXES = (".parquet", ".csv", ".jsonl", ".json", ".tsv")


class DbConfig(BaseModel):
    dialect: Dialect
    database: str = Field(description="Nombre de la base (o ruta del archivo en SQLite)")
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    user: str | None = None
    query: str = Field(min_length=1, description="Consulta SQL de solo lectura")
    max_rows: int = Field(default=5_000_000, ge=1)
    odbc_driver: str = "ODBC Driver 18 for SQL Server"


def db_url(cfg: DbConfig, password: str | None) -> Any:
    from sqlalchemy.engine import URL

    if cfg.dialect == "sqlite":
        path = Path(cfg.database)
        if not path.is_file():
            raise ValidationError(f"no existe la base SQLite {cfg.database}")
        return URL.create("sqlite", database=str(path))
    query = (
        {"driver": cfg.odbc_driver, "TrustServerCertificate": "yes"}
        if cfg.dialect == "mssql"
        else {}
    )
    return URL.create(
        _DRIVERS[cfg.dialect],
        username=cfg.user,
        password=password,
        host=cfg.host,
        port=cfg.port,
        database=cfg.database,
        query=query,
    )


def materialize_db(cfg: DbConfig, password: str | None, dest: Path) -> int:
    """Ejecuta la consulta por lotes y la guarda como Parquet. Devuelve las filas leídas."""
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url(cfg, password))
    parts_dir = dest.parent / f".{dest.stem}-parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)
    rows = 0
    try:
        # Solo lectura por intención: la transacción se descarta al terminar.
        with engine.connect() as conn, conn.begin() as tx:
            # Cursor de SQLAlchemy por lotes (sin el camino asyncio de polars.read_database).
            result = conn.execution_options(stream_results=True).execute(text(cfg.query))
            columns = list(result.keys())
            k = 0
            while rows < cfg.max_rows:
                chunk = result.fetchmany(min(BATCH_ROWS, cfg.max_rows - rows))
                if not chunk:
                    break
                batch = pl.DataFrame(
                    [tuple(r) for r in chunk],
                    schema=columns,
                    orient="row",
                    infer_schema_length=None,
                )
                batch.write_parquet(parts_dir / f"part-{k:05d}.parquet")
                rows += batch.height
                k += 1
            result.close()
            tx.rollback()
    except ValidationError:
        raise
    except Exception as exc:  # errores del driver: se informan sin la URL (tiene credenciales)
        raise ValidationError(
            f"no se pudo leer la base: {type(exc).__name__}: {exc}"[:500]
        ) from None
    finally:
        engine.dispose()
    parts = sorted(parts_dir.glob("*.parquet"))
    if not parts:
        raise ValidationError("la consulta no devolvió filas")
    # Los tipos pueden diferir entre lotes (p. ej. nulos al principio): unión flexible.
    pl.concat([pl.scan_parquet(p) for p in parts], how="diagonal_relaxed").sink_parquet(dest)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return rows


# ------------------------------------------------------------------ Hugging Face y Kaggle


def _offline() -> None:
    if os.environ.get("PERCEPTRON_OFFLINE", "").lower() in {"1", "true", "yes"}:
        raise ValidationError("sin conexión (PERCEPTRON_OFFLINE): no se pueden descargar datasets")


def hf_data_files(files: list[str], split: str | None) -> list[str]:
    """Archivos de datos del split pedido (prefiere Parquet; si no, CSV/JSONL)."""
    data = [
        f for f in files if f.lower().endswith(_DATA_SUFFIXES) and not Path(f).name.startswith(".")
    ]
    if split:
        data = [
            f for f in data if split in Path(f).as_posix().split("/")[-1] or f"/{split}/" in f"/{f}"
        ]
    for suffix in _DATA_SUFFIXES:
        chosen = [f for f in data if f.lower().endswith(suffix)]
        if chosen:
            return sorted(chosen)
    return []


def _hf_list(repo_id: str, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    return list(HfApi(token=token).list_repo_files(repo_id, repo_type="dataset"))


def _hf_download(repo_id: str, filename: str, token: str | None, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id, filename, repo_type="dataset", token=token, local_dir=dest)
    )


def download_hf(repo_id: str, split: str | None, token: str | None, dest_dir: Path) -> Path:
    """Descarga los archivos de datos de un dataset del Hub y devuelve un archivo tabular."""
    _offline()
    files = hf_data_files(_hf_list(repo_id, token), split)
    if not files:
        raise ValidationError(
            f"el dataset {repo_id} no tiene archivos de datos para el split {split!r}"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = [_hf_download(repo_id, f, token, dest_dir) for f in files]
    if len(local) == 1:
        return local[0]
    # Varias partes del mismo formato: un único Parquet.
    frames = [pl.scan_parquet(p) if p.suffix == ".parquet" else pl.scan_csv(p) for p in local]
    out = dest_dir / "data.parquet"
    pl.concat(frames, how="diagonal_relaxed").sink_parquet(out)
    return out


def _http_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(60.0, read=600.0), follow_redirects=True)


def download_kaggle(ref: str, credentials: str | None, dest_dir: Path) -> Path:
    """Descarga un dataset de Kaggle (`owner/slug`) con la API oficial; devuelve el zip."""
    _offline()
    if ref.count("/") != 1:
        raise ValidationError("el dataset de Kaggle va como owner/slug")
    if not credentials or ":" not in credentials:
        raise ValidationError("Kaggle requiere credenciales «usuario:clave» (kaggle.json)")
    user, key = credentials.split(":", 1)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{ref.replace('/', '__')}.zip"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"
    size = 0
    with _http_client() as client, client.stream("GET", url, auth=(user, key)) as resp:
        if resp.status_code in (401, 403):
            raise ValidationError("Kaggle rechazó las credenciales o el acceso al dataset")
        if resp.status_code == 404:
            raise ValidationError(f"no existe el dataset de Kaggle {ref}")
        resp.raise_for_status()
        with out.open("wb") as f:
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValidationError("el dataset supera el máximo de 10 GB")
                f.write(chunk)
    return out
