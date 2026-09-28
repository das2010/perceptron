"""Diff y linaje entre versiones de datos (RF-MON-07).

- **Filas**: hash de cada fila (sin la columna de split) → agregadas, eliminadas y comunes.
- **Esquema**: columnas agregadas, eliminadas y cambios de tipo.
- **Distribución**: las mismas métricas del drift (PSI, KS, χ², JS) columna por columna.
- **Archivos**: diff del manifiesto (imágenes, audio, anotaciones…).
- **Linaje**: cadena de padres con la transformación que produjo cada versión.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from perceptron.data.splits import SPLIT_COLUMN
from perceptron.data.view import MANIFEST_FILE, DatasetView, Purpose
from perceptron.domain.models import DatasetVersion
from perceptron.monitoring.drift import DataDrift, data_drift
from perceptron.storage.manifest import Manifest

MAX_CATEGORIES = 50
SAMPLE_KEYS = 20


class SchemaChange(BaseModel):
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    retyped: dict[str, list[str]] = Field(
        default_factory=dict, description="Columna → [tipo en A, tipo en B]"
    )


class DatasetDiff(BaseModel):
    a: str
    b: str
    rows_a: int
    rows_b: int
    rows_added: int
    rows_removed: int
    rows_common: int
    sample_added: list[dict[str, Any]] = Field(default_factory=list)
    schema_change: SchemaChange
    distribution: DataDrift | None = None
    files: dict[str, list[str]] = Field(default_factory=dict)
    target_changed: bool = False


class LineageNode(BaseModel):
    id: str
    parent_id: str | None
    transformation: str | None
    content_hash: str
    num_samples: int
    created_at: str


def _row_hashes(df: pl.DataFrame) -> pl.Series:
    cols = sorted(c for c in df.columns if c != SPLIT_COLUMN)
    return df.select(cols).hash_rows(seed=0)


def _kinds(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for name, dtype in df.schema.items():
        if name == SPLIT_COLUMN:
            continue
        if dtype.is_numeric():
            numeric.append(name)
        elif (
            dtype in (pl.Utf8, pl.Categorical, pl.Boolean) and df[name].n_unique() <= MAX_CATEGORIES
        ):
            categorical.append(name)
    return numeric, categorical


def _manifest(root: Path) -> Manifest | None:
    path = root / MANIFEST_FILE
    if not path.is_file():
        return None
    return Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def dataset_diff(
    a: DatasetVersion, va: DatasetView, b: DatasetVersion, vb: DatasetView
) -> DatasetDiff:
    """Diff de B respecto de A. Lee todas las filas (auditoría: no entrena ni elige)."""
    da = va.read(purpose=Purpose.VERSIONING)
    db = vb.read(purpose=Purpose.VERSIONING)
    common_cols = [c for c in da.columns if c in db.columns and c != SPLIT_COLUMN]
    ha = set(_row_hashes(da.select(common_cols)).to_list()) if common_cols else set()
    hb_series = (
        _row_hashes(db.select(common_cols)) if common_cols else pl.Series([], dtype=pl.UInt64)
    )
    hb = set(hb_series.to_list())
    added_mask = [h not in ha for h in hb_series.to_list()]
    sample = db.filter(pl.Series(added_mask)).drop(SPLIT_COLUMN, strict=False).head(SAMPLE_KEYS)
    schema = SchemaChange(
        added=[c for c in db.columns if c not in da.columns and c != SPLIT_COLUMN],
        removed=[c for c in da.columns if c not in db.columns and c != SPLIT_COLUMN],
        retyped={
            c: [str(da.schema[c]), str(db.schema[c])]
            for c in common_cols
            if da.schema[c] != db.schema[c]
        },
    )
    numeric, categorical = _kinds(da.select(common_cols)) if common_cols else ([], [])
    distribution = data_drift(da, db, numeric, categorical) if common_cols else None
    ma, mb = _manifest(va.root), _manifest(vb.root)
    files = ma.diff(mb) if ma and mb else {}
    return DatasetDiff(
        a=a.id,
        b=b.id,
        rows_a=da.height,
        rows_b=db.height,
        rows_added=sum(added_mask),
        rows_removed=len(ha - hb),
        rows_common=len(ha & hb),
        sample_added=sample.to_dicts(),
        schema_change=schema,
        distribution=distribution,
        files=files,
        target_changed=a.target != b.target,
    )


def lineage(versions: dict[str, DatasetVersion], start: str) -> list[LineageNode]:
    """Cadena desde `start` hasta la raíz (sin ciclos)."""
    chain: list[LineageNode] = []
    seen: set[str] = set()
    current: str | None = start
    while current and current in versions and current not in seen:
        dv = versions[current]
        seen.add(current)
        chain.append(
            LineageNode(
                id=dv.id,
                parent_id=dv.parent_id,
                transformation=dv.transformation,
                content_hash=dv.content_hash,
                num_samples=dv.num_samples,
                created_at=dv.created_at.isoformat(),
            )
        )
        current = dv.parent_id
    return chain
