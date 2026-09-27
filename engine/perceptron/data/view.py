"""Acceso de solo lectura a una `DatasetVersion` materializada (ADR-0016).

El split de test está sellado (RF-ING-08, RF-AGT-02): solo se entrega con
`purpose=Purpose.FINAL_EVALUATION`. HPO, entrenamiento y el agente usan
train/val.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import polars as pl

from perceptron.core.errors import NotFoundError, PerceptronError
from perceptron.data.schema import TableSchema
from perceptron.data.series import SeriesConfig
from perceptron.data.splits import SPLIT_COLUMN, TEST
from perceptron.domain.enums import Modality

TABLE_FILE = "table.parquet"
INDEX_FILE = "index.parquet"
SCHEMA_FILE = "schema.json"
META_FILE = "dataset.json"
MANIFEST_FILE = "manifest.json"
FILES_DIR = "files"


TABLE_MODALITIES = frozenset({Modality.TABULAR, Modality.TEXT, Modality.TIMESERIES})


class Purpose(StrEnum):
    TRAINING = "training"
    PROFILING = "profiling"
    FINAL_EVALUATION = "final_evaluation"


class SealedTestSetError(PerceptronError):
    code = "sealed_test_set"
    http_status = 403


class DatasetView:
    def __init__(self, root: Path) -> None:
        if not (root / META_FILE).is_file():
            raise NotFoundError(f"no es un dataset materializado: {root}")
        self.root = root
        self.meta: dict[str, object] = json.loads((root / META_FILE).read_text(encoding="utf-8"))
        self.modality = Modality(str(self.meta["modality"]))
        self.schema = TableSchema.model_validate_json(
            (root / SCHEMA_FILE).read_text(encoding="utf-8")
        )

    @property
    def target(self) -> str | None:
        return self.schema.target

    @property
    def data_file(self) -> Path:
        return self.root / (TABLE_FILE if self.modality in TABLE_MODALITIES else INDEX_FILE)

    @property
    def series(self) -> SeriesConfig | None:
        raw = self.meta.get("series")
        return SeriesConfig.model_validate(raw) if raw else None

    @property
    def text_column(self) -> str | None:
        col = self.meta.get("text_column")
        return str(col) if col else None

    @property
    def files_dir(self) -> Path:
        return self.root / FILES_DIR

    def scan(
        self, split: str | None = None, *, purpose: Purpose = Purpose.TRAINING
    ) -> pl.LazyFrame:
        """Filas del split pedido (todas las no-test si `split` es None)."""
        lf = pl.scan_parquet(self.data_file)
        if split == TEST:
            if purpose is not Purpose.FINAL_EVALUATION:
                raise SealedTestSetError(
                    "El test set está sellado: solo la evaluación final puede leerlo",
                    details={"purpose": purpose.value},
                )
            return lf.filter(pl.col(SPLIT_COLUMN) == TEST)
        if split is not None:
            return lf.filter(pl.col(SPLIT_COLUMN) == split)
        if purpose is Purpose.FINAL_EVALUATION:
            return lf
        return lf.filter(pl.col(SPLIT_COLUMN) != TEST)

    def read(
        self, split: str | None = None, *, purpose: Purpose = Purpose.TRAINING
    ) -> pl.DataFrame:
        return self.scan(split, purpose=purpose).collect()
