"""Inferencia de esquema y tipos semánticos (RF-ING-06).

El esquema describe columnas sin guardar valores individuales: se puede usar
como metadata en el ProfileCard (privacidad L1).
"""

from __future__ import annotations

import re
from enum import StrEnum

import polars as pl
from pydantic import BaseModel, Field

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".ogg"})

_TARGET_HINTS = re.compile(
    r"^(target|label|labels|y|clase|class|categoria|category|etiqueta|churn|"
    r"objetivo|resultado|outcome|anomalia|anomaly)$",
    re.IGNORECASE,
)
_ID_HINTS = re.compile(r"(^id$|_id$|^id_|^codigo|^code$|uuid|^nro|^numero)", re.IGNORECASE)
_PATH_SUFFIX = re.compile(
    r"\.(" + "|".join(e[1:] for e in sorted(IMAGE_EXTENSIONS | AUDIO_EXTENSIONS)) + r")$",
    re.IGNORECASE,
)

MAX_CATEGORICAL_UNIQUE = 50
TEXT_MIN_AVG_LEN = 30
TEXT_MIN_AVG_WORDS = 4


class SemanticType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    ID = "id"
    FILEPATH = "filepath"


class ColumnSchema(BaseModel):
    name: str
    dtype: str
    semantic: SemanticType
    nullable: bool
    n_unique: int
    id_like: bool = Field(
        default=False,
        description="Parece un identificador pero se usa como dato: es la única entrada",
    )


class TableSchema(BaseModel):
    columns: list[ColumnSchema]
    target: str | None = None
    target_candidates: list[str] = Field(default_factory=list)

    def column(self, name: str) -> ColumnSchema:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def feature_columns(self) -> list[ColumnSchema]:
        return [
            c for c in self.columns if c.name != self.target and c.semantic is not SemanticType.ID
        ]

    def with_overrides(self, overrides: dict[str, SemanticType]) -> TableSchema:
        """Corrección manual de tipos (RF-ING-06): la decisión de la persona manda."""
        cols = [
            c.model_copy(
                update={
                    "semantic": overrides[c.name],
                    # Confirmar el mismo tipo conserva el aviso; cambiarlo lo resuelve.
                    "id_like": c.id_like and overrides[c.name] is c.semantic,
                }
            )
            if c.name in overrides
            else c
            for c in self.columns
        ]
        return self.model_copy(update={"columns": cols})

    def keep_sole_inputs(self, target: str | None = None) -> TableSchema:
        """Si todas las columnas de entrada parecen ids, las numéricas se usan como dato.

        Un id se descarta para que el modelo no memorice filas, pero si es lo único que hay
        el modelo se queda sin entradas y solo aprende el promedio (p. ej. `numero → multiplo`).
        Quedan marcadas `id_like` para avisarlo; se puede corregir a mano.
        """
        target = target or self.target
        inputs = [c for c in self.columns if c.name != target]
        if target is None or not inputs or any(c.semantic is not SemanticType.ID for c in inputs):
            return self
        cols = [
            c.model_copy(update={"semantic": SemanticType.NUMERIC, "id_like": True})
            if c.name != target and _numeric_dtype(c.dtype)
            else c
            for c in self.columns
        ]
        return self.model_copy(update={"columns": cols})


DATETIME_FORMATS = (
    None,  # inferido por Polars
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y %H:%M",
)


def datetime_format(s: pl.Series) -> str | None:
    """Formato con el que parsea ≥ 90 % de la muestra ('' = inferido), o None si no es fecha."""
    sample = s.drop_nulls().head(200)
    if sample.is_empty() or not sample.str.contains(r"\d").all():
        return None
    for fmt in DATETIME_FORMATS:
        try:
            parsed = sample.str.to_datetime(format=fmt, strict=False)
        except pl.exceptions.PolarsError:
            continue
        if parsed.null_count() / len(sample) < 0.1:
            return fmt or ""
    return None


def _string_parses_as_datetime(s: pl.Series) -> bool:
    return datetime_format(s) is not None


def series_mean(s: pl.Series) -> float:
    """Media como float (0.0 si la serie está vacía o no es numérica)."""
    m = s.mean()
    return float(m) if isinstance(m, int | float) else 0.0


def _numeric_dtype(dtype: str) -> bool:
    return dtype.startswith(("Int", "UInt", "Float", "Decimal"))


def _is_consecutive(s: pl.Series) -> bool:
    """Enteros únicos que cubren un rango contiguo (1..n, 1000..1000+n): típico de un id."""
    if s.len() < 20:
        return False
    lo, hi = s.min(), s.max()
    return isinstance(lo, int) and isinstance(hi, int) and hi - lo + 1 == s.len()


def _infer_column(s: pl.Series, n_rows: int) -> SemanticType:
    dtype = s.dtype
    non_null = s.drop_nulls()
    n_unique = non_null.n_unique()
    all_unique = n_rows > 1 and n_unique == len(non_null) == n_rows

    if dtype == pl.Boolean:
        return SemanticType.BOOLEAN
    if dtype.is_temporal():
        return SemanticType.DATETIME
    id_named = bool(_ID_HINTS.search(s.name))
    if dtype.is_numeric():
        if dtype.is_integer() and all_unique and (id_named or _is_consecutive(non_null)):
            return SemanticType.ID
        if n_unique <= 2:
            return SemanticType.BOOLEAN
        return SemanticType.NUMERIC
    if dtype in (pl.String, pl.Categorical, pl.Enum):
        strs = non_null.cast(pl.String)
        if strs.is_empty():
            return SemanticType.CATEGORICAL
        path_ratio = series_mean(strs.head(200).str.contains(_PATH_SUFFIX.pattern))
        if path_ratio > 0.9:
            return SemanticType.FILEPATH
        if _string_parses_as_datetime(strs):
            return SemanticType.DATETIME
        avg_len = series_mean(strs.str.len_chars())
        if all_unique and (id_named or avg_len < TEXT_MIN_AVG_LEN / 2):
            return SemanticType.ID
        avg_words = series_mean(strs.str.split(" ").list.len())
        if avg_len >= TEXT_MIN_AVG_LEN or avg_words >= TEXT_MIN_AVG_WORDS:
            return SemanticType.TEXT
        if n_unique <= 2:
            return SemanticType.BOOLEAN
        if n_unique <= max(MAX_CATEGORICAL_UNIQUE, int(0.05 * n_rows)):
            return SemanticType.CATEGORICAL
        return SemanticType.TEXT
    return SemanticType.CATEGORICAL


def _target_candidates(columns: list[ColumnSchema]) -> list[str]:
    eligible = [
        c
        for c in columns
        if c.semantic in (SemanticType.BOOLEAN, SemanticType.CATEGORICAL, SemanticType.NUMERIC)
    ]
    hinted = [c.name for c in eligible if _TARGET_HINTS.match(c.name)]
    last = [columns[-1].name] if columns and columns[-1] in eligible else []
    return list(dict.fromkeys(hinted + last))


def infer_schema(df: pl.DataFrame, target: str | None = None) -> TableSchema:
    n_rows = df.height
    columns = [
        ColumnSchema(
            name=s.name,
            dtype=str(s.dtype),
            semantic=_infer_column(s, n_rows),
            nullable=s.null_count() > 0,
            n_unique=s.n_unique(),
        )
        for s in df.iter_columns()
    ]
    candidates = _target_candidates(columns)
    if target is not None and target not in df.columns:
        raise KeyError(f"la columna target '{target}' no existe")
    return TableSchema(columns=columns, target=target, target_candidates=candidates)
