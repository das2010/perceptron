"""Dataset Profile Card (RF-PRF-07).

Es la entrada principal del LLM en privacidad L1 (SPEC §7.7.3), así que **no
contiene valores individuales**:
- números: solo agregados (media, desvío, cuantiles, histograma) redondeados a
  `SIG_DIGITS` cifras significativas; nunca mínimos/máximos exactos;
- categorías: solo las que aparecen en ≥ `K_ANONYMITY` filas;
- texto, ids y rutas: solo longitudes y conteos.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, Field

from perceptron.data.schema import SemanticType
from perceptron.data.series import SeriesProfile
from perceptron.data.vision_tasks import VisionTaskProfile
from perceptron.domain.enums import Modality, TaskType

SIG_DIGITS = 4
K_ANONYMITY = 5
MAX_CATEGORIES = 20


def sig(x: float | None, digits: int = SIG_DIGITS) -> float | None:
    """Redondea a `digits` cifras significativas (None/NaN → None)."""
    if x is None or not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return round(x, digits - 1 - math.floor(math.log10(abs(x))))


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"


class AlertCode(StrEnum):
    CLASS_IMBALANCE = "class_imbalance"
    TARGET_LEAKAGE = "target_leakage"
    ID_COLUMN = "id_column"
    FUTURE_DATES = "future_dates"
    CONSTANT_COLUMN = "constant_column"
    HIGH_NULLS = "high_nulls"
    DUPLICATE_ROWS = "duplicate_rows"
    INSUFFICIENT_DATA = "insufficient_data"
    CORRUPT_FILES = "corrupt_files"
    NEAR_DUPLICATES = "near_duplicates"
    MIXED_RESOLUTIONS = "mixed_resolutions"
    MIXED_CHANNELS = "mixed_channels"
    MISSING_TARGET = "missing_target"
    DUPLICATE_TEXTS = "duplicate_texts"
    LONG_TEXTS = "long_texts"
    CLIPPING = "clipping"
    MIXED_SAMPLE_RATES = "mixed_sample_rates"
    SILENT_AUDIO = "silent_audio"
    ID_LIKE_FEATURE = "id_like_feature"
    NO_FEATURES = "no_features"


class Alert(BaseModel):
    code: AlertCode
    severity: AlertSeverity
    column: str | None = None
    message: str
    evidence: dict[str, float | int | str | None] = Field(default_factory=dict)


class NumericStats(BaseModel):
    mean: float | None
    std: float | None
    quantiles: dict[str, float | None] = Field(description="p05, p25, p50, p75, p95")
    histogram: list[int] = Field(description="Conteos en bins equiespaciados entre p01 y p99")
    outlier_fraction: float = Field(description="Fuera de [Q1 − 1,5·IQR, Q3 + 1,5·IQR]")
    skew: float | None = None


class CategoryCount(BaseModel):
    value: str
    count: int


class CategoricalStats(BaseModel):
    top: list[CategoryCount] = Field(description=f"Solo categorías con ≥ {K_ANONYMITY} filas")
    other_count: int = Field(description="Filas en categorías omitidas (raras o fuera del top)")


class TextStats(BaseModel):
    mean_length: float | None
    p95_length: float | None
    mean_words: float | None


class ColumnProfile(BaseModel):
    name: str
    semantic: SemanticType
    dtype: str
    null_fraction: float
    n_unique: int
    numeric: NumericStats | None = None
    categorical: CategoricalStats | None = None
    text: TextStats | None = None
    target_association: float | None = Field(
        default=None, description="|Pearson| (numérica) o V de Cramér (categórica) con el target"
    )


class TargetProfile(BaseModel):
    name: str
    semantic: SemanticType
    task_hint: TaskType
    classes: list[CategoryCount] | None = None
    imbalance_ratio: float | None = Field(default=None, description="minoritaria / mayoritaria")
    numeric: NumericStats | None = None


class ImageProfile(BaseModel):
    count: int
    corrupt: int
    resolutions: list[CategoryCount] = Field(description="Resoluciones WxH frecuentes (≥ k filas)")
    width_quantiles: dict[str, float | None]
    height_quantiles: dict[str, float | None]
    channels: list[CategoryCount]
    formats: list[CategoryCount]
    near_duplicate_pairs: int
    near_duplicate_fraction: float


class AudioProfile(BaseModel):
    """Audio (RF-PRF-05)."""

    count: int
    corrupt: int
    duration_quantiles: dict[str, float | None]
    sample_rates: list[CategoryCount]
    channels: list[CategoryCount]
    silence_fraction: float | None
    clipping_fraction: float | None
    snr_db: float | None


class TextProfile(BaseModel):
    """Texto (RF-PRF-03): solo agregados; el vocabulario no se expone."""

    column: str
    language: str | None
    language_confidence: float
    tokens_p50: float | None
    tokens_p95: float | None
    vocabulary_size: int
    empty_fraction: float
    duplicate_fraction: float


class ProfileCard(BaseModel):
    """Resumen agregado del dataset (solo train + val; el test sellado no se perfila)."""

    card_version: str = "1.0"
    dataset_version_id: str | None = None
    content_hash: str | None = None
    modality: Modality
    num_samples: int
    profiled_samples: int
    split_counts: dict[str, int]
    num_features: int
    target: TargetProfile | None
    columns: list[ColumnProfile]
    images: ImageProfile | None = None
    text: TextProfile | None = None
    audio: AudioProfile | None = None
    vision_task: VisionTaskProfile | None = None
    series: SeriesProfile | None = None
    duplicate_row_fraction: float | None = None
    alerts: list[Alert] = Field(default_factory=list)

    def alerts_by(self, code: AlertCode) -> list[Alert]:
        return [a for a in self.alerts if a.code is code]
