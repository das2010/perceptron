"""Alertas de calidad (RF-PRF-06): desbalance, leakage, columnas constantes,
nulos, duplicados, fechas futuras, datos insuficientes y problemas de imágenes."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from perceptron.data.profiling.card import (
    Alert,
    AlertCode,
    AlertSeverity,
    ColumnProfile,
    ImageProfile,
    TargetProfile,
)
from perceptron.data.schema import SemanticType, TableSchema
from perceptron.domain.enums import TaskType

IMBALANCE_WARNING = 0.2
IMBALANCE_HIGH = 0.05
LEAKAGE_ASSOCIATION = 0.98
HIGH_NULLS = 0.5
QUASI_CONSTANT = 0.99
MIN_SAMPLES = 50
MIN_PER_CLASS = 10
NEAR_DUP_WARNING = 0.05


def _a(
    code: AlertCode,
    sev: AlertSeverity,
    msg: str,
    column: str | None = None,
    **ev: float | int | str | None,
) -> Alert:
    return Alert(code=code, severity=sev, column=column, message=msg, evidence=ev)


def target_alerts(target: TargetProfile | None, n: int) -> list[Alert]:
    if target is None:
        return [
            _a(
                AlertCode.MISSING_TARGET,
                AlertSeverity.WARNING,
                "No se identificó una columna objetivo; indicá cuál querés predecir.",
            )
        ]
    alerts: list[Alert] = []
    if target.task_hint is TaskType.CLASSIFICATION and target.classes:
        ratio = target.imbalance_ratio or 0
        if ratio < IMBALANCE_WARNING:
            sev = AlertSeverity.HIGH if ratio < IMBALANCE_HIGH else AlertSeverity.WARNING
            alerts.append(
                _a(
                    AlertCode.CLASS_IMBALANCE,
                    sev,
                    "Clases desbalanceadas: conviene usar pesos de clase y métricas F1 o PR-AUC.",
                    target.name,
                    ratio=ratio,
                )
            )
        minority = min(c.count for c in target.classes)
        if minority < MIN_PER_CLASS:
            alerts.append(
                _a(
                    AlertCode.INSUFFICIENT_DATA,
                    AlertSeverity.HIGH,
                    f"Hay clases con menos de {MIN_PER_CLASS} ejemplos.",
                    target.name,
                    min_class_count=minority,
                )
            )
    if n < MIN_SAMPLES:
        alerts.append(
            _a(
                AlertCode.INSUFFICIENT_DATA,
                AlertSeverity.HIGH,
                f"Solo hay {n} ejemplos: los resultados serán poco confiables.",
                samples=n,
            )
        )
    return alerts


def column_alerts(schema: TableSchema, cols: list[ColumnProfile], df: pl.DataFrame) -> list[Alert]:
    alerts: list[Alert] = []
    now = datetime.now(UTC).replace(tzinfo=None)
    id_like = {c.name for c in schema.columns if c.id_like}
    for c in cols:
        if c.name in id_like:
            alerts.append(
                _a(
                    AlertCode.ID_LIKE_FEATURE,
                    AlertSeverity.WARNING,
                    "Parece un identificador (valores únicos o nombre de id), pero es la única "
                    "columna de entrada: se usa como dato. Si es un id, no hay con qué predecir.",
                    c.name,
                )
            )
        if c.semantic is SemanticType.ID:
            alerts.append(
                _a(
                    AlertCode.ID_COLUMN,
                    AlertSeverity.INFO,
                    "Columna identificadora: se excluye de las features para evitar memorización.",
                    c.name,
                )
            )
            continue
        if c.n_unique <= 1:
            alerts.append(
                _a(
                    AlertCode.CONSTANT_COLUMN,
                    AlertSeverity.WARNING,
                    "Columna constante: no aporta información.",
                    c.name,
                )
            )
        elif c.categorical and c.categorical.top:
            top_share = c.categorical.top[0].count / max(df.height, 1)
            if top_share > QUASI_CONSTANT:
                alerts.append(
                    _a(
                        AlertCode.CONSTANT_COLUMN,
                        AlertSeverity.INFO,
                        "Columna casi constante.",
                        c.name,
                        top_share=round(top_share, 4),
                    )
                )
        if c.null_fraction > HIGH_NULLS:
            alerts.append(
                _a(
                    AlertCode.HIGH_NULLS,
                    AlertSeverity.WARNING,
                    "Más de la mitad de los valores faltan.",
                    c.name,
                    null_fraction=c.null_fraction,
                )
            )
        if c.target_association is not None and c.target_association >= LEAKAGE_ASSOCIATION:
            alerts.append(
                _a(
                    AlertCode.TARGET_LEAKAGE,
                    AlertSeverity.HIGH,
                    "La columna predice el objetivo casi perfectamente: probable fuga "
                    "de información (p. ej. se calcula después del resultado). Revisala.",
                    c.name,
                    association=c.target_association,
                )
            )
    for col in schema.columns:
        if col.semantic is SemanticType.DATETIME and df[col.name].dtype.is_temporal():
            future = df.filter(pl.col(col.name).cast(pl.Datetime) > now).height
            if future:
                alerts.append(
                    _a(
                        AlertCode.FUTURE_DATES,
                        AlertSeverity.WARNING,
                        "Hay fechas posteriores a hoy: verificá que no sean datos del futuro.",
                        col.name,
                        future_fraction=round(future / df.height, 4),
                    )
                )
    if schema.target is not None and not schema.feature_columns:
        alerts.append(
            _a(
                AlertCode.NO_FEATURES,
                AlertSeverity.HIGH,
                "No queda ninguna columna de entrada (solo el objetivo e identificadores): "
                "el modelo no tendría con qué predecir. Cambiá el tipo de alguna columna.",
            )
        )
    return alerts


def duplicate_alert(fraction: float) -> list[Alert]:
    if fraction <= 0:
        return []
    sev = AlertSeverity.WARNING if fraction > 0.01 else AlertSeverity.INFO
    return [
        _a(
            AlertCode.DUPLICATE_ROWS,
            sev,
            "Hay filas duplicadas; pueden inflar las métricas si caen en train y validación.",
            duplicate_fraction=round(fraction, 4),
        )
    ]


def image_alerts(img: ImageProfile) -> list[Alert]:
    alerts: list[Alert] = []
    if img.corrupt:
        alerts.append(
            _a(
                AlertCode.CORRUPT_FILES,
                AlertSeverity.WARNING,
                "Hay imágenes que no se pueden leer; se excluyen del entrenamiento.",
                corrupt=img.corrupt,
            )
        )
    if img.near_duplicate_fraction > NEAR_DUP_WARNING:
        alerts.append(
            _a(
                AlertCode.NEAR_DUPLICATES,
                AlertSeverity.WARNING,
                "Muchas imágenes son casi idénticas: entre splits distintos inflan la validación.",
                pairs=img.near_duplicate_pairs,
                fraction=img.near_duplicate_fraction,
            )
        )
    if len(img.resolutions) > 1:
        alerts.append(
            _a(
                AlertCode.MIXED_RESOLUTIONS,
                AlertSeverity.INFO,
                "Las imágenes tienen resoluciones distintas; el pipeline las redimensiona.",
                distinct=len(img.resolutions),
            )
        )
    if len(img.channels) > 1:
        alerts.append(
            _a(
                AlertCode.MIXED_CHANNELS,
                AlertSeverity.INFO,
                "Hay imágenes con distinta cantidad de canales; se convierten a RGB.",
            )
        )
    return alerts
