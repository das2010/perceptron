"""Auditoría de privacidad: ¿algún valor individual del dataset salió hacia el LLM?

Se usa en los tests de propiedad (§15.3) y en la aceptación de la Capa 2 (O4): se
recorren los `LLMCall` de un proyecto y se buscan, textualmente, los valores que
identifican filas (strings raros, números con muchos dígitos, rutas de archivos).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import polars as pl

from perceptron.data.profiling.card import K_ANONYMITY
from perceptron.data.splits import SPLIT_COLUMN
from perceptron.data.view import DatasetView, Purpose

MIN_STR_LEN = 4
MIN_INT_ABS = 10_000
MIN_FLOAT_DIGITS = 6


@dataclass(frozen=True)
class Leak:
    column: str
    value: str
    call_id: str | None


def _float_needles(v: float) -> list[str]:
    text = repr(v)
    digits = sum(ch.isdigit() for ch in text.split("e")[0].lstrip("-0."))
    return [text] if digits >= MIN_FLOAT_DIGITS else []


def individual_values(view: DatasetView, *, k: int = K_ANONYMITY) -> dict[str, set[str]]:
    """Por columna, las representaciones textuales que identificarían una fila."""
    df = view.read(None, purpose=Purpose.PRIVACY_AUDIT)
    out: dict[str, set[str]] = {}
    for col in df.columns:
        if col == SPLIT_COLUMN:
            continue
        series = df.get_column(col).drop_nulls()
        if series.len() == 0:
            continue
        counts = series.value_counts()
        rare = counts.filter(pl.col("count") < k).get_column(col).to_list()
        needles: set[str] = set()
        for value in rare:
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                if len(value.strip()) >= MIN_STR_LEN:
                    needles.add(value)
            elif isinstance(value, int):
                if abs(value) >= MIN_INT_ABS:
                    needles.add(str(value))
            elif isinstance(value, float):
                needles.update(_float_needles(value))
        if needles:
            out[col] = needles
    return out


def find_leaks(
    payloads: Iterable[tuple[str | None, Any]], values: dict[str, set[str]]
) -> list[Leak]:
    """`payloads`: pares (llm_call_id, payload enviado)."""
    leaks: list[Leak] = []
    for call_id, payload in payloads:
        hay = json.dumps(payload, ensure_ascii=False)
        for col, needles in values.items():
            leaks += [Leak(col, n, call_id) for n in needles if n in hay]
    return leaks
