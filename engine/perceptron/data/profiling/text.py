"""Profiling de texto (RF-PRF-03)."""

from __future__ import annotations

import numpy as np
import polars as pl

from perceptron.data.profiling.card import Alert, AlertCode, AlertSeverity, TextProfile, sig
from perceptron.data.text import guess_language, tokenize

LONG_TEXT_TOKENS = 512
DUPLICATE_WARNING = 0.05


def profile_text(df: pl.DataFrame, column: str) -> TextProfile:
    texts = df[column].cast(pl.String).fill_null("").to_list()
    lengths = np.array([len(tokenize(t)) for t in texts]) if texts else np.array([0])
    vocab: set[str] = set()
    for t in texts:
        vocab.update(tokenize(t.lower()))
    non_empty = [t for t in texts if t.strip()]
    lang, conf = guess_language(non_empty)
    unique = len(set(non_empty))
    return TextProfile(
        column=column,
        language=lang,
        language_confidence=conf,
        tokens_p50=sig(float(np.quantile(lengths, 0.5))),
        tokens_p95=sig(float(np.quantile(lengths, 0.95))),
        vocabulary_size=len(vocab),
        empty_fraction=round(1 - len(non_empty) / max(len(texts), 1), 4),
        duplicate_fraction=round(1 - unique / max(len(non_empty), 1), 4),
    )


def text_alerts(tp: TextProfile) -> list[Alert]:
    alerts: list[Alert] = []
    if tp.duplicate_fraction > DUPLICATE_WARNING:
        alerts.append(
            Alert(
                code=AlertCode.DUPLICATE_TEXTS,
                severity=AlertSeverity.WARNING,
                column=tp.column,
                message="Hay textos repetidos: si caen en train y test inflan las métricas.",
                evidence={"duplicate_fraction": tp.duplicate_fraction},
            )
        )
    if (tp.tokens_p95 or 0) > LONG_TEXT_TOKENS:
        alerts.append(
            Alert(
                code=AlertCode.LONG_TEXTS,
                severity=AlertSeverity.INFO,
                column=tp.column,
                message=f"El 5 % de los textos supera {LONG_TEXT_TOKENS} tokens: se truncan.",
                evidence={"tokens_p95": tp.tokens_p95},
            )
        )
    return alerts
