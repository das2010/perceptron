"""Profiling de audio (RF-PRF-05): duración, sample rate, canales, silencio, clipping, SNR."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from perceptron.data.audio import load, quality
from perceptron.data.profiling.card import (
    K_ANONYMITY,
    Alert,
    AlertCode,
    AlertSeverity,
    AudioProfile,
    CategoryCount,
    sig,
)

SAMPLE_FILES = 200
CLIPPING_WARNING = 0.001
SILENCE_WARNING = 0.5


def _counts(s: pl.Series) -> list[CategoryCount]:
    vc = s.drop_nulls().cast(pl.String).value_counts(sort=True, name="n")
    return [CategoryCount(value=v, count=int(n)) for v, n in vc.iter_rows() if n >= K_ANONYMITY]


def profile_audio(index: pl.DataFrame, files_dir: Path) -> AudioProfile:
    ok = index.filter(~pl.col("corrupt"))
    q: dict[str, list[float]] = {"silence_fraction": [], "clipping_fraction": [], "snr_db": []}
    for rel in ok["path"].head(SAMPLE_FILES).to_list():
        try:
            x, _ = load(files_dir / rel)
        except Exception:  # noqa: S112 - un archivo ilegible no invalida el perfil
            continue
        for k, v in quality(x).items():
            q[k].append(v)
    d = ok["duration"].drop_nulls().to_numpy()
    quant = {
        k: sig(float(np.quantile(d, v))) if len(d) else None
        for k, v in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))
    }
    return AudioProfile(
        count=index.height,
        corrupt=index.filter(pl.col("corrupt")).height,
        duration_quantiles=quant,
        sample_rates=_counts(ok["sample_rate"]),
        channels=_counts(ok["channels"]),
        silence_fraction=sig(float(np.mean(q["silence_fraction"])), 3)
        if q["silence_fraction"]
        else None,
        clipping_fraction=sig(float(np.mean(q["clipping_fraction"])), 3)
        if q["clipping_fraction"]
        else None,
        snr_db=sig(float(np.mean(q["snr_db"])), 3) if q["snr_db"] else None,
    )


def audio_alerts(ap: AudioProfile) -> list[Alert]:
    alerts: list[Alert] = []
    if ap.corrupt:
        alerts.append(
            Alert(
                code=AlertCode.CORRUPT_FILES,
                severity=AlertSeverity.WARNING,
                message="Hay audios ilegibles; se excluyen.",
                evidence={"corrupt": ap.corrupt},
            )
        )
    if (ap.clipping_fraction or 0) > CLIPPING_WARNING:
        alerts.append(
            Alert(
                code=AlertCode.CLIPPING,
                severity=AlertSeverity.WARNING,
                message="Hay saturación (clipping) en las grabaciones.",
                evidence={"clipping_fraction": ap.clipping_fraction},
            )
        )
    if len(ap.sample_rates) > 1:
        alerts.append(
            Alert(
                code=AlertCode.MIXED_SAMPLE_RATES,
                severity=AlertSeverity.INFO,
                message="Hay distintos sample rates; se remuestrea a uno común.",
            )
        )
    if (ap.silence_fraction or 0) > SILENCE_WARNING:
        alerts.append(
            Alert(
                code=AlertCode.SILENT_AUDIO,
                severity=AlertSeverity.WARNING,
                message="Más de la mitad del audio es silencio.",
                evidence={"silence_fraction": ap.silence_fraction},
            )
        )
    return alerts
