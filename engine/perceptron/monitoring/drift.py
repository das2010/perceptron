"""Drift de datos y de embeddings (RF-MON-02), sin dependencias pesadas (ADR-0033).

Tabular, por feature, contra la referencia (split de entrenamiento del champion):
- numéricas: PSI con bins por cuantiles de la referencia, KS (estadístico y p-valor) y
  distancia de Jensen-Shannon;
- categóricas: χ² (p-valor), Jensen-Shannon y la fracción de categorías nunca vistas.

No estructurado (o el espacio de salida del modelo): MMD con kernel RBF (ancho por la
mediana), distancia de centroides y un clasificador de dominio (AUC: 0,5 = sin drift).

Severidad: none < low < medium < high, por umbrales convencionales (PSI 0,1/0,2/0,3).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from perceptron.domain.enums import Severity

PSI_BINS = 10
_EPS = 1e-6


class FeatureDrift(BaseModel):
    feature: str
    kind: str  # numeric | categorical
    severity: Severity
    psi: float | None = None
    ks: float | None = None
    ks_pvalue: float | None = None
    chi2_pvalue: float | None = None
    js: float | None = None
    unseen_fraction: float | None = None
    reference_mean: float | None = None
    current_mean: float | None = None
    top_changes: dict[str, list[float]] = Field(
        default_factory=dict, description="Categoría → [share referencia, share actual]"
    )


class DataDrift(BaseModel):
    severity: Severity
    share_drifted: float
    n_reference: int
    n_current: int
    features: list[FeatureDrift]

    @property
    def drifted(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.severity.rank >= Severity.MEDIUM.rank]


class EmbeddingDrift(BaseModel):
    severity: Severity
    mmd: float
    mmd_pvalue: float | None
    centroid_distance: float
    domain_auc: float
    n_reference: int
    n_current: int


# ------------------------------------------------------------------ métricas base


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Distancia de Jensen-Shannon (base 2, en [0, 1])."""
    p = np.asarray(p, dtype=float) + _EPS
    q = np.asarray(q, dtype=float) + _EPS
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * float(np.sum(p * np.log2(p / m))) + 0.5 * float(np.sum(q * np.log2(q / m)))
    return math.sqrt(max(js, 0.0))


def psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    r = np.clip(np.asarray(ref_share, dtype=float), _EPS, None)
    c = np.clip(np.asarray(cur_share, dtype=float), _EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _bins(reference: np.ndarray) -> np.ndarray:
    qs = np.unique(np.quantile(reference, np.linspace(0, 1, PSI_BINS + 1)))
    if len(qs) < 3:  # casi constante: dos bins alrededor del valor
        v = float(reference[0]) if len(reference) else 0.0
        return np.array([-np.inf, v, np.inf])
    qs[0], qs[-1] = -np.inf, np.inf
    return qs


def _shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts / max(counts.sum(), 1)


def severity_from_psi(value: float) -> Severity:
    if value >= 0.3:
        return Severity.HIGH
    if value >= 0.2:
        return Severity.MEDIUM
    if value >= 0.1:
        return Severity.LOW
    return Severity.NONE


def _severity_from_js(value: float) -> Severity:
    if value >= 0.3:
        return Severity.HIGH
    if value >= 0.2:
        return Severity.MEDIUM
    if value >= 0.1:
        return Severity.LOW
    return Severity.NONE


# ------------------------------------------------------------------ tabular


def numeric_drift(name: str, ref: np.ndarray, cur: np.ndarray) -> FeatureDrift:
    from scipy import stats

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return FeatureDrift(feature=name, kind="numeric", severity=Severity.NONE)
    edges = _bins(ref)
    r, c = _shares(ref, edges), _shares(cur, edges)
    value = psi(r, c)
    ks = stats.ks_2samp(ref, cur)
    sev = severity_from_psi(value)
    # KS muy significativo con un efecto no trivial sube la severidad (muestras chicas).
    if sev.rank < Severity.MEDIUM.rank and ks.pvalue < 1e-3 and ks.statistic > 0.2:
        sev = Severity.MEDIUM
    return FeatureDrift(
        feature=name,
        kind="numeric",
        severity=sev,
        psi=round(value, 6),
        ks=round(float(ks.statistic), 6),
        ks_pvalue=float(ks.pvalue),
        js=round(js_distance(r, c), 6),
        reference_mean=float(ref.mean()),
        current_mean=float(cur.mean()),
    )


def categorical_drift(name: str, ref: list[Any], cur: list[Any]) -> FeatureDrift:
    from scipy import stats

    ref_s = [str(v) for v in ref if v is not None]
    cur_s = [str(v) for v in cur if v is not None]
    if not ref_s or not cur_s:
        return FeatureDrift(feature=name, kind="categorical", severity=Severity.NONE)
    cats = sorted(set(ref_s) | set(cur_s))
    rc = np.array([ref_s.count(k) for k in cats], dtype=float)
    cc = np.array([cur_s.count(k) for k in cats], dtype=float)
    r, c = rc / rc.sum(), cc / cc.sum()
    unseen = float(sum(cc[i] for i, k in enumerate(cats) if rc[i] == 0) / cc.sum())
    table = np.vstack([rc, cc])
    table = table[:, table.sum(axis=0) > 0]
    pvalue = float(stats.chi2_contingency(table)[1]) if table.shape[1] > 1 else 1.0
    js = js_distance(r, c)
    sev = max(_severity_from_js(js), severity_from_psi(psi(r, c)), key=lambda s: s.rank)
    if unseen >= 0.05:  # categorías nuevas relevantes
        sev = max(sev, Severity.MEDIUM, key=lambda s: s.rank)
    changes = sorted(
        ((k, float(r[i]), float(c[i])) for i, k in enumerate(cats)),
        key=lambda t: abs(t[2] - t[1]),
        reverse=True,
    )[:5]
    return FeatureDrift(
        feature=name,
        kind="categorical",
        severity=sev,
        psi=round(psi(r, c), 6),
        chi2_pvalue=pvalue,
        js=round(js, 6),
        unseen_fraction=round(unseen, 6),
        top_changes={k: [round(a, 4), round(b, 4)] for k, a, b in changes},
    )


def data_drift(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> DataDrift:
    features: list[FeatureDrift] = []
    for col in numeric:
        if col in reference.columns and col in current.columns:
            ref = reference[col].cast(pl.Float64, strict=False).to_numpy()
            cur = current[col].cast(pl.Float64, strict=False).to_numpy()
            features.append(numeric_drift(col, ref.astype(float), cur.astype(float)))
    for col in categorical:
        if col in reference.columns and col in current.columns:
            features.append(
                categorical_drift(col, reference[col].to_list(), current[col].to_list())
            )
    drifted = [f for f in features if f.severity.rank >= Severity.MEDIUM.rank]
    share = len(drifted) / len(features) if features else 0.0
    worst = max((f.severity for f in features), key=lambda s: s.rank, default=Severity.NONE)
    # Una sola feature con drift leve no es una alerta alta: se pondera por la proporción.
    if worst is Severity.HIGH and share < 0.25 and len(features) > 3:
        worst = Severity.MEDIUM
    return DataDrift(
        severity=worst,
        share_drifted=round(share, 4),
        n_reference=reference.height,
        n_current=current.height,
        features=features,
    )


# ------------------------------------------------------------------ embeddings


def _rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None, :] - 2 * a @ b.T
        out: np.ndarray = np.exp(-gamma * np.clip(d, 0, None))
        return out

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def embedding_drift(
    reference: np.ndarray, current: np.ndarray, *, permutations: int = 50, seed: int = 0
) -> EmbeddingDrift:
    """Drift entre dos nubes de embeddings (filas = muestras)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    rng = np.random.default_rng(seed)
    x = np.asarray(reference, dtype=float)
    y = np.asarray(current, dtype=float)
    x = x[rng.permutation(len(x))[:500]]
    y = y[rng.permutation(len(y))[:500]]
    both = np.vstack([x, y])
    mu, sd = both.mean(0), both.std(0) + _EPS
    x, y, both = (x - mu) / sd, (y - mu) / sd, (both - mu) / sd
    sample = both[rng.permutation(len(both))[:300]]
    d = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    gamma = 1.0 / max(med, _EPS)
    mmd = _rbf_mmd(x, y, gamma)
    exceed = 0
    for _ in range(permutations):  # test de permutación
        perm = rng.permutation(len(both))
        exceed += _rbf_mmd(both[perm[: len(x)]], both[perm[len(x) :]], gamma) >= mmd
    pvalue = (exceed + 1) / (permutations + 1)
    centroid = float(np.linalg.norm(x.mean(0) - y.mean(0)))
    labels = np.r_[np.zeros(len(x)), np.ones(len(y))]
    folds = min(5, int(min(len(x), len(y))))
    auc = 0.5
    if folds >= 2:
        proba = cross_val_predict(
            LogisticRegression(max_iter=500), both, labels, cv=folds, method="predict_proba"
        )[:, 1]
        auc = float(roc_auc_score(labels, proba))
    if auc >= 0.8 or (pvalue < 0.01 and centroid > 1.0):
        sev = Severity.HIGH
    elif auc >= 0.7 or pvalue < 0.01:
        sev = Severity.MEDIUM
    elif auc >= 0.6 or pvalue < 0.05:
        sev = Severity.LOW
    else:
        sev = Severity.NONE
    return EmbeddingDrift(
        severity=sev,
        mmd=round(mmd, 6),
        mmd_pvalue=round(pvalue, 4),
        centroid_distance=round(centroid, 6),
        domain_auc=round(auc, 4),
        n_reference=len(x),
        n_current=len(y),
    )
