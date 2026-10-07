"""Chequeos de ajuste determinísticos: forma de los datos, errores con patrón y la fórmula.

Caso «Tabla X» (salida = entrada + √(2/7)·entrada^(3/2)): el sistema recomendó y entrenó una
regresión lineal sin medir si los datos eran lineales (R² 0,982, ±6681 de error, valores
negativos con entradas chicas). El diagnóstico dijo «sin underfitting», porque las curvas de
entrenamiento no lo muestran, y se registró como candidato un modelo 8 millones de veces peor que
la fórmula sugerida del mismo proyecto. Tres chequeos baratos lo evitan:

- **linealidad:** R² de una recta frente al de un ajuste con curvatura (cúbico aditivo en cada
  entrada numérica) sobre train. Si la recta explica todo, la lineal es lo correcto; si la curva
  explica claramente más, hacen falta capas ocultas;
- **errores con patrón:** si una curva de las entradas explica buena parte del error de
  validación, el modelo no capturó parte de la relación (underfitting estructural);
- **la fórmula como referencia:** si la fórmula sugerida explica mucho más que el modelo.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

from perceptron.data.view import Purpose

if TYPE_CHECKING:
    from perceptron.data.view import DatasetView
    from perceptron.domain.models import Run, SymbolicFit
    from perceptron.llm.schemas import Problem, SuggestedAction
    from perceptron.services.workflow import Workflow

logger = logging.getLogger(__name__)

LINEAR_EXACT = 0.9999  # una recta que explica esto es la relación
CURVATURE_GAIN = 0.005  # R² que la curva tiene que sumar para contar como curvatura real
CURVED_MIN = 0.9
MIN_ROWS = 20
MAX_ROWS = 5_000
RESIDUAL_STRUCTURE = 0.3  # parte del error que explica una curva de las entradas
RESIDUAL_RELEVANT = 0.01  # error (RMSE / desvío del objetivo) a partir del cual importa
FORMULA_R2 = 0.999
FORMULA_GAP = 0.005
FORMULA_RMSE_RATIO = 10.0


class Linearity(BaseModel):
    r2_linear: float
    r2_curved: float
    n: int
    features: list[str] = Field(default_factory=list)

    @property
    def linear_explains(self) -> bool:
        return self.r2_linear >= LINEAR_EXACT

    @property
    def evident_curvature(self) -> bool:
        return self.r2_curved >= CURVED_MIN and self.r2_curved - self.r2_linear >= CURVATURE_GAIN


class RegistrationWarning(BaseModel):
    code: str
    message: str


# ---------------------------------------------------------------------- álgebra


def _standardize(x: np.ndarray) -> np.ndarray:
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    return np.asarray((x - mean) / std)


def _r2(a: np.ndarray, y: np.ndarray) -> float:
    a = np.c_[a, np.ones(len(a))]
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef
    total = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid**2).sum()) / total if total > 0 else 1.0


def curved_basis(x: np.ndarray) -> np.ndarray:
    """Cúbico aditivo: z, z², z³ por entrada (z estandarizada). Barato y capta curvaturas
    suaves como x^1,5 o √x en todo el rango."""
    z = _standardize(x)
    return np.asarray(np.c_[z, z**2, z**3])


def linearity_of(x: np.ndarray, y: np.ndarray, features: list[str]) -> Linearity | None:
    if len(y) < MIN_ROWS or float(np.std(y)) == 0.0:
        return None
    z = _standardize(x)
    return Linearity(
        r2_linear=round(_r2(z, y), 6),
        r2_curved=round(_r2(curved_basis(x), y), 6),
        n=len(y),
        features=features,
    )


def _numeric_frame(
    view: DatasetView, split: str, features: list[str], target: str
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    df = view.read(split, purpose=Purpose.TRAINING)
    cols = [c for c in features if c in df.columns and c != target]
    if not cols or target not in df.columns:
        return None
    df = df.select([*cols, target]).drop_nulls()
    if df.height > MAX_ROWS:
        df = df.sample(MAX_ROWS, seed=0)
    try:
        x = df.select(cols).to_numpy().astype(np.float64)
        y = df[target].to_numpy().astype(np.float64)
    except (TypeError, ValueError):
        return None
    return x, y, cols


def measure_linearity(view: DatasetView, target: str, features: list[str]) -> Linearity | None:
    """Linealidad de la relación en train (regresión con entradas numéricas)."""
    frame = _numeric_frame(view, "train", features, target)
    return linearity_of(*frame) if frame is not None else None


# ---------------------------------------------------------------------- diagnóstico


def latest_formula(wf: Workflow, project_id: str, dataset_version_id: str) -> SymbolicFit | None:
    from perceptron.domain.models import SymbolicFit

    fits = [
        f
        for f in wf.ctx.repo(SymbolicFit).list(filters={"project_id": project_id}, limit=200)
        if f.dataset_version_id == dataset_version_id
    ]
    return max(fits, key=lambda f: f.created_at) if fits else None


def _formula_gap(run: Run, fit: SymbolicFit | None) -> tuple[float, float] | None:
    """(R² fórmula, R² modelo) en validación si la fórmula es claramente mejor."""
    if fit is None:
        return None
    f_r2 = (fit.metrics.get("val") or {}).get("r2")
    f_rmse = (fit.metrics.get("val") or {}).get("rmse")
    m_r2, m_rmse = run.metrics.get("val_r2"), run.metrics.get("val_rmse")
    if f_r2 is None or m_r2 is None or f_r2 < FORMULA_R2:
        return None
    much_better = f_r2 - m_r2 >= FORMULA_GAP or (
        f_rmse is not None and m_rmse is not None and f_rmse * FORMULA_RMSE_RATIO < m_rmse
    )
    return (float(f_r2), float(m_r2)) if much_better else None


def residual_structure(wf: Workflow, run: Run) -> float | None:
    """Parte del error de validación que explica una curva de las entradas (0–1), o None si no
    aplica. Carga el modelo entrenado y predice el split de validación."""
    from perceptron.training.data import make_dataset
    from perceptron.training.inference import load_trained, predict

    run_dir = wf.ctx.settings.paths.project(run.project_id).run(run.id)
    trained = load_trained(run_dir)
    view = wf.view(wf.dataset(run.dataset_version_id))
    preds = predict(trained, make_dataset(view, trained.pipeline, "val", train=False))
    if preds.y_true is None:
        return None
    y = np.asarray(preds.y_true, dtype=np.float64).reshape(len(preds.y_true), -1)[:, 0]
    pred = np.asarray(preds.y_pred, dtype=np.float64).reshape(len(y), -1)[:, 0]
    df = view.read("val", purpose=Purpose.TRAINING)
    cols = [c for c in trained.pipeline.numeric_features if c in df.columns]
    if not cols or df.height != len(y):
        return None
    x = df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float64)
    resid = y - pred
    scale = float(np.std(y)) or 1.0
    if float(np.sqrt(np.mean(resid**2))) / scale < RESIDUAL_RELEVANT or float(np.std(resid)) == 0:
        return None
    return round(_r2(curved_basis(x), resid), 4)


def structural_findings(
    wf: Workflow, run: Run, task: str
) -> tuple[list[Problem], list[SuggestedAction]]:
    """Underfitting que las curvas no muestran: errores con patrón y la fórmula como referencia."""
    from perceptron.llm.schemas import Problem, SuggestedAction

    if task != "regression":
        return [], []
    problems: list[Problem] = []
    try:
        structure = residual_structure(wf, run)
    except Exception:  # sin checkpoint o datos que no se pueden leer: el chequeo no aplica
        logger.warning("sin chequeo de errores con patrón", exc_info=True)
        structure = None
    if structure is not None and structure >= RESIDUAL_STRUCTURE:
        problems.append(
            Problem(
                kind="underfitting",
                severity="high" if structure >= 0.6 else "medium",
                evidence=f"Los errores de validación siguen a las entradas: una curva de las "
                f"entradas explica el {structure:.0%} del error.",
                explanation="El modelo no captura parte de la relación (por ejemplo, una curva "
                "que una recta no puede seguir). Las curvas de entrenamiento no lo muestran.",
            )
        )
    gap = _formula_gap(run, latest_formula(wf, run.project_id, run.dataset_version_id))
    if gap is not None:
        problems.append(
            Problem(
                kind="underfitting",
                severity="high",
                evidence=f"La fórmula sugerida explica R² {gap[0]:.4f} en validación; este "
                f"modelo, {gap[1]:.4f}.",
                explanation="Hay una regla exacta que este modelo no aprendió: la fórmula es la "
                "mejor opción, o una arquitectura con más capacidad.",
            )
        )
    actions = (
        [
            SuggestedAction(
                kind="change_architecture",
                rationale="Probar una arquitectura con capas ocultas (o usar la fórmula "
                "sugerida): hay estructura que este modelo no captura.",
            )
        ]
        if problems
        else []
    )
    return problems, actions


def registration_warnings(wf: Workflow, run: Run) -> list[RegistrationWarning]:
    """Avisos antes de registrar un modelo como candidato."""
    out: list[RegistrationWarning] = []
    gap = _formula_gap(run, latest_formula(wf, run.project_id, run.dataset_version_id))
    if gap is not None:
        out.append(
            RegistrationWarning(
                code="formula_better",
                message=f"La fórmula sugerida de este proyecto explica mucho más (R² {gap[0]:.4f} "
                f"en validación) que este modelo ({gap[1]:.4f}). Conviene usar la fórmula o "
                "entrenar una arquitectura con más capacidad.",
            )
        )
    diagnosis: dict[str, Any] = run.diagnosis or {}
    structural = [
        p
        for p in diagnosis.get("problems") or []
        if p.get("kind") == "underfitting" and p.get("severity") == "high"
    ]
    if structural and not out:
        out.append(
            RegistrationWarning(
                code="underfitting",
                message="El diagnóstico detectó que el modelo no captura parte de la relación: "
                + str(structural[0].get("evidence", "")),
            )
        )
    return out
