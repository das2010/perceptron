"""Fórmula sugerida: regresión simbólica como modelo de referencia (ADR-0039).

Para regresión tabular con entradas numéricas busca una fórmula cerrada sobre los valores
reales (sin el escalado del pipeline). Varias búsquedas con distinto tope de longitud
(PyOperon); con validación se elige la más corta cuyo error esté a la altura de la mejor; el
test sellado solo se usa para las métricas finales.

Las fórmulas se guardan como texto y se evalúan con un intérprete propio sobre `ast` (sin
`eval`): solo números, variables `x1…xn`, + − × ÷ ^ y √, exp, log, |x|. Así una fórmula
que llega en un paquete importado no puede ejecutar código.
"""

from __future__ import annotations

import ast
import functools
import math
import operator
import re
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from perceptron.core.errors import ValidationError
from perceptron.data.schema import SemanticType
from perceptron.data.view import DatasetView, Purpose

MAX_FEATURES = 20
MAX_ROWS = 10_000
LENGTHS = (7, 15, 25)  # topes de longitud: de la fórmula más simple a la más expresiva
OPERATORS = "add,sub,mul,div,constant,variable,sqrt,square"
COLLINEAR = 0.98  # correlación de rangos entre entradas a partir de la cual se avisa
LOW_R2 = 0.9
PARITY_POINTS = 500
TINY = 1e-5  # constantes menores se anulan al redondear (si la fórmula sigue prediciendo igual)
REL_TINY = 1e-6  # ... o menores que esto × la escala del objetivo (0,00035 frente a ±1785)
CONST_RTOL = 1e-6  # tolerancia relativa para reconocer una constante conocida (3^(1/5), π…)
LONG_DIGITS = 5  # una constante con más cifras significativas cuenta como «fea» al elegir
_VAR = re.compile(r"^x(\d+)$")
_NOT_ALLOWED = "la fórmula tiene una construcción no permitida"


class SymbolicConfig(BaseModel):
    time_limit_s: int = Field(default=60, ge=5, le=1800, description="Tope total de búsqueda")
    seed: int = 42


class Candidate(BaseModel):
    length: int
    val_rmse: float
    formula: str


class SymbolicOutcome(BaseModel):
    target: str
    features: list[str]
    expression: str = Field(description="Fórmula evaluable sobre x1…xn (orden de `features`)")
    formula: str = Field(description="Legible, con los nombres de las columnas")
    python: str
    excel_es: str
    excel_en: str
    candidates: list[Candidate]
    metrics: dict[str, dict[str, float | None]]
    parity: list[tuple[float, float]] = Field(
        default_factory=list, description="Test: (valor real, fórmula)"
    )
    warnings: list[str] = Field(default_factory=list)
    n_train: int
    duration_s: float


# ------------------------------------------------------------------ expresiones seguras

_BIN: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_FUNCS: dict[str, Callable[[Any], Any]] = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "Abs": np.abs,
    "abs": np.abs,
}

# Constantes con nombre que puede producir la forma cerrada (sympy las escribe así).
_CONSTS: dict[str, float] = {"pi": math.pi, "E": math.e}
_CONST_EXCEL = {"pi": "PI()", "E": "EXP(1)"}
_CONST_PYTHON = {"pi": "math.pi", "E": "math.e"}


def parse_expression(text: str) -> ast.Expression:
    """Valida la fórmula: cualquier otra construcción (atributos, llamadas, nombres) se rechaza."""
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except SyntaxError as e:
        raise ValidationError("la fórmula no es válida") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression | ast.Load | ast.operator | ast.unaryop):
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            continue
        if isinstance(node, ast.Name) and (
            _VAR.match(node.id) or node.id in _FUNCS or node.id in _CONSTS
        ):
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and len(node.args) == 1
            and not node.keywords
        ):
            continue
        raise ValidationError(_NOT_ALLOWED)
    return tree


def _number(node: ast.Constant) -> float:
    value = node.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(_NOT_ALLOWED)
    return float(value)


def _func(node: ast.Call) -> str:
    if not isinstance(node.func, ast.Name):
        raise ValidationError(_NOT_ALLOWED)
    return node.func.id


def _var(node: ast.Name) -> int:
    match = _VAR.match(node.id)
    if match is None:
        raise ValidationError(_NOT_ALLOWED)
    return int(match.group(1))


def evaluate_expression(text: str, x: np.ndarray) -> np.ndarray:
    """Evalúa la fórmula sobre `x` (filas × variables, en el orden x1…xn)."""
    tree = parse_expression(text)

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return _number(node)
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.Name):
            i = _var(node) - 1
            if not 0 <= i < x.shape[1]:
                raise ValidationError("la fórmula usa una variable que no existe")
            return x[:, i]
        if isinstance(node, ast.BinOp):
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            value = ev(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call):
            return _FUNCS[_func(node)](ev(node.args[0]))
        raise ValidationError(_NOT_ALLOWED)

    with np.errstate(all="ignore"):
        out = np.asarray(ev(tree), dtype=float)
    return np.broadcast_to(out, (x.shape[0],)).astype(float)


def to_excel(text: str, *, spanish: bool) -> str:
    """Fórmula de Excel con x1 → A2, x2 → B2… (funciones en español o en inglés)."""
    names = {"sqrt": "RAIZ" if spanish else "SQRT", "exp": "EXP", "log": "LN"}
    names |= {"Abs": "ABS", "abs": "ABS"}
    symbols = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}
    tree = parse_expression(text)

    def num(v: float) -> str:
        s = str(int(v)) if v.is_integer() else repr(v)
        return s.replace(".", ",") if spanish else s

    def em(node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return em(node.body)
        if isinstance(node, ast.Constant):
            return num(_number(node))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONST_EXCEL[node.id]
        if isinstance(node, ast.Name):
            return f"{_column_letter(_var(node))}2"
        if isinstance(node, ast.BinOp):
            return f"({em(node.left)}{symbols[type(node.op)]}{em(node.right)})"
        if isinstance(node, ast.UnaryOp):
            return f"(-{em(node.operand)})" if isinstance(node.op, ast.USub) else em(node.operand)
        if isinstance(node, ast.Call):
            return f"{names[_func(node)]}({em(node.args[0])})"
        raise ValidationError(_NOT_ALLOWED)

    return "=" + em(tree)


def _column_letter(n: int) -> str:
    letters = ""
    while n:
        n, r = divmod(n - 1, 26)
        letters = chr(65 + r) + letters
    return letters


# ------------------------------------------------------------------ simplificación (sympy)


def _to_sympy(text: str) -> Any:
    import sympy as sp

    tree = parse_expression(text)
    funcs = {"sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log, "Abs": sp.Abs, "abs": sp.Abs}
    ops: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def conv(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return conv(node.body)
        if isinstance(node, ast.Constant):
            return sp.Float(_number(node))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return {"pi": sp.pi, "E": sp.E}[node.id]
        if isinstance(node, ast.Name):
            return sp.Symbol(node.id)
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](conv(node.left), conv(node.right))
        if isinstance(node, ast.UnaryOp):
            v = conv(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Call):
            return funcs[_func(node)](conv(node.args[0]))
        raise ValidationError(_NOT_ALLOWED)

    return conv(tree)


def _snap(value: float, digits: int = 4) -> float:
    """Redondea a `digits` cifras significativas y a entero si está muy cerca (0,99998 → 1)."""
    if value == 0 or not math.isfinite(value):
        return value
    if abs(value - round(value)) <= 1e-3 * max(1.0, abs(value)):
        return float(round(value))
    return float(f"{value:.{digits}g}")


def simplify(text: str, *, snap: bool) -> str:
    """Simplifica con sympy; si el resultado no es una fórmula permitida, queda la original."""
    import sympy as sp

    try:
        expr = sp.simplify(_to_sympy(text))
        # Dos pasadas: la primera anula términos diminutos (1e-7·x) y redondea; al simplificar
        # se combinan constantes que se compensaban (0,4095 × 2,4421 → 1,00002) y la segunda
        # las lleva a la regla (→ 1). Quien llama acepta el resultado solo si predice igual.
        for _ in range(2 if snap else 0):
            replace = {}
            for f in expr.atoms(sp.Float):
                v = 0.0 if abs(float(f)) < TINY else _snap(float(f))
                replace[f] = sp.Integer(int(v)) if v.is_integer() else sp.Float(v)
            expr = sp.simplify(expr.xreplace(replace))
        out = str(sp.sstr(expr.xreplace({f: sp.Float(f, 12) for f in expr.atoms(sp.Float)})))
        parse_expression(out)
    except (ValidationError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return text
    return out


@functools.cache
def _known_constants() -> tuple[tuple[float, Any], ...]:
    """Constantes con forma cerrada: raíces de enteros chicos (y sus inversas), múltiplos
    simples de π, e y logaritmos comunes. Caso «Tabla X»: 1,2457309… = 3^(1/5)."""
    import sympy as sp

    found: list[Any] = []
    for k in range(2, 13):
        for n in range(2, 6):
            if round(k ** (1 / n)) ** n == k:
                continue  # raíz exacta (4^(1/2) = 2): la resuelve el redondeo
            root = sp.Integer(k) ** sp.Rational(1, n)
            found += [root, 1 / root]
    for num in range(1, 5):
        for den in range(1, 5):
            if math.gcd(num, den) == 1:
                found += [sp.pi * sp.Rational(num, den), sp.Rational(den, num) / sp.pi]
    found += [sp.E, 1 / sp.E, sp.log(2), sp.log(10)]
    return tuple((float(c), c) for c in found)


def _recognize(value: float) -> Any | None:
    for v, c in _known_constants():
        if abs(abs(value) - v) <= CONST_RTOL * v:
            return c if value > 0 else -c
    return None


def closed_form(text: str, scale: float) -> str:
    """Anula constantes despreciables frente a la escala del objetivo y reemplaza las que tienen
    forma cerrada conocida (1,24573093… → 3^(1/5)). Quien llama la acepta solo si predice igual
    de bien que la original; si no es una fórmula permitida, queda la original."""
    import sympy as sp

    try:
        expr = sp.simplify(_to_sympy(text))
        replace: dict[Any, Any] = {}
        for f in expr.atoms(sp.Float):
            v = float(f)
            if abs(v) < max(TINY, REL_TINY * scale):
                replace[f] = sp.Integer(0)
            elif abs(v - round(v)) <= 1e-9 * max(1.0, abs(v)):
                replace[f] = sp.Integer(round(v))
            elif (c := _recognize(v)) is not None:
                replace[f] = c
        expr = sp.simplify(expr.xreplace(replace))
        out = str(sp.sstr(expr.xreplace({f: sp.Float(f, 12) for f in expr.atoms(sp.Float)})))
        parse_expression(out)
    except (ValidationError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return text
    return out


def _ugliness(text: str) -> tuple[int, int]:
    """(constantes con muchas cifras, nodos): para preferir la forma más legible."""
    long_floats = sum(
        1
        for node in ast.walk(parse_expression(text))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and len(f"{node.value:.12g}".replace("-", "").replace(".", "").lstrip("0")) > LONG_DIGITS
    )
    return long_floats, sum(1 for _ in ast.walk(parse_expression(text)))


def _rename(text: str, names: list[str]) -> str:
    """x1…xn → nombres, en una sola pasada (un nombre nunca pisa a otro)."""
    return re.sub(r"\bx(\d+)\b", lambda m: names[int(m.group(1)) - 1], text)


def readable(text: str, names: list[str]) -> str:
    text = re.sub(r"\b(pi|E)\b", lambda m: {"pi": "π", "E": "e"}[m.group(1)], text)
    return _rename(text, names).replace("**", "^").replace("*", "·")


def to_python(text: str, names: list[str]) -> str:
    args = [_identifier(n, i) for i, n in enumerate(names)]
    # Las constantes antes que las variables: un nombre de columna podría llamarse «pi».
    text = re.sub(r"\b(pi|E)\b", lambda m: _CONST_PYTHON[m.group(1)], text)
    body = _rename(text, args)
    body = re.sub(r"\b(sqrt|exp|log)\(", r"math.\1(", body).replace("Abs(", "abs(")
    return f"import math\n\n\ndef formula({', '.join(args)}):\n    return {body}\n"


def _identifier(name: str, i: int) -> str:
    ident = re.sub(r"\W", "_", name.strip()) or f"x{i + 1}"
    return f"_{ident}" if ident[0].isdigit() else ident


# ------------------------------------------------------------------ búsqueda


def _frame(view: DatasetView, split: str, cols: list[str]) -> pl.DataFrame:
    purpose = Purpose.FINAL_EVALUATION if split == "test" else Purpose.TRAINING
    return view.read(split, purpose=purpose).select(cols).drop_nulls()


def _arrays(df: pl.DataFrame, features: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    """float64 en orden C y escribibles: PyOperon (nanobind) no acepta las vistas de Polars."""
    x = df.select(pl.col(c).cast(pl.Float64) for c in features).to_numpy()
    y = df[target].cast(pl.Float64).to_numpy()
    return np.array(x, dtype=np.float64, order="C"), np.array(y, dtype=np.float64, order="C")


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    err = pred - y
    return float(np.sqrt(np.mean(err**2))) if np.all(np.isfinite(err)) else math.inf


def _collinear(x: np.ndarray, names: list[str]) -> list[str]:
    if x.shape[1] < 2 or x.shape[0] < 3:
        return []
    ranks = np.argsort(np.argsort(x, axis=0), axis=0).astype(float)
    with np.errstate(all="ignore"):
        corr = np.corrcoef(ranks, rowvar=False)
    pairs = [
        f"«{names[i]}» y «{names[j]}»"
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if abs(corr[i, j]) >= COLLINEAR
    ]
    if not pairs:
        return []
    return [
        "Hay entradas que varían juntas ("
        + ", ".join(pairs)
        + "): muchas fórmulas distintas explican igual estos datos y la sugerida puede no ser "
        "la regla real. Para inferirla hacen falta datos donde varíen por separado."
    ]


def check_symbolic(view: DatasetView) -> tuple[str, list[str]]:
    schema = view.schema
    target = schema.target
    if target is None or schema.column(target).semantic is not SemanticType.NUMERIC:
        raise ValidationError(
            "La fórmula sugerida es para regresión: el objetivo tiene que ser numérico.",
            details={"reason": "not_regression"},
        )
    features = [c.name for c in schema.feature_columns if c.semantic is SemanticType.NUMERIC]
    if not features:
        raise ValidationError(
            "No hay columnas numéricas de entrada para buscar una fórmula.",
            details={"reason": "no_numeric_features"},
        )
    if len(features) > MAX_FEATURES:
        raise ValidationError(
            f"La fórmula sugerida admite hasta {MAX_FEATURES} columnas numéricas de entrada.",
            details={"reason": "too_many_features", "features": len(features)},
        )
    return target, features


def _search(x: np.ndarray, y: np.ndarray, cfg: SymbolicConfig, progress: Any) -> list[str]:
    from pyoperon.sklearn import SymbolicRegressor

    names = [f"x{i + 1}" for i in range(x.shape[1])]
    per_run = max(1, cfg.time_limit_s // len(LENGTHS))
    found: list[str] = []
    for i, length in enumerate(LENGTHS):
        model = SymbolicRegressor(
            allowed_symbols=OPERATORS,
            max_length=length,
            generations=1_000_000,  # corta el tiempo, no las generaciones
            population_size=500,
            objectives=["r2"],
            optimizer_iterations=10,
            random_state=cfg.seed,
            max_time=per_run,
            n_threads=1,  # determinismo: misma semilla, misma fórmula
        )
        model.fit(x, y)
        found.append(model.get_model_string(model.model_, 12, names))
        if progress:
            progress(i + 1, len(LENGTHS))
    return found


def _choose(raw: list[str], x_va: np.ndarray, y_va: np.ndarray) -> list[tuple[int, float, str]]:
    """Cada candidata simplificada. Variantes: la exacta, la redondeada (3,00001 → 3) y la de
    forma cerrada (1,2457309 → 3^(1/5), sin términos despreciables). Entre las que predicen
    igual de bien que la exacta en validación, la más legible: es la regla, no una
    aproximación. Devuelve (nodos, rmse, fórmula)."""
    scale = float(np.std(y_va)) or 1.0
    out: list[tuple[int, float, str]] = []
    for text in raw:
        exact = simplify(text, snap=False)
        e_rmse = _rmse(y_va, evaluate_expression(exact, x_va))
        variants = [(exact, e_rmse)]
        for other in (simplify(text, snap=True), closed_form(exact, scale)):
            if other != exact:
                variants.append((other, _rmse(y_va, evaluate_expression(other, x_va))))
        ok = [v for v in variants if v[1] <= e_rmse * 1.01 + 1e-9 * scale]
        best, rmse = min(ok, key=lambda v: _ugliness(v[0]))
        out.append((sum(1 for _ in ast.walk(parse_expression(best))), rmse, best))
    return out


def fit_symbolic(
    view: DatasetView,
    config: SymbolicConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SymbolicOutcome:
    from perceptron.evaluation.metrics import regression_metrics
    from perceptron.tasks.base import flat_numbers

    cfg = config or SymbolicConfig()
    target, features = check_symbolic(view)
    cols = [*features, target]
    train = _frame(view, "train", cols)
    if train.height > MAX_ROWS:
        train = train.sample(MAX_ROWS, seed=cfg.seed)
    val = _frame(view, "val", cols)
    if train.height < 5 or val.height < 2:
        raise ValidationError(
            "Hay muy pocas filas completas para buscar una fórmula.",
            details={"reason": "too_few_rows"},
        )
    x_tr, y_tr = _arrays(train, features, target)
    x_va, y_va = _arrays(val, features, target)

    started = time.monotonic()
    candidates = _choose(_search(x_tr, y_tr, cfg, progress), x_va, y_va)
    best = min(c[1] for c in candidates)
    tolerance = max(best * 1.05, best + 1e-6 * (float(np.std(y_va)) or 1.0))
    _, _, expression = min((c for c in candidates if c[1] <= tolerance), key=lambda c: c[0])

    test = _frame(view, "test", cols)
    metrics: dict[str, dict[str, float | None]] = {}
    parity: list[tuple[float, float]] = []
    for split, (x, y) in {"val": (x_va, y_va), "test": _arrays(test, features, target)}.items():
        if not len(y):
            continue
        pred = evaluate_expression(expression, x)
        ok = np.isfinite(pred)
        reg, _ = regression_metrics(y[ok], pred[ok])
        metrics[split] = dict(flat_numbers(reg.model_dump()))
        if split == "test":
            idx = np.linspace(0, len(y) - 1, min(len(y), PARITY_POINTS)).astype(int)
            parity = [(float(y[i]), float(pred[i])) for i in idx if ok[i]]

    warnings = _collinear(x_tr, features)
    if (metrics.get("val", {}).get("r2") or 0.0) < LOW_R2:
        warnings.append(
            "No se encontró una fórmula simple que explique bien los datos: la red neuronal "
            "probablemente sea mejor. Podés buscar con más tiempo."
        )
    return SymbolicOutcome(
        target=target,
        features=features,
        expression=expression,
        formula=readable(expression, features),
        python=to_python(expression, features),
        excel_es=to_excel(expression, spanish=True),
        excel_en=to_excel(expression, spanish=False),
        candidates=[
            Candidate(length=c[0], val_rmse=c[1], formula=readable(c[2], features))
            for c in sorted(candidates, key=lambda c: c[0])
        ],
        metrics=metrics,
        parity=parity,
        warnings=warnings,
        n_train=train.height,
        duration_s=round(time.monotonic() - started, 2),
    )
