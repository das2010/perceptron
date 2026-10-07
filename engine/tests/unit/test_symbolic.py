"""Fórmula sugerida (ADR-0039): intérprete seguro, traducciones y búsqueda."""

from __future__ import annotations

import numpy as np
import pytest

from perceptron.core.errors import ValidationError
from perceptron.evaluation import symbolic as sym


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('echo')",
        "x1.real",
        "open('archivo')",
        "(lambda: 1)()",
        "x1 if x1 else 2",
        "[x1]",
        "sqrt(x1, x2)",
        "y",
    ],
)
def test_only_arithmetic_is_allowed(text: str) -> None:
    """Una fórmula importada en un paquete no puede ejecutar código."""
    with pytest.raises(ValidationError):
        sym.evaluate_expression(text, np.ones((1, 2)))


def test_evaluation_and_translations() -> None:
    x = np.array([[0.25, 0.81], [4.0, 0.0]])
    assert sym.evaluate_expression("x1*(sqrt(x2) + 1)", x).tolist() == pytest.approx([0.475, 4.0])
    assert sym.evaluate_expression("3", x).tolist() == [3.0, 3.0]
    assert sym.to_excel("x1*(sqrt(x2) + 1.5)", spanish=True) == "=(A2*(RAIZ(B2)+1,5))"
    assert sym.to_excel("x1*(sqrt(x2) + 1.5)", spanish=False) == "=(A2*(SQRT(B2)+1.5))"
    code = sym.to_python("x1*(sqrt(x2) + 1)", ["Sensor 1", "2do sensor"])
    assert "def formula(Sensor_1, _2do_sensor):" in code and "math.sqrt(_2do_sensor)" in code
    scope: dict[str, object] = {}
    exec(code, scope)  # noqa: S102 - código generado por el propio test
    assert scope["formula"](0.25, 0.81) == pytest.approx(0.475)  # type: ignore[operator]
    # x12 no se confunde con x1 al renombrar.
    names = [f"c{i}" for i in range(1, 13)]
    assert sym.readable("x1*(sqrt(x12) + x2)", names) == "c1·(sqrt(c12) + c2)"


def test_snapping_keeps_the_rule_not_the_noise() -> None:
    """3,00001·x → 3·x; términos de 1e-7 se anulan y las constantes que se compensan se unen."""
    assert sym.simplify("3.00001*x1 - 2e-8", snap=True) == "3*x1"
    messy = "0.409483484475*x1*sqrt(x2)*(1.09851831603e-7*x2 + 2.44210100174) + 1.0*x1"
    assert sym.simplify(messy, snap=True) == "x1*(sqrt(x2) + 1)"
    assert sym.simplify("not a formula", snap=True) == "not a formula"  # queda tal cual


def test_collinear_inputs_are_flagged() -> None:
    s1 = np.linspace(0.01, 1, 50)
    assert sym._collinear(np.c_[s1, np.sqrt(s1)], ["Sensor1", "Sensor2"])
    rng = np.random.default_rng(0)
    assert not sym._collinear(rng.uniform(0, 1, (200, 2)), ["a", "b"])


def test_search_recovers_the_sensor_rule() -> None:
    """Caso «Sensores» con las entradas independientes: la regla exacta, que extrapola."""
    rng = np.random.default_rng(0)

    def rule(x: np.ndarray) -> np.ndarray:
        return x[:, 0] * (1 + np.sqrt(x[:, 1]))

    x, xv = rng.uniform(0, 1, (300, 2)), rng.uniform(0, 1, (100, 2))
    raw = sym._search(x, rule(x), sym.SymbolicConfig(time_limit_s=15), None)
    candidates = sym._choose(raw, xv, rule(xv))
    _, rmse, best = min(candidates, key=lambda c: c[1])
    assert rmse < 1e-6
    far = np.array([[10.0, 0.25], [5.0, 3.0]])  # muy fuera del rango de entrenamiento
    assert sym.evaluate_expression(best, far) == pytest.approx(rule(far), rel=1e-4)


def test_closed_form_recognizes_known_constants() -> None:
    """Caso «Tabla X»: salida = entrada × 3^(1/5); se encontraba 1,24573112574·x + 0,00035."""
    found = "1.24573112574*x1 + 0.000349473820465"
    assert sym.closed_form(found, scale=1785.0) == "3**(1/5)*x1"
    assert sym.readable("3**(1/5)*x1", ["entrada"]) == "3^(1/5)·entrada"
    assert sym.to_excel("3**(1/5)*x1", spanish=True) == "=((3^(1/5))*A2)"
    assert sym.closed_form("3.14159265*x1**2", scale=1.0) == "pi*x1**2"
    assert sym.readable("pi*x1**2", ["r"]) == "π·r^2"
    assert sym.to_excel("pi*x1**2", spanish=False) == "=(PI()*(A2^2))"
    assert "math.pi" in sym.to_python("pi*x1**2", ["r"])
    assert sym.evaluate_expression("pi*x1", np.array([[2.0]]))[0] == pytest.approx(2 * np.pi)
    # Una constante cualquiera queda como está (solo se reconocen formas cerradas).
    assert sym.closed_form("1.2345678*x1", scale=1.0) == "1.2345678*x1"


def test_choose_prefers_the_closed_form_when_it_predicts_as_well() -> None:
    x = np.arange(3, 5001, 7, dtype=float).reshape(-1, 1)
    y = x[:, 0] * 3 ** (1 / 5)
    [(_, rmse, best)] = sym._choose(["1.24573112574*x1 + 0.000349473820465"], x, y)
    assert best == "3**(1/5)*x1" and rmse < 1e-9
    # Con ruido, una constante parecida pero no reconocible se conserva.
    noisy = x[:, 0] * 1.31 + np.random.default_rng(0).normal(0, 1, len(x))
    [(_, _, kept)] = sym._choose(["1.31000412*x1"], x, noisy)
    assert "3**" not in kept and "pi" not in kept


def test_closed_form_tabla_x_curve() -> None:
    """Caso «Tabla X» (curva): salida = entrada + √(2/7)·entrada^(3/2)."""
    found = "0.53452250732*x1**1.5 + 0.99999833107*x1"
    closed = sym.closed_form(found, scale=60_000.0)
    x = np.arange(1, 5001, 3, dtype=float).reshape(-1, 1)
    y = x[:, 0] + np.sqrt(2 / 7) * x[:, 0] ** 1.5
    assert "sqrt(14)" in closed and "0.9999" not in closed, closed
    assert np.max(np.abs(sym.evaluate_expression(closed, x) - y)) < 1e-6
    [(_, rmse, best)] = sym._choose([found], x, y)
    assert best == closed and rmse < 1e-6
