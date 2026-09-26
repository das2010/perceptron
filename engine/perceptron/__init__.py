"""Perceptron Engine."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("perceptron-engine")
except PackageNotFoundError:  # pragma: no cover - ejecución desde el árbol sin instalar
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
