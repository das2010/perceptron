"""Enlace opcional a la UI de MLflow (RF-TRK-02)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from perceptron.core.errors import NotFoundError
from perceptron.tracking.mlflow_ui import MlflowUi


def test_server_uses_the_configured_public_url(tmp_path: Path) -> None:
    ui = MlflowUi(
        "http://mlflow:5000", tmp_path, public_url="https://mlflow.empresa.com/", can_launch=False
    )
    assert ui.run_url("3", "abc") == "https://mlflow.empresa.com/#/experiments/3/runs/abc"
    with pytest.raises(NotFoundError):
        MlflowUi("http://mlflow:5000", tmp_path, can_launch=False).base_url()
    with pytest.raises(NotFoundError):  # SQLite en el servidor: no se levanta una UI en 127.0.0.1
        MlflowUi(f"sqlite:///{tmp_path / 'm.db'}", tmp_path, can_launch=False).base_url()


def test_desktop_launches_a_local_ui_once(tmp_path: Path) -> None:
    db = tmp_path / "mlflow con espacio" / "mlflow.db"
    db.parent.mkdir()
    ui = MlflowUi(f"sqlite:///{db.as_posix()}", tmp_path / "artifacts")
    try:
        url = ui.base_url()
        assert url.startswith("http://127.0.0.1:")
        assert httpx.get(url, timeout=10).status_code == 200
        assert ui.base_url() == url  # se reutiliza
    finally:
        ui.close()
