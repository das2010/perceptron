from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from perceptron.core.config import LoggingSettings, RuntimeMode, Settings
from perceptron.core.errors import ConflictError, NotFoundError, PerceptronError
from perceptron.core.events import EventBus
from perceptron.core.ids import IdPrefix, is_valid_id, new_id, parse_id
from perceptron.core.logging import JsonFormatter, configure_logging, log_context
from perceptron.core.paths import WorkspacePaths, default_workspace_dir

# ------------------------------------------------------------------ config / paths


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PERCEPTRON_MODE", "server")
    monkeypatch.setenv("PERCEPTRON_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PERCEPTRON_API__PORT", "8765")
    s = Settings()
    assert s.mode is RuntimeMode.SERVER
    assert s.workspace_dir == tmp_path
    assert s.api.port == 8765
    assert s.api.host == "127.0.0.1"


def test_token_is_secret() -> None:
    s = Settings(api={"token": "abc"})  # type: ignore[arg-type]
    assert "abc" not in repr(s)
    assert s.api.token is not None
    assert s.api.token.get_secret_value() == "abc"


def test_default_workspace_dir_is_absolute() -> None:
    assert default_workspace_dir().is_absolute()


def test_workspace_layout_with_awkward_path(workspace_dir: Path) -> None:
    paths = WorkspacePaths(workspace_dir).ensure()
    assert paths.projects_dir.is_dir()
    assert paths.models_cache_dir.is_dir()
    proj = paths.project("prj_x").ensure()
    for d in (proj.datasets_dir, proj.runs_dir, proj.exports_dir, proj.archspecs_dir):
        assert d.is_dir()
    assert proj.dataset("abc") == proj.datasets_dir / "abc"


# ------------------------------------------------------------------ ids


def test_new_id_roundtrip() -> None:
    value = new_id(IdPrefix.PROJECT)
    assert value.startswith("prj_")
    prefix, _ = parse_id(value)
    assert prefix is IdPrefix.PROJECT
    assert is_valid_id(value, IdPrefix.PROJECT)
    assert not is_valid_id(value, IdPrefix.RUN)


def test_ids_are_time_ordered() -> None:
    ids = [new_id(IdPrefix.RUN) for _ in range(50)]
    assert len(set(ids)) == 50
    assert [parse_id(i)[1].timestamp for i in ids] == sorted(parse_id(i)[1].timestamp for i in ids)


@given(st.text(max_size=40))
def test_is_valid_id_never_raises(value: str) -> None:
    assert isinstance(is_valid_id(value), bool)


# ------------------------------------------------------------------ errors / events


def test_error_serialization() -> None:
    err = NotFoundError("no está", details={"id": "x"})
    assert err.to_dict() == {"code": "not_found", "message": "no está", "details": {"id": "x"}}
    assert err.http_status == 404
    assert isinstance(ConflictError("c"), PerceptronError)


def test_event_bus_isolates_failing_handlers() -> None:
    bus = EventBus()
    seen: list[str] = []

    def boom(_: object) -> None:
        raise RuntimeError("x")

    bus.subscribe("a", boom)
    unsubscribe = bus.subscribe("a", lambda e: seen.append(e.topic))
    bus.subscribe("*", lambda e: seen.append("*" + e.topic))
    bus.publish("a", k=1)
    assert seen == ["a", "*a"]
    unsubscribe()
    bus.publish("a")
    assert seen == ["a", "*a", "*a"]


# ------------------------------------------------------------------ logging


def _capture(record_fn: object) -> dict[str, object]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    from perceptron.core.logging import ContextFilter

    handler.addFilter(ContextFilter())
    logger = logging.getLogger("perceptron.test")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        record_fn(logger)  # type: ignore[operator]
    finally:
        logger.removeHandler(handler)
    return json.loads(stream.getvalue().strip().splitlines()[-1])


def test_json_log_includes_context_and_redacts_secrets() -> None:
    def emit(logger: logging.Logger) -> None:
        with log_context(project_id="prj_1"), log_context(run_id="run_2"):
            logger.info("entrenando época %d", 3, extra={"api_key": "sk-123", "lr": 0.1})

    data = _capture(emit)
    assert data["msg"] == "entrenando época 3"
    assert data["project_id"] == "prj_1"
    assert data["run_id"] == "run_2"
    assert data["api_key"] == "***"
    assert data["lr"] == 0.1


def test_log_context_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="desconocidos"), log_context(foo="x"):
        pass


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    settings = LoggingSettings(level="DEBUG")
    configure_logging(settings, tmp_path / "logs")
    configure_logging(settings, tmp_path / "logs")
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_perceptron_handler", False)]
    assert len(ours) == 2  # stream + archivo, sin duplicar
    logging.getLogger("x").info("hola ñ")
    for h in ours:
        h.flush()
    assert "hola ñ" in (tmp_path / "logs" / "engine.log").read_text(encoding="utf-8")
    configure_logging(LoggingSettings(to_file=False))  # dejar el entorno limpio


def test_workspace_dir_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regresión: con `-w` relativo el worker (otro cwd) no encontraba run.json."""
    monkeypatch.chdir(tmp_path)
    s = Settings(workspace_dir=Path("rel") / "ws")
    assert s.workspace_dir.is_absolute()
    assert s.workspace_dir == Path.cwd() / "rel" / "ws"
