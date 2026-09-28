"""Lectura de fuentes streaming/API hacia su buffer (RF-ING-05)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from perceptron.core.errors import ValidationError
from perceptron.data.sources.stream import StreamBuffer, build_source
from perceptron.domain.enums import DataSourceType
from perceptron.domain.models import DataSource, utcnow

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext


def buffer_for(ctx: EngineContext, src: DataSource) -> StreamBuffer:
    root: Path = ctx.settings.paths.project(src.project_id).root / "streams" / src.id
    return StreamBuffer(root)


def pull_source(ctx: EngineContext, source_id: str) -> dict[str, Any]:
    """Trae un lote, lo agrega al buffer y guarda el cursor. Devuelve el estado del buffer."""
    repo = ctx.repo(DataSource)
    src = repo.get(source_id)
    if src.type not in (DataSourceType.STREAM, DataSourceType.API) or "stream" not in src.config:
        raise ValidationError("la fuente no es streaming ni API")
    spec = src.config["stream"]
    source = build_source(str(spec["kind"]), dict(spec["config"]), net=ctx.settings.net_policy())
    buffer = buffer_for(ctx, src)
    secret_ref = src.secret_refs.get("token")
    secret = ctx.llm.secrets.get(secret_ref) if secret_ref else None
    batch = source.fetch(buffer.state(), secret)
    added = buffer.append(batch.rows)
    buffer.save_state(batch.state)
    repo.update(
        repo.get(src.id).model_copy(
            update={"config": {**src.config, "last_poll": utcnow().isoformat()}}
        )
    )
    ctx.events.publish("stream.batch", source_id=src.id, project_id=src.project_id, rows=added)
    return {"added": added, **buffer.stats()}
