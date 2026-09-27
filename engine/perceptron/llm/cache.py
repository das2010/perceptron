"""Caché de respuestas por hash de (proveedor, modelo, prompt, parámetros) (RF-LLM-07)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from perceptron.llm.types import LLMRequest
from perceptron.storage.db import Database, LLMCacheRow


def cache_key(provider: str, request: LLMRequest) -> str:
    body = request.model_dump(mode="json", exclude={"purpose"})
    raw = json.dumps({"provider": provider, **body}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, key: str) -> dict[str, Any] | None:
        with self.db.session() as s:
            row = s.get(LLMCacheRow, key)
            return dict(row.data) if row is not None else None

    def put(self, key: str, data: dict[str, Any]) -> None:
        with self.db.session() as s:
            row = s.get(LLMCacheRow, key)
            if row is None:
                s.add(LLMCacheRow(key=key, created_at=datetime.now(UTC), data=data))
            else:
                row.data = data
