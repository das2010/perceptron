"""Payload compacto para modelos locales lentos (CPU): menos tokens, misma información clave.

Se aplica después del PrivacyFilter (nunca agrega datos): quita histogramas y descripciones,
deja solo la mediana de los cuantiles, recorta curvas, observaciones y listas largas.
"""

from __future__ import annotations

from typing import Any

DROP = frozenset({"histogram", "description", "curves", "details"})
KEEP_LAST = {"evidence": 4, "runs": 4, "alerts": 6}
KEEP_FIRST = {"top": 5, "classes": 20}
HISTORY_POINTS = 8


def _downsample(items: list[Any], n: int) -> list[Any]:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [*(items[int(i * step)] for i in range(n - 1)), items[-1]]


def compact_payload(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in DROP:
                continue
            if k == "quantiles" and isinstance(v, dict):
                out[k] = {q: x for q, x in v.items() if q == "p50"}
                continue
            out[k] = compact_payload(v, k)
        return out
    if isinstance(value, list):
        items = [compact_payload(v) for v in value]
        if key == "history":
            return _downsample(items, HISTORY_POINTS)
        if key in KEEP_LAST:
            return items[-KEEP_LAST[key] :]
        if key in KEEP_FIRST:
            return items[: KEEP_FIRST[key]]
        return items
    return value
