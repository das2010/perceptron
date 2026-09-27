"""Extracción de JSON de respuestas en texto (modelos sin salida estructurada)."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Primer objeto JSON del texto: acepta bloques ``` y texto alrededor."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for chunk in candidates:
        start = chunk.find("{")
        while start != -1:
            depth, in_str, escape = 0, False, False
            for i in range(start, len(chunk)):
                ch = chunk[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            value = json.loads(chunk[start : i + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(value, dict):
                            return value
                        break
            start = chunk.find("{", start + 1)
    raise ValueError("la respuesta no contiene un objeto JSON")
