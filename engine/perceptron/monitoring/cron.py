"""Expresiones cron de 5 campos (minuto hora día-del-mes mes día-de-la-semana), en UTC.

Soporta `*`, listas (`1,15`), rangos (`1-5`), pasos (`*/15`, `0-30/10`) y domingo como 0 o 7.
Alcanza para los disparadores de reentrenamiento (RF-MON-05) sin sumar dependencias.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from perceptron.core.errors import ValidationError

_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
MAX_SCAN = timedelta(days=8)


def _field(spec: str, lo: int, hi: int) -> set[int]:
    values: set[int] = set()
    for raw in spec.split(","):
        step = 1
        part = raw
        if "/" in raw:
            part, step_s = raw.split("/", 1)
            step = int(step_s)
            if step < 1:
                raise ValueError("paso inválido")
        if part in ("*", ""):
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        if start < lo or end > hi or start > end:
            raise ValueError(f"{part} fuera de rango")
        values.update(range(start, end + 1, step))
    return values


class Cron:
    def __init__(self, expr: str) -> None:
        parts = expr.split()
        if len(parts) != 5:
            raise ValidationError(f"cron de 5 campos esperado: {expr!r}")
        try:
            fields = [_field(p, lo, hi) for p, (lo, hi) in zip(parts, _RANGES, strict=True)]
        except ValueError as exc:
            raise ValidationError(f"cron inválido {expr!r}: {exc}") from None
        self.minutes, self.hours, self.days, self.months, dows = fields
        self.dows = {d % 7 for d in dows}
        self.any_day = parts[2] == "*"
        self.any_dow = parts[4] == "*"

    def matches(self, dt: datetime) -> bool:
        if (
            dt.minute not in self.minutes
            or dt.hour not in self.hours
            or dt.month not in self.months
        ):
            return False
        dom = dt.day in self.days
        dow = (dt.isoweekday() % 7) in self.dows
        # Como cron clásico: si se restringen ambos, alcanza con uno.
        if self.any_day:
            return dow
        if self.any_dow:
            return dom
        return dom or dow

    def due(self, last: datetime | None, now: datetime) -> bool:
        """¿Hubo un disparo en (last, now]? Sin `last`, solo si `now` coincide."""
        now = now.replace(second=0, microsecond=0)
        if last is None:
            return self.matches(now)
        t = last.replace(second=0, microsecond=0) + timedelta(minutes=1)
        if now - t > MAX_SCAN:
            t = now - MAX_SCAN
        while t <= now:
            if self.matches(t):
                return True
            t += timedelta(minutes=1)
        return False
