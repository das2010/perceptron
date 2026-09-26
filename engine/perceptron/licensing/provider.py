"""Proveedores de licencia (RF-LIC-01).

Futuras implementaciones (RF-LIC-03): por asiento, servidor o GPU; suscripción
online; archivo firmado Ed25519 validable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LicenseInfo:
    licensee: str
    plan: str
    expires_at: datetime | None = None
    enabled_features: frozenset[str] | None = None  # None = todas
    limits: dict[str, int] = field(default_factory=dict)


class LicenseProvider(Protocol):
    def current(self) -> LicenseInfo: ...

    def is_feature_enabled(self, feature_key: str) -> bool: ...


class DevLicenseProvider:
    """Todo habilitado. Default en desarrollo y en v1."""

    def current(self) -> LicenseInfo:
        return LicenseInfo(licensee="development", plan="dev")

    def is_feature_enabled(self, feature_key: str) -> bool:
        return True
