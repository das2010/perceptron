"""Registro de features: `features.is_enabled("<feature_key>")` (RF-LIC-02)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml

from perceptron.core.errors import FeatureDisabledError
from perceptron.licensing.provider import DevLicenseProvider, LicenseProvider


@dataclass(frozen=True, slots=True)
class FeatureDef:
    key: str
    tier: str
    description: str


def load_feature_defs() -> dict[str, FeatureDef]:
    text = resources.files("perceptron.licensing").joinpath("features.yaml").read_text("utf-8")
    raw: dict[str, Any] = yaml.safe_load(text)["features"]
    return {k: FeatureDef(key=k, tier=v["tier"], description=v["description"]) for k, v in raw.items()}


class FeatureRegistry:
    def __init__(self, provider: LicenseProvider | None = None) -> None:
        self.defs = load_feature_defs()
        self.provider: LicenseProvider = provider or DevLicenseProvider()

    def is_enabled(self, feature_key: str) -> bool:
        if feature_key not in self.defs:
            raise KeyError(f"feature desconocida: {feature_key} (declarar en features.yaml)")
        return self.provider.is_feature_enabled(feature_key)

    def require(self, feature_key: str) -> None:
        if not self.is_enabled(feature_key):
            raise FeatureDisabledError(
                f"La función '{feature_key}' no está habilitada por la licencia",
                details={"feature": feature_key},
            )


features = FeatureRegistry()
