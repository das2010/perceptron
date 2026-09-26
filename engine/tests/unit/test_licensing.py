from __future__ import annotations

import pytest

from perceptron.core.errors import FeatureDisabledError
from perceptron.licensing import FeatureRegistry, LicenseInfo, features


def test_dev_provider_enables_everything() -> None:
    assert features.defs
    assert all(features.is_enabled(k) for k in features.defs)


def test_unknown_feature_is_an_error() -> None:
    with pytest.raises(KeyError):
        features.is_enabled("no.existe")


def test_custom_provider_can_disable() -> None:
    class Restricted:
        def current(self) -> LicenseInfo:
            return LicenseInfo(licensee="acme", plan="basic", enabled_features=frozenset())

        def is_feature_enabled(self, feature_key: str) -> bool:
            return False

    reg = FeatureRegistry(Restricted())
    assert not reg.is_enabled("llm.agent")
    with pytest.raises(FeatureDisabledError):
        reg.require("llm.agent")
