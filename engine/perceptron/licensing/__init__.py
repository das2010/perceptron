"""Hooks de licenciamiento (SPEC §7.17). Sin enforcement en v1 (RF-LIC-03)."""

from perceptron.licensing.features import FeatureRegistry, features
from perceptron.licensing.provider import DevLicenseProvider, LicenseInfo, LicenseProvider

__all__ = ["DevLicenseProvider", "FeatureRegistry", "LicenseInfo", "LicenseProvider", "features"]
