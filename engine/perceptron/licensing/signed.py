"""Licencia en archivo firmado con Ed25519, validable offline (D1, ADR-0035).

Formato (`license.json`):

    {"license": {...campos...}, "key_id": "preteco-2026", "signature": "<base64>"}

La firma cubre el JSON canónico de `license` (claves ordenadas, sin espacios, UTF-8). Campos:
`id`, `licensee`, `edition`, `seats` (usuarios activos del Team Server), `servers`
(instalaciones del Team Server), `gpus` (workers GPU), `features` (lista o `["*"]`),
`issued_at`, `not_before`, `expires_at` (opcional).

Las claves públicas confiables vienen en el paquete (`licensing/keys/*.pub`, PEM o raw
base64) o en `PERCEPTRON_LICENSE__PUBLIC_KEYS`. La privada nunca está en el producto: se firma
con `perceptron license issue` en la máquina de Preteco.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from perceptron.licensing.provider import LicenseInfo


class LicenseStatus(StrEnum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"  # formato o firma
    UNTRUSTED = "untrusted"  # firmada con una clave que no conocemos
    NOT_YET_VALID = "not_yet_valid"
    EXPIRED = "expired"


class LicenseTerms(BaseModel):
    id: str = Field(min_length=1)
    licensee: str = Field(min_length=1)
    edition: str = Field(default="team")
    seats: int | None = Field(default=None, ge=0, description="None = sin tope")
    servers: int | None = Field(default=None, ge=0)
    gpus: int | None = Field(default=None, ge=0)
    features: list[str] = Field(default_factory=lambda: ["*"])
    issued_at: datetime
    not_before: datetime | None = None
    expires_at: datetime | None = None


class LicenseFile(BaseModel):
    license: dict[str, Any]
    key_id: str
    signature: str


def canonical(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _load_public(raw: str) -> Any:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    text = raw.strip()
    if text.startswith("-----BEGIN"):
        key = load_pem_public_key(text.encode())
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("la clave pública no es Ed25519")
        return key
    return Ed25519PublicKey.from_public_bytes(base64.b64decode(text))


def bundled_keys() -> dict[str, str]:
    """Claves públicas empaquetadas: `licensing/keys/<key_id>.pub`."""
    out: dict[str, str] = {}
    folder = resources.files("perceptron.licensing").joinpath("keys")
    if not folder.is_dir():
        return out
    for entry in folder.iterdir():
        if entry.name.endswith(".pub"):
            out[entry.name.removesuffix(".pub")] = entry.read_text("utf-8")
    return out


@dataclass(frozen=True, slots=True)
class LicenseCheck:
    status: LicenseStatus
    terms: LicenseTerms | None = None
    message: str = ""

    @property
    def valid(self) -> bool:
        return self.status is LicenseStatus.VALID


def verify(
    content: str | bytes, trusted: dict[str, str], now: datetime | None = None
) -> LicenseCheck:
    """Valida formato, firma y vigencia. Nunca lanza: informa el estado."""
    now = now or datetime.now(UTC)
    try:
        doc = LicenseFile.model_validate_json(content)
    except ValidationError as exc:
        return LicenseCheck(LicenseStatus.INVALID, message=f"formato inválido: {exc.errors()[:1]}")
    raw_key = trusted.get(doc.key_id)
    if raw_key is None:
        return LicenseCheck(LicenseStatus.UNTRUSTED, message=f"clave desconocida: {doc.key_id}")
    try:
        from cryptography.exceptions import InvalidSignature

        _load_public(raw_key).verify(base64.b64decode(doc.signature), canonical(doc.license))
    except (InvalidSignature, ValueError, TypeError):
        return LicenseCheck(LicenseStatus.INVALID, message="la firma no es válida")
    try:
        terms = LicenseTerms.model_validate(doc.license)
    except ValidationError as exc:
        return LicenseCheck(
            LicenseStatus.INVALID, message=f"términos inválidos: {exc.errors()[:1]}"
        )
    if terms.not_before and now < terms.not_before:
        return LicenseCheck(LicenseStatus.NOT_YET_VALID, terms, "todavía no está vigente")
    if terms.expires_at and now >= terms.expires_at:
        return LicenseCheck(LicenseStatus.EXPIRED, terms, "venció")
    return LicenseCheck(LicenseStatus.VALID, terms)


def sign(terms: dict[str, Any], private_pem: bytes, key_id: str) -> str:
    """Emite un `license.json` (herramienta de Preteco; la clave privada no viaja)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    LicenseTerms.model_validate(terms)
    key = load_pem_private_key(private_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("la clave privada no es Ed25519")
    signature = base64.b64encode(key.sign(canonical(terms))).decode()
    return json.dumps(
        {"license": terms, "key_id": key_id, "signature": signature}, indent=2, ensure_ascii=False
    )


def keygen() -> tuple[bytes, str]:
    """Par nuevo: (privada PEM, pública PEM)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private, public.decode()


class SignedLicenseProvider:
    """Proveedor con archivo firmado. Sin `enforce` (v1, RF-LIC-03) todo queda habilitado y
    la licencia solo se informa; con `enforce`, las features salen de la licencia."""

    def __init__(self, path: Path, trusted: dict[str, str], *, enforce: bool = False) -> None:
        self.path = path
        self.trusted = trusted
        self.enforce = enforce

    def check(self) -> LicenseCheck:
        if not self.path.is_file():
            return LicenseCheck(LicenseStatus.MISSING, message="no hay licencia instalada")
        return verify(self.path.read_bytes(), self.trusted)

    def current(self) -> LicenseInfo:
        result = self.check()
        if not result.terms:
            return LicenseInfo(licensee="sin licencia", plan=result.status.value)
        t = result.terms
        features = None if "*" in t.features else frozenset(t.features)
        limits = {
            k: v
            for k, v in (("seats", t.seats), ("servers", t.servers), ("gpus", t.gpus))
            if v is not None
        }
        return LicenseInfo(
            licensee=t.licensee,
            plan=t.edition if result.valid else f"{t.edition} ({result.status.value})",
            expires_at=t.expires_at,
            enabled_features=features,
            limits=limits,
        )

    def is_feature_enabled(self, feature_key: str) -> bool:
        if not self.enforce:
            return True
        result = self.check()
        if not result.valid or result.terms is None:
            return False
        return "*" in result.terms.features or feature_key in result.terms.features


def license_provider(settings: Any) -> SignedLicenseProvider:
    """Proveedor configurado desde los settings del Engine."""
    cfg = settings.license
    path = cfg.path or settings.workspace_dir / "license.json"
    return SignedLicenseProvider(path, {**bundled_keys(), **cfg.public_keys}, enforce=cfg.enforce)
