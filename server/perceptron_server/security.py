"""Primitivas de seguridad: Argon2id para contraseñas, JWT de acceso y tokens opacos."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Literal

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

ISSUER = "perceptron-team-server"
_ALGORITHM = "HS256"
_hasher = PasswordHasher()  # Argon2id con los parámetros recomendados por RFC 9106


def hash_password(password: str) -> str:
    return _hasher.hash(password)


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return _hasher.hash(secrets.token_urlsafe(16))


def verify_password(stored: str | None, password: str) -> bool:
    """Tiempo similar exista o no la cuenta (evita enumerar usuarios por timing)."""
    try:
        return _hasher.verify(stored or _dummy_hash(), password) and stored is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(stored: str) -> bool:
    return _hasher.check_needs_rehash(stored)


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class AccessClaims:
    user_id: str
    session_id: str
    expires_at: datetime


def issue_access(user_id: str, session_id: str, secret: str, ttl_s: int) -> tuple[str, datetime]:
    now = datetime.now(UTC)
    exp = now + timedelta(seconds=ttl_s)
    claims = {
        "sub": user_id,
        "sid": session_id,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "iss": ISSUER,
        "typ": "access",
    }
    return jwt.encode(claims, secret, algorithm=_ALGORITHM), exp


def decode_access(token: str, secret: str) -> AccessClaims | Literal["expired"] | None:
    """Claims si el token es válido, `"expired"` si solo venció, `None` si es inválido."""
    try:
        data = jwt.decode(
            token,
            secret,
            algorithms=[_ALGORITHM],
            issuer=ISSUER,
            options={"require": ["sub", "sid", "exp", "iss", "typ"]},
        )
    except jwt.ExpiredSignatureError:
        return "expired"
    except jwt.PyJWTError:
        return None
    if data.get("typ") != "access":
        return None
    return AccessClaims(
        user_id=str(data["sub"]),
        session_id=str(data["sid"]),
        expires_at=datetime.fromtimestamp(int(data["exp"]), UTC),
    )
