"""Claves de API (RF-LLM-08).

- Desktop: keychain del SO vía `keyring`.
- Servidor: archivo cifrado con AES-GCM; la clave maestra viene de `PERCEPTRON_MASTER_KEY`
  (32 bytes en base64) o de un KMS que la inyecte en esa variable.
- Siempre, como último recurso: variables de entorno (CI, contenedores).

Los valores nunca se loguean ni llegan a la auditoría; solo se guarda la referencia.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets as pysecrets
from pathlib import Path
from typing import Protocol

from perceptron.storage.filesystem import atomic_write_text

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "perceptron"
MASTER_KEY_ENV = "PERCEPTRON_MASTER_KEY"
NONCE_BYTES = 12


class SecretStore(Protocol):
    def get(self, name: str) -> str | None: ...

    def set(self, name: str, value: str) -> None: ...

    def delete(self, name: str) -> None: ...


class MemorySecrets:
    """Para tests."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


class EnvSecrets:
    """Solo lectura: `ANTHROPIC_API_KEY`, etc."""

    def get(self, name: str) -> str | None:
        return os.environ.get(name) or None

    def set(self, name: str, value: str) -> None:
        raise PermissionError("las variables de entorno no se escriben desde el Engine")

    def delete(self, name: str) -> None:
        raise PermissionError("las variables de entorno no se escriben desde el Engine")


class KeyringSecrets:
    def get(self, name: str) -> str | None:
        import keyring

        try:
            value = keyring.get_password(KEYRING_SERVICE, name)
        except Exception:  # sin backend (Linux headless): se sigue con el resto de la cadena
            logger.debug("keyring no disponible", exc_info=True)
            return None
        return str(value) if value else None

    def set(self, name: str, value: str) -> None:
        import keyring

        keyring.set_password(KEYRING_SERVICE, name, value)

    def delete(self, name: str) -> None:
        import keyring

        try:
            keyring.delete_password(KEYRING_SERVICE, name)
        except Exception:
            logger.debug("no se pudo borrar del keyring", exc_info=True)


class EncryptedFileSecrets:
    """Archivo `{nombre: base64(nonce + ciphertext)}` cifrado con AES-GCM."""

    def __init__(self, path: Path, master_key: bytes) -> None:
        if len(master_key) != 32:
            raise ValueError("la clave maestra debe tener 32 bytes (AES-256)")
        self.path = path
        self._key = master_key

    @staticmethod
    def generate_key() -> str:
        return base64.b64encode(pysecrets.token_bytes(32)).decode()

    def _load(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        data: dict[str, str] = json.loads(self.path.read_text(encoding="utf-8"))
        return data

    def get(self, name: str) -> str | None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        blob = self._load().get(name)
        if blob is None:
            return None
        raw = base64.b64decode(blob)
        plain = AESGCM(self._key).decrypt(raw[:NONCE_BYTES], raw[NONCE_BYTES:], name.encode())
        return plain.decode()

    def set(self, name: str, value: str) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = pysecrets.token_bytes(NONCE_BYTES)
        ct = AESGCM(self._key).encrypt(nonce, value.encode(), name.encode())
        data = self._load()
        data[name] = base64.b64encode(nonce + ct).decode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(data, indent=2))

    def delete(self, name: str) -> None:
        data = self._load()
        if data.pop(name, None) is not None:
            atomic_write_text(self.path, json.dumps(data, indent=2))


class ChainSecrets:
    """Lee del primero que tenga el valor; escribe en el primero escribible."""

    def __init__(self, stores: list[SecretStore]) -> None:
        self.stores = stores

    def get(self, name: str) -> str | None:
        for store in self.stores:
            value = store.get(name)
            if value:
                return value
        return None

    def set(self, name: str, value: str) -> None:
        try:
            self.stores[0].set(name, value)
        except PermissionError:
            raise
        except Exception as exc:  # p. ej. Linux sin Secret Service: keyring.errors.NoKeyringError
            from perceptron.core.errors import ValidationError

            raise ValidationError(
                "No hay un llavero del sistema disponible para guardar la credencial. "
                f"Definí {MASTER_KEY_ENV} para guardarla cifrada en archivo.",
                details={"reason": type(exc).__name__},
            ) from exc

    def delete(self, name: str) -> None:
        for store in self.stores:
            try:
                store.delete(name)
            except PermissionError:
                continue


def default_secret_store(*, server: bool, secrets_file: Path) -> SecretStore:
    master = os.environ.get(MASTER_KEY_ENV)
    primary: SecretStore
    if server:
        if not master:
            return ChainSecrets([EnvSecrets()])
        primary = EncryptedFileSecrets(secrets_file, base64.b64decode(master))
    else:
        primary = KeyringSecrets()
    return ChainSecrets([primary, EnvSecrets()])
