"""Emite una licencia de Perceptron firmada con Ed25519 (ADR-0035), sin instalar el producto.

Solo necesita `cryptography`. Mismo formato que `perceptron license issue`:

    uv run --no-project --with cryptography scripts/issue_license.py \\
        --key ~/claves-perceptron/preteco-2026.key --key-id preteco-2026 \\
        --licensee "Acme SA" --seats 10 --servers 1 --gpus 2 --days 365 \\
        --out license-acme.json

La clave privada no sale de la máquina de Preteco. La licencia se verifica en cada instalación
con la clave pública empaquetada (`perceptron/licensing/keys/<key-id>.pub`).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def canonical(data: dict[str, Any]) -> bytes:
    """Idéntico a `perceptron.licensing.signed.canonical`."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def issue(
    private_pem: bytes,
    key_id: str,
    *,
    licensee: str,
    edition: str = "team",
    seats: int | None = None,
    servers: int | None = None,
    gpus: int | None = None,
    features: list[str] | None = None,
    days: int | None = None,
    now: datetime | None = None,
) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    key = load_pem_private_key(private_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("la clave privada no es Ed25519")
    now = (now or datetime.now(UTC)).replace(microsecond=0)
    terms: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "licensee": licensee,
        "edition": edition,
        "seats": seats,
        "servers": servers,
        "gpus": gpus,
        "features": features or ["*"],
        "issued_at": now.isoformat(),
        "not_before": now.isoformat(),
        "expires_at": (now + timedelta(days=days)).isoformat() if days else None,
    }
    signature = base64.b64encode(key.sign(canonical(terms))).decode()
    return json.dumps(
        {"license": terms, "key_id": key_id, "signature": signature}, indent=2, ensure_ascii=False
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--key", type=Path, required=True, help="Clave privada PEM de Preteco")
    p.add_argument("--key-id", required=True, help="Identificador (nombre del .pub empaquetado)")
    p.add_argument("--licensee", required=True, help="Cliente")
    p.add_argument("--out", type=Path, required=True, help="Archivo license.json a generar")
    p.add_argument("--edition", default="team")
    p.add_argument("--seats", type=int, help="Usuarios del Team Server (sin = sin tope)")
    p.add_argument("--servers", type=int, help="Instalaciones del Team Server")
    p.add_argument("--gpus", type=int, help="Workers GPU")
    p.add_argument("--features", default="*", help="Claves separadas por coma, o *")
    p.add_argument("--days", type=int, help="Vigencia en días (sin = perpetua)")
    a = p.parse_args(argv)
    if a.out.exists():
        print(f"ya existe {a.out}: no se sobrescribe", file=sys.stderr)
        return 1
    doc = issue(
        a.key.read_bytes(),
        a.key_id,
        licensee=a.licensee,
        edition=a.edition,
        seats=a.seats,
        servers=a.servers,
        gpus=a.gpus,
        features=[f.strip() for f in a.features.split(",") if f.strip()],
        days=a.days,
    )
    a.out.write_text(doc, encoding="utf-8")
    lic = json.loads(doc)["license"]
    print(f"licencia {lic['id']} para {lic['licensee']} → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
