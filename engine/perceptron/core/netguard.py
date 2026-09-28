"""Destinos de red pedidos por usuarios: protección contra SSRF (Capa 7, ASVS V12.6).

Las fuentes REST/WebSocket, los webhooks de alertas y las bases de datos remotas conectan a
URLs que carga un usuario. En el Team Server eso no puede servir para llegar a la red interna
(loopback, LAN, link-local, metadata de la nube): se resuelve el host y se rechaza si alguna
dirección no es pública, salvo los hosts que el administrador habilite. En el desktop las
direcciones internas se permiten por defecto (una API de la intranet, un servicio local).

La verificación se hace antes de cada conexión (también en cada página y sin seguir redirects),
lo que acota el DNS rebinding a la ventana entre la resolución y la conexión.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from perceptron.core.errors import ValidationError

DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


@dataclass(frozen=True)
class NetPolicy:
    allow_private: bool = True
    allowed_hosts: frozenset[str] = field(default_factory=frozenset)

    def allows_host(self, host: str) -> bool:
        return self.allow_private or host.lower().rstrip(".") in self.allowed_hosts


def _is_public(ip: IPAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _resolve(host: str, port: int) -> Iterable[IPAddress]:
    try:
        return {
            ipaddress.ip_address(info[4][0].split("%", 1)[0])
            for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }
    except (socket.gaierror, UnicodeError, ValueError):
        raise ValidationError(f"no se pudo resolver el host {host!r}") from None


def check_host(host: str, port: int, policy: NetPolicy) -> None:
    """El host (nombre o IP) es un destino permitido por la política."""
    if not host:
        raise ValidationError("falta el host")
    if policy.allows_host(host):
        return
    for ip in _resolve(host.strip("[]"), port):
        if not _is_public(ip):
            raise ValidationError(
                "destino de red no permitido: el host apunta a una dirección interna",
                details={"host": host},
            )


def check_url(url: str, policy: NetPolicy, *, schemes: Iterable[str] = ("http", "https")) -> str:
    """Valida esquema y host de `url`; devuelve la URL sin cambios."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in set(schemes):
        raise ValidationError(f"esquema no permitido: {scheme or '(ninguno)'}")
    if parts.username or parts.password:
        raise ValidationError("las credenciales van en el almacén de secretos, no en la URL")
    try:
        port = parts.port or DEFAULT_PORTS.get(scheme, 443)
    except ValueError:
        raise ValidationError("puerto inválido en la URL") from None
    check_host(parts.hostname or "", port, policy)
    return url


def same_origin(a: str, b: str) -> bool:
    pa, pb = urlsplit(a), urlsplit(b)
    return (pa.scheme.lower(), pa.hostname, pa.port) == (pb.scheme.lower(), pb.hostname, pb.port)
