"""Destinos de red pedidos por usuarios: SSRF (Capa 7, ASVS V12.6)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from perceptron.core.config import NetworkSettings, RuntimeMode, Settings
from perceptron.core.errors import ValidationError
from perceptron.core.netguard import NetPolicy, check_host, check_url, same_origin
from perceptron.data.sources.stream import RestConfig, RestSource, WebSocketConfig, WebSocketSource

SERVER = NetPolicy(allow_private=False)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/x",
        "http://10.0.0.5/api",
        "http://192.168.1.1",
        "http://172.16.0.1",
        "http://100.64.0.1",  # CGNAT
        "http://169.254.169.254/latest/meta-data/",  # metadata de la nube
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",  # IPv4 mapeada en IPv6
        "http://[fd00::1]/",
        "http://0.0.0.0/",
        "http://224.0.0.1/",
    ],
)
def test_server_policy_blocks_internal_addresses(url: str) -> None:
    with pytest.raises(ValidationError, match="dirección interna"):
        check_url(url, SERVER)


def test_public_addresses_and_allowed_hosts_pass() -> None:
    assert check_url("https://93.184.216.34/datos", SERVER)
    allowed = NetPolicy(allow_private=False, allowed_hosts=frozenset({"10.0.0.5"}))
    assert check_url("http://10.0.0.5/api", allowed)
    check_host("10.0.0.5", 5432, allowed)
    with pytest.raises(ValidationError):
        check_host("10.0.0.6", 5432, allowed)


@pytest.mark.parametrize(
    "url",
    ["ftp://93.184.216.34/x", "file:///etc/passwd", "gopher://x", "https://u:p@93.184.216.34/"],
)
def test_rejects_other_schemes_and_credentials_in_url(url: str) -> None:
    with pytest.raises(ValidationError):
        check_url(url, NetPolicy())


def test_desktop_allows_local_services_and_server_blocks_them(tmp_path: Path) -> None:
    desktop = Settings(workspace_dir=tmp_path).net_policy()
    server = Settings(workspace_dir=tmp_path, mode=RuntimeMode.SERVER).net_policy()
    assert desktop.allow_private and not server.allow_private
    check_url("http://127.0.0.1:9000/x", desktop)
    with pytest.raises(ValidationError):
        check_url("http://127.0.0.1:9000/x", server)
    opened = Settings(
        workspace_dir=tmp_path, mode=RuntimeMode.SERVER, network=NetworkSettings(allow_private=True)
    )
    assert opened.net_policy().allow_private


def test_same_origin() -> None:
    assert same_origin("https://a.test/p1", "https://a.test/p2?x=1")
    assert not same_origin("https://a.test/p1", "https://b.test/p1")
    assert not same_origin("https://a.test/p1", "http://a.test/p1")
    assert not same_origin("https://a.test/p1", "https://a.test:8443/p1")


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_rest_source_checks_every_page_and_never_follows_other_hosts() -> None:
    calls: list[str] = []

    def to_internal(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        return httpx.Response(200, json=[])

    blocked = RestSource(
        RestConfig(url="http://169.254.169.254/latest"), lambda: _client(to_internal), SERVER
    )
    with pytest.raises(ValidationError):
        blocked.fetch({}, None)
    assert calls == []  # no llegó a conectar

    def link_elsewhere(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        return httpx.Response(
            200, json=[{"n": 1}], headers={"Link": '<https://otro.test/p2>; rel="next"'}
        )

    cfg = RestConfig(url="https://a.test/p1", pagination="link", auth="bearer")
    source = RestSource(cfg, lambda: _client(link_elsewhere))
    with pytest.raises(ValidationError, match="otro host"):
        source.fetch({}, "secreto")
    assert calls == ["https://a.test/p1"]  # el token no viajó a otro host


def test_rest_source_does_not_follow_redirects() -> None:
    def redirect(req: httpx.Request) -> httpx.Response:
        if req.url.host == "a.test":
            return httpx.Response(302, headers={"Location": "http://127.0.0.1/admin"})
        raise AssertionError("siguió el redirect")

    from perceptron.data.sources.stream import default_http

    assert default_http().follow_redirects is False
    source = RestSource(RestConfig(url="https://a.test/x"), lambda: _client(redirect))
    with pytest.raises(httpx.HTTPStatusError):
        source.fetch({}, None)


def test_websocket_source_checks_url_before_connecting() -> None:
    source = WebSocketSource(WebSocketConfig(url="ws://127.0.0.1:9/stream"), SERVER)
    with pytest.raises(ValidationError):
        source.fetch({}, None)
