"""Autenticación local del Team Server (RF-SRV-01): cookies, CSRF, bearer, rotación, bloqueo."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from srv_helpers import ADMIN_EMAIL, ADMIN_PASSWORD, API, PASSWORD, UserFactory, login, ok

from perceptron_server import accounts


def test_health_public_api_requires_session(anon: TestClient) -> None:
    assert anon.get(f"{API}/system/health").status_code == 200
    assert ok(anon.get(f"{API}/auth/config"))["mode"] == "server"
    r = anon.get(f"{API}/projects")
    assert r.status_code == 401 and r.json()["code"] == "unauthorized"
    assert anon.get(f"{API}/auth/me").status_code == 401
    assert anon.get(f"{API}/admin/users").status_code == 401


def test_login_sets_hardened_cookies_and_me(app: FastAPI, anon: TestClient) -> None:
    c = TestClient(app)
    r = c.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL.upper(), "password": ADMIN_PASSWORD})
    me = ok(r)
    assert me["is_server_admin"] and me["user"]["email"] == ADMIN_EMAIL
    assert [w["name"] for w in me["workspaces"]] == ["Equipo"]
    cookies = r.headers.get_list("set-cookie")
    by_name = {h.split("=", 1)[0]: h for h in cookies}
    assert set(by_name) == {"pt_access", "pt_refresh", "pt_csrf"}
    for name in ("pt_access", "pt_refresh"):
        assert "HttpOnly" in by_name[name] and "SameSite=Strict" in by_name[name]
    assert "HttpOnly" not in by_name["pt_csrf"]  # la UI la lee para la cabecera
    assert ADMIN_PASSWORD not in r.text
    assert ok(c.get(f"{API}/projects")) == []


def test_wrong_password_is_generic_and_locks_account(app: FastAPI, admin: TestClient) -> None:
    anon = TestClient(app)
    for _ in range(5):
        r = anon.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": "mala"})
        assert r.status_code == 401
        assert r.json()["message"] == "email o contraseña incorrectos"
    unknown = anon.post(f"{API}/auth/login", json={"email": "nadie@x.test", "password": "mala"})
    assert unknown.status_code == 401 and unknown.json()["message"] == r.json()["message"]
    locked = anon.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert locked.status_code == 429
    events = ok(admin.get(f"{API}/admin/audit", params={"action": "auth.login_failed"}))
    assert len(events) >= 6 and events[0]["details"]["email"]


def test_csrf_required_for_cookie_writes(app: FastAPI, admin: TestClient) -> None:
    no_header = TestClient(app, cookies=dict(admin.cookies))
    r = no_header.post(f"{API}/projects", json={"name": "Sin CSRF"})
    assert r.status_code == 403 and r.json()["code"] == "csrf_failed"
    wrong = no_header.post(
        f"{API}/projects", json={"name": "CSRF malo"}, headers={"X-CSRF-Token": "otro"}
    )
    assert wrong.status_code == 403
    assert no_header.get(f"{API}/projects").status_code == 200  # lecturas sin CSRF
    project = ok(admin.post(f"{API}/projects", json={"name": "Con CSRF"}), 201)
    assert project["workspace_id"] and project["scope"] == "team"


def test_bearer_tokens_rotation_and_reuse(
    app: FastAPI, anon: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = ok(
        anon.post(f"{API}/auth/token", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    )
    bearer = TestClient(app, headers={"Authorization": f"Bearer {tokens['access_token']}"})
    # Sin cookies no hay CSRF que pedir: el bearer no viaja solo.
    assert ok(bearer.post(f"{API}/projects", json={"name": "Por CLI"}), 201)["id"]
    assert (
        TestClient(app, headers={"Authorization": "Bearer basura"})
        .get(f"{API}/projects")
        .status_code
        == 401
    )

    rotated = ok(anon.post(f"{API}/auth/refresh", json={"refresh_token": tokens["refresh_token"]}))
    assert rotated["refresh_token"] != tokens["refresh_token"]
    # Reusar el token viejo pasada la gracia = posible robo: se revoca toda la sesión.
    monkeypatch.setattr(accounts, "REFRESH_GRACE", timedelta(0))
    reuse = anon.post(f"{API}/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert reuse.status_code == 401
    again = anon.post(f"{API}/auth/refresh", json={"refresh_token": rotated["refresh_token"]})
    assert again.status_code == 401
    assert bearer.get(f"{API}/projects").status_code == 401  # la sesión quedó cerrada


def test_expired_access_cookie_is_renewed_transparently(admin: TestClient) -> None:
    old_refresh = admin.cookies["pt_refresh"]
    admin.cookies.delete("pt_access")  # como si hubiera vencido
    r = admin.get(f"{API}/auth/me")
    assert r.status_code == 200
    renewed = {h.split("=", 1)[0] for h in r.headers.get_list("set-cookie")}
    assert renewed == {"pt_access", "pt_refresh"}
    assert admin.cookies["pt_refresh"] != old_refresh
    assert ok(admin.get(f"{API}/projects")) == []


def test_logout_and_password_change_close_sessions(app: FastAPI, make_user: UserFactory) -> None:
    _, ana = make_user("ana@preteco.test", "editor")
    cookies = dict(ana.cookies)
    assert ok(ana.post(f"{API}/auth/logout")) is None
    stale = TestClient(app, cookies=cookies)
    assert stale.get(f"{API}/auth/me").status_code == 401

    ana = login(app, "ana@preteco.test", PASSWORD)
    bad = ana.post(
        f"{API}/auth/password", json={"current_password": "x", "new_password": "otra-clave-larga"}
    )
    assert bad.status_code == 401
    short = ana.post(
        f"{API}/auth/password", json={"current_password": PASSWORD, "new_password": "corta"}
    )
    assert short.status_code == 422
    r = ana.post(
        f"{API}/auth/password",
        json={"current_password": PASSWORD, "new_password": "una-clave-nueva-larga"},
    )
    assert r.status_code == 204
    assert TestClient(app, cookies=dict(ana.cookies)).get(f"{API}/auth/me").status_code == 401
    assert login(app, "ana@preteco.test", "una-clave-nueva-larga")


def test_deactivated_user_loses_access(
    admin: TestClient, make_user: UserFactory, app: FastAPI
) -> None:
    uid, beto = make_user("beto@preteco.test", "viewer")
    assert beto.get(f"{API}/projects").status_code == 200
    ok(admin.patch(f"{API}/admin/users/{uid}", json={"is_active": False}))
    assert beto.get(f"{API}/projects").status_code == 401
    r = TestClient(app).post(
        f"{API}/auth/login", json={"email": "beto@preteco.test", "password": PASSWORD}
    )
    assert r.status_code == 401


def test_last_server_admin_cannot_be_removed(admin: TestClient) -> None:
    me = ok(admin.get(f"{API}/auth/me"))
    r = admin.patch(f"{API}/admin/users/{me['user']['id']}", json={"is_server_admin": False})
    assert r.status_code == 409
