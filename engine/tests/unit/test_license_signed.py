"""Licencia firmada Ed25519 validable offline (RF-LIC-01..03, ADR-0035)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from perceptron.api.app import create_app
from perceptron.api.context import EngineContext
from perceptron.cli.main import app as cli
from perceptron.core.config import LicenseSettings, LoggingSettings, Settings
from perceptron.licensing.signed import (
    LicenseStatus,
    SignedLicenseProvider,
    keygen,
    sign,
    verify,
)


def _terms(**over: object) -> dict[str, object]:
    now = datetime(2026, 9, 28, tzinfo=UTC)
    return {
        "id": "lic-1",
        "licensee": "Acme SA",
        "edition": "team",
        "seats": 10,
        "servers": 1,
        "gpus": 2,
        "features": ["*"],
        "issued_at": now.isoformat(),
        "not_before": now.isoformat(),
        "expires_at": (now + timedelta(days=365)).isoformat(),
        **over,
    }


def test_sign_verify_and_tampering() -> None:
    private, public = keygen()
    trusted = {"k1": public}
    doc = sign(_terms(), private, "k1")
    at = datetime(2026, 10, 1, tzinfo=UTC)
    ok = verify(doc, trusted, now=at)
    assert ok.valid and ok.terms is not None and ok.terms.seats == 10

    tampered = json.loads(doc)
    tampered["license"]["seats"] = 1000
    assert verify(json.dumps(tampered), trusted, now=at).status is LicenseStatus.INVALID
    assert verify(doc, {"otra": public}, now=at).status is LicenseStatus.UNTRUSTED
    _, other_public = keygen()
    assert verify(doc, {"k1": other_public}, now=at).status is LicenseStatus.INVALID
    assert verify("no es json", trusted).status is LicenseStatus.INVALID
    assert (
        verify(doc, trusted, now=datetime(2026, 1, 1, tzinfo=UTC)).status
        is LicenseStatus.NOT_YET_VALID
    )
    assert (
        verify(doc, trusted, now=datetime(2028, 1, 1, tzinfo=UTC)).status is LicenseStatus.EXPIRED
    )


def test_provider_without_and_with_enforcement(tmp_path: Path) -> None:
    private, public = keygen()
    path = tmp_path / "license.json"
    loose = SignedLicenseProvider(path, {"k1": public})
    assert loose.check().status is LicenseStatus.MISSING
    assert loose.is_feature_enabled("llm.agent")  # v1: sin enforcement, todo habilitado
    strict = SignedLicenseProvider(path, {"k1": public}, enforce=True)
    assert not strict.is_feature_enabled("llm.agent")
    now = datetime.now(UTC)
    path.write_text(
        sign(
            _terms(
                features=["llm.copilot"],
                issued_at=now.isoformat(),
                not_before=None,
                expires_at=None,
            ),
            private,
            "k1",
        ),
        encoding="utf-8",
    )
    assert strict.is_feature_enabled("llm.copilot") and not strict.is_feature_enabled("llm.agent")
    info = strict.current()
    assert info.licensee == "Acme SA" and info.limits == {"seats": 10, "servers": 1, "gpus": 2}


def test_license_api_and_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    keys = tmp_path / "claves"
    assert (
        runner.invoke(cli, ["license", "keygen", "--out", str(keys), "--key-id", "k1"]).exit_code
        == 0
    )
    assert (
        runner.invoke(cli, ["license", "keygen", "--out", str(keys), "--key-id", "k1"]).exit_code
        != 0
    )
    lic = tmp_path / "license.json"
    issued = runner.invoke(
        cli,
        [
            "license",
            "issue",
            "--key",
            str(keys / "k1.key"),
            "--key-id",
            "k1",
            "--licensee",
            "Acme SA",
            "--seats",
            "5",
            "--days",
            "30",
            "--out",
            str(lic),
        ],
    )
    assert issued.exit_code == 0, issued.stdout
    public = (keys / "k1.pub").read_text(encoding="utf-8")
    settings = Settings(
        workspace_dir=tmp_path / "ws ñ",
        logging=LoggingSettings(to_file=False),
        license=LicenseSettings(public_keys={"k1": public}),
    )
    ctx = EngineContext.create(settings)
    try:
        with TestClient(create_app(ctx=ctx)) as client:
            assert client.get("/api/v1/system/license").json()["status"] == "missing"
            bad = client.put(
                "/api/v1/system/license", json={"content": lic.read_text().replace("Acme", "Otra")}
            )
            assert bad.status_code == 422 and bad.json()["details"]["status"] == "invalid"
            view = client.put("/api/v1/system/license", json={"content": lic.read_text()}).json()
            assert (
                view["status"] == "valid" and view["seats"] == 5 and view["licensee"] == "Acme SA"
            )
            assert view["enforce"] is False
            assert (settings.workspace_dir / "license.json").is_file()
    finally:
        ctx.close()
