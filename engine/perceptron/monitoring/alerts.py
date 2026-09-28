"""Alertas del monitoreo (RF-MON-04): en la app, email (SMTP) y webhook genérico.

El webhook recibe JSON con `text` (lo que muestran Slack y Teams con un "incoming webhook")
más los campos estructurados. La URL del webhook puede tener tokens: va al almacén de
secretos (`deployment/<id>/webhook`), no a la entidad.
"""

from __future__ import annotations

import logging
import smtplib
from collections.abc import Callable
from datetime import timedelta
from email.message import EmailMessage
from typing import TYPE_CHECKING, Any

import httpx

from perceptron.core.errors import ValidationError
from perceptron.core.netguard import check_url
from perceptron.domain.enums import AlertKind, AlertStatus, Severity
from perceptron.domain.models import Alert, Deployment, utcnow

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

logger = logging.getLogger(__name__)
ALERT_TOPIC = "alert.created"
SMTP_SECRET = "alerts/smtp/password"  # noqa: S105 - nombre del secreto, no su valor


def default_http() -> httpx.Client:
    return httpx.Client(timeout=10.0)


def default_smtp(host: str, port: int) -> smtplib.SMTP:
    return smtplib.SMTP(host, port, timeout=15)


def webhook_secret(deployment_id: str) -> str:
    return f"deployment/{deployment_id}/webhook"


class AlertService:
    def __init__(
        self,
        ctx: EngineContext,
        *,
        http: Callable[[], httpx.Client] | None = None,
        smtp: Callable[[str, int], smtplib.SMTP] | None = None,
    ) -> None:
        self.ctx = ctx
        self.settings = ctx.settings.alerts
        # Se resuelven al usarse: los tests reemplazan default_http/default_smtp del módulo.
        self._http = http
        self._smtp = smtp

    def raise_alert(
        self,
        *,
        project_id: str,
        kind: AlertKind,
        severity: Severity,
        title: str,
        message: str = "",
        deployment: Deployment | None = None,
        details: dict[str, Any] | None = None,
    ) -> Alert | None:
        """Crea la alerta y la envía; `None` si ya hay una igual abierta (cooldown)."""
        repo = self.ctx.repo(Alert)
        cutoff = utcnow() - timedelta(seconds=self.settings.cooldown_s)
        for open_ in repo.list(filters={"project_id": project_id}, limit=200):
            same = open_.kind is kind and open_.deployment_id == (
                deployment.id if deployment else None
            )
            if same and open_.status is AlertStatus.OPEN and open_.created_at >= cutoff:
                return None
        alert = Alert(
            project_id=project_id,
            deployment_id=deployment.id if deployment else None,
            kind=kind,
            severity=severity,
            title=title,
            message=message,
            details=details or {},
        )
        channels = ["app"]
        cfg = (deployment.monitoring if deployment else {}) or {}
        if self._email(alert, cfg.get("email") or []):
            channels.append("email")
        if deployment is not None and self._webhook(alert, deployment):
            channels.append("webhook")
        alert.channels = channels
        repo.add(alert)
        self.ctx.events.publish(ALERT_TOPIC, alert_id=alert.id, project_id=project_id)
        return alert

    def _link(self, alert: Alert) -> str | None:
        base = self.settings.public_url
        return f"{base.rstrip('/')}/projects/{alert.project_id}/monitoring" if base else None

    def _email(self, alert: Alert, recipients: list[str]) -> bool:
        s = self.settings
        if not recipients or not s.smtp_host or not s.smtp_from:
            return False
        msg = EmailMessage()
        msg["Subject"] = f"[Perceptron · {alert.severity.value}] {alert.title}"
        msg["From"] = s.smtp_from
        msg["To"] = ", ".join(recipients)
        link = self._link(alert)
        msg.set_content(f"{alert.message}\n\n{link or ''}".strip())
        try:
            with (self._smtp or default_smtp)(s.smtp_host, s.smtp_port) as server:
                if s.smtp_starttls:
                    server.starttls()
                password = self.ctx.llm.secrets.get(SMTP_SECRET)
                if s.smtp_user and password:
                    server.login(s.smtp_user, password)
                server.send_message(msg)
        except (OSError, smtplib.SMTPException):
            logger.exception("no se pudo enviar la alerta por email")
            return False
        return True

    def _webhook(self, alert: Alert, deployment: Deployment) -> bool:
        url = self.ctx.llm.secrets.get(webhook_secret(deployment.id))
        if not url:
            return False
        try:
            check_url(url, self.ctx.settings.net_policy())  # la red pudo cambiar desde que se cargó
        except ValidationError:
            logger.warning("webhook de alertas bloqueado por la política de red")
            return False
        link = self._link(alert)
        payload = {
            "text": f"*[{alert.severity.value}] {alert.title}*\n{alert.message}"
            + (f"\n{link}" if link else ""),
            "title": alert.title,
            "severity": alert.severity.value,
            "kind": alert.kind.value,
            "project_id": alert.project_id,
            "deployment_id": deployment.id,
            "details": alert.details,
        }
        try:
            with (self._http or default_http)() as client:
                client.post(url, json=payload).raise_for_status()
        except httpx.HTTPError:
            logger.exception("no se pudo enviar la alerta al webhook")
            return False
        return True
