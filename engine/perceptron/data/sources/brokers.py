"""Fuentes streaming de brokers (RF-ING-05): Kafka y MQTT, sobre la interfaz `StreamSource`.

Cada `fetch` se conecta, lee mensajes JSON hasta `max_messages` o `idle_timeout_s` sin
mensajes nuevos, confirma lo leído y se desconecta:
- Kafka: consumidor con `group_id` y commit manual; la próxima lectura sigue donde quedó.
- MQTT: sesión persistente (`clean_session=False`, mismo `client_id`, QoS 1): el broker guarda
  lo que llega entre sondeos.

Los hosts pasan por la política de red (SSRF, `core/netguard.py`). Las credenciales van al
almacén de secretos. Extra opcional `streaming` (kafka-python Apache-2.0, paho-mqtt EDL-1.0).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from perceptron.core.errors import ValidationError
from perceptron.core.netguard import NetPolicy, check_host

MAX_MESSAGES = 100_000


def _rows(raw: bytes | str, records_path: str | None) -> list[dict[str, Any]]:
    from perceptron.data.sources.stream import _flatten, dig

    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []  # mensaje no JSON: se descarta
    data = dig(msg, records_path)
    items = data if isinstance(data, list) else [data]
    return [_flatten(i) for i in items if i is not None]


# ------------------------------------------------------------------ Kafka


class KafkaConfig(BaseModel):
    bootstrap_servers: list[str] = Field(min_length=1, description="host:puerto")
    topic: str = Field(min_length=1)
    group_id: str = Field(default="perceptron", min_length=1)
    auth: Literal["none", "sasl_plain", "sasl_scram_256", "sasl_scram_512"] = "none"
    username: str | None = None
    ssl: bool = False
    records_path: str | None = None
    max_messages: int = Field(default=1000, ge=1, le=MAX_MESSAGES)
    idle_timeout_s: float = Field(default=5.0, gt=0, le=300)


KafkaFactory = Callable[..., Any]


def default_kafka(**kwargs: Any) -> Any:
    try:
        from kafka import KafkaConsumer
    except ImportError:
        raise ValidationError("falta el extra `streaming` (kafka-python)") from None
    topic = kwargs.pop("topic")
    return KafkaConsumer(topic, **kwargs)


class KafkaSource:
    def __init__(
        self, config: KafkaConfig, factory: KafkaFactory | None = None, net: NetPolicy | None = None
    ) -> None:
        self.config = config
        self.factory = factory or default_kafka
        self.net = net or NetPolicy()

    def _kwargs(self, secret: str | None) -> dict[str, Any]:
        cfg = self.config
        kwargs: dict[str, Any] = {
            "topic": cfg.topic,
            "bootstrap_servers": cfg.bootstrap_servers,
            "group_id": cfg.group_id,
            "enable_auto_commit": False,
            "auto_offset_reset": "earliest",
            "consumer_timeout_ms": int(cfg.idle_timeout_s * 1000),
        }
        protocol = "SSL" if cfg.ssl else "PLAINTEXT"
        if cfg.auth != "none":
            if not (cfg.username and secret):
                raise ValidationError("la fuente Kafka requiere usuario y contraseña")
            protocol = "SASL_SSL" if cfg.ssl else "SASL_PLAINTEXT"
            kwargs["sasl_mechanism"] = {
                "sasl_plain": "PLAIN",
                "sasl_scram_256": "SCRAM-SHA-256",
                "sasl_scram_512": "SCRAM-SHA-512",
            }[cfg.auth]
            kwargs["sasl_plain_username"] = cfg.username
            kwargs["sasl_plain_password"] = secret
        kwargs["security_protocol"] = protocol
        return kwargs

    def fetch(self, state: dict[str, Any], secret: str | None) -> Any:
        from perceptron.data.sources.stream import Batch

        cfg = self.config
        for server in cfg.bootstrap_servers:
            host, _, port = server.rpartition(":")
            if not host or not port.isdigit():
                raise ValidationError(f"bootstrap inválido: {server!r} (host:puerto)")
            check_host(host, int(port), self.net)
        consumer = self.factory(**self._kwargs(secret))
        rows: list[dict[str, Any]] = []
        try:
            while len(rows) < cfg.max_messages:
                polled = consumer.poll(
                    timeout_ms=int(cfg.idle_timeout_s * 1000),
                    max_records=cfg.max_messages - len(rows),
                )
                if not polled:
                    break
                for records in polled.values():
                    for record in records:
                        rows.extend(_rows(record.value, cfg.records_path))
            consumer.commit()
        finally:
            consumer.close()
        return Batch(
            rows=rows[: cfg.max_messages],
            state={**state, "messages": int(state.get("messages", 0)) + len(rows)},
        )


# ------------------------------------------------------------------ MQTT


class MqttConfig(BaseModel):
    host: str = Field(min_length=1)
    port: int = Field(default=1883, ge=1, le=65535)
    topic: str = Field(min_length=1, description="Admite comodines + y #")
    qos: Literal[0, 1, 2] = 1
    client_id: str = Field(default="perceptron", min_length=1, max_length=64)
    username: str | None = None
    tls: bool = False
    records_path: str | None = None
    max_messages: int = Field(default=1000, ge=1, le=MAX_MESSAGES)
    idle_timeout_s: float = Field(default=5.0, gt=0, le=300)


MqttFactory = Callable[[str], Any]


def default_mqtt(client_id: str) -> Any:
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        raise ValidationError("falta el extra `streaming` (paho-mqtt)") from None
    return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=False)


class MqttSource:
    def __init__(
        self, config: MqttConfig, factory: MqttFactory | None = None, net: NetPolicy | None = None
    ) -> None:
        self.config = config
        self.factory = factory or default_mqtt
        self.net = net or NetPolicy()

    def fetch(self, state: dict[str, Any], secret: str | None) -> Any:
        from perceptron.data.sources.stream import Batch

        cfg = self.config
        check_host(cfg.host, cfg.port, self.net)
        client = self.factory(cfg.client_id)
        rows: list[dict[str, Any]] = []
        lock = threading.Lock()
        last = {"t": time.monotonic()}
        connected = threading.Event()

        def on_connect(c: Any, *_: Any) -> None:
            c.subscribe(cfg.topic, qos=cfg.qos)
            connected.set()

        def on_message(_c: Any, _u: Any, message: Any) -> None:
            with lock:
                rows.extend(_rows(message.payload, cfg.records_path))
                last["t"] = time.monotonic()

        client.on_connect = on_connect
        client.on_message = on_message
        if cfg.username:
            client.username_pw_set(cfg.username, secret)
        if cfg.tls:
            client.tls_set()
        client.connect(cfg.host, cfg.port, keepalive=30)
        client.loop_start()
        try:
            if not connected.wait(timeout=15):
                raise ValidationError("no se pudo conectar al broker MQTT")
            last["t"] = time.monotonic()
            while True:
                with lock:
                    enough = len(rows) >= cfg.max_messages
                    idle = time.monotonic() - last["t"] >= cfg.idle_timeout_s
                if enough or idle:
                    break
                time.sleep(0.05)
        finally:
            client.loop_stop()
            client.disconnect()
        with lock:
            taken = rows[: cfg.max_messages]
        return Batch(
            rows=taken,
            state={**state, "messages": int(state.get("messages", 0)) + len(taken)},
        )
