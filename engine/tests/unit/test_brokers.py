"""Fuentes streaming Kafka y MQTT (RF-ING-05) con clientes falsos."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from perceptron.core.errors import ValidationError
from perceptron.core.netguard import NetPolicy
from perceptron.data.sources.brokers import KafkaConfig, KafkaSource, MqttConfig, MqttSource
from perceptron.data.sources.stream import build_source


class FakeConsumer:
    def __init__(self, messages: list[Any], **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.batches = [messages[:2], messages[2:]] if messages else []
        self.committed = self.closed = False

    def poll(self, timeout_ms: int, max_records: int) -> dict[str, list[Any]]:
        if not self.batches:
            return {}
        batch = self.batches.pop(0)[:max_records]
        return {"tp0": [SimpleNamespace(value=v) for v in batch]}

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True


def test_kafka_reads_json_commits_and_closes() -> None:
    messages = [json.dumps({"x": i, "meta": {"ok": True}}).encode() for i in range(3)]
    messages.append(b"no es json")
    made: list[FakeConsumer] = []

    def factory(**kwargs: Any) -> FakeConsumer:
        made.append(FakeConsumer(messages, **kwargs))
        return made[-1]

    cfg = KafkaConfig(
        bootstrap_servers=["broker.test:9092"], topic="sensores", auth="sasl_plain", username="svc"
    )
    batch = KafkaSource(cfg, factory).fetch({"messages": 10}, "clave")
    assert [r["x"] for r in batch.rows] == [0, 1, 2] and batch.rows[0]["meta.ok"] is True
    assert batch.state["messages"] == 13
    consumer = made[0]
    assert consumer.committed and consumer.closed
    assert consumer.kwargs["group_id"] == "perceptron" and not consumer.kwargs["enable_auto_commit"]
    assert consumer.kwargs["security_protocol"] == "SASL_PLAINTEXT"
    assert consumer.kwargs["sasl_plain_password"] == "clave"
    with pytest.raises(ValidationError, match="usuario y contraseña"):
        KafkaSource(cfg, factory).fetch({}, None)


def test_kafka_hosts_go_through_the_network_policy() -> None:
    cfg = KafkaConfig(bootstrap_servers=["10.0.0.5:9092"], topic="t")
    with pytest.raises(ValidationError, match="interna"):
        KafkaSource(cfg, lambda **_: FakeConsumer([]), NetPolicy(allow_private=False)).fetch(
            {}, None
        )
    with pytest.raises(ValidationError, match="host:puerto"):
        KafkaSource(KafkaConfig(bootstrap_servers=["sin-puerto"], topic="t")).fetch({}, None)


class FakeMqtt:
    def __init__(self, client_id: str, payloads: list[bytes]) -> None:
        self.client_id = client_id
        self.payloads = payloads
        self.subscribed: list[tuple[str, int]] = []
        self.on_connect: Any = None
        self.on_message: Any = None
        self.auth: tuple[str, str | None] | None = None
        self.stopped = False

    def username_pw_set(self, user: str, password: str | None) -> None:
        self.auth = (user, password)

    def subscribe(self, topic: str, qos: int) -> None:
        self.subscribed.append((topic, qos))

    def connect(self, host: str, port: int, keepalive: int) -> None:
        self.host = (host, port)

    def loop_start(self) -> None:
        def run() -> None:
            self.on_connect(self, None, None, 0, None)
            for p in self.payloads:
                self.on_message(self, None, SimpleNamespace(payload=p))

        threading.Thread(target=run, daemon=True).start()

    def loop_stop(self) -> None:
        self.stopped = True

    def disconnect(self) -> None:
        pass


def test_mqtt_subscribes_and_collects_until_idle() -> None:
    made: list[FakeMqtt] = []

    def factory(client_id: str) -> FakeMqtt:
        made.append(FakeMqtt(client_id, [json.dumps({"t": i}).encode() for i in range(5)]))
        return made[-1]

    cfg = MqttConfig(
        host="broker.test",
        topic="planta/+/temp",
        username="svc",
        max_messages=3,
        idle_timeout_s=0.3,
    )
    batch = MqttSource(cfg, factory).fetch({}, "clave")
    assert [r["t"] for r in batch.rows] == [0, 1, 2]  # corta en max_messages
    client = made[0]
    assert client.subscribed == [("planta/+/temp", 1)] and client.auth == ("svc", "clave")
    assert client.stopped and client.client_id == "perceptron"


def test_build_source_knows_brokers() -> None:
    assert isinstance(
        build_source("kafka", {"bootstrap_servers": ["b:9092"], "topic": "t"}), KafkaSource
    )
    assert isinstance(build_source("mqtt", {"host": "b", "topic": "t"}), MqttSource)
