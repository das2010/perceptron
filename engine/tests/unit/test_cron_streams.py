"""Cron de los disparadores y fuentes streaming/API (RF-MON-05, RF-ING-05)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from perceptron.core.errors import ValidationError
from perceptron.data.sources import stream
from perceptron.data.sources.stream import (
    FileConfig,
    FileTailSource,
    RestConfig,
    RestSource,
    StreamBuffer,
    dig,
)
from perceptron.monitoring.cron import Cron


def test_cron_matching_and_due() -> None:
    every15 = Cron("*/15 * * * *")
    assert every15.matches(datetime(2026, 9, 28, 10, 30, tzinfo=UTC))
    assert not every15.matches(datetime(2026, 9, 28, 10, 31, tzinfo=UTC))
    monday3 = Cron("0 3 * * 1")
    assert monday3.matches(datetime(2026, 9, 28, 3, 0, tzinfo=UTC))  # lunes
    assert not monday3.matches(datetime(2026, 9, 29, 3, 0, tzinfo=UTC))
    assert Cron("0 0 * * 7").matches(datetime(2026, 9, 27, 0, 0, tzinfo=UTC))  # domingo = 7
    last = datetime(2026, 9, 28, 2, 50, tzinfo=UTC)
    assert monday3.due(last, datetime(2026, 9, 28, 3, 5, tzinfo=UTC))
    assert not monday3.due(
        datetime(2026, 9, 28, 3, 0, tzinfo=UTC), datetime(2026, 9, 28, 9, 0, tzinfo=UTC)
    )
    for bad in ("* * *", "61 * * * *", "*/0 * * * *", "5-2 * * * *"):
        with pytest.raises(ValidationError):
            Cron(bad)


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_rest_offset_pagination_resumes_from_state() -> None:
    records = [{"id": i, "cliente": {"plan": "básico" if i % 2 else "premium"}} for i in range(250)]
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.headers.get("authorization", ""))
        off, lim = int(req.url.params["offset"]), int(req.url.params["limit"])
        return httpx.Response(200, json={"data": {"items": records[off : off + lim]}})

    cfg = RestConfig(
        url="https://api.example.test/clientes",
        auth="bearer",
        records_path="data.items",
        pagination="offset",
        page_size=100,
    )
    src = RestSource(cfg, lambda: _client(handler))
    batch = src.fetch({}, "tok")
    assert len(batch.rows) == 250 and batch.state == {"offset": 250} and batch.exhausted
    assert batch.rows[1]["cliente.plan"] == "básico"  # JSON anidado → columnas
    assert set(seen) == {"Bearer tok"}
    records.extend({"id": i, "cliente": {"plan": "x"}} for i in range(250, 260))
    again = src.fetch(batch.state, "tok")
    assert [r["id"] for r in again.rows] == list(range(250, 260))
    with pytest.raises(ValidationError, match="token"):
        src.fetch({}, None)


def test_rest_cursor_and_link_pagination() -> None:
    pages = {None: ([1, 2], "c2"), "c2": ([3], None)}

    def by_cursor(req: httpx.Request) -> httpx.Response:
        items, nxt = pages[req.url.params.get("cursor")]
        return httpx.Response(200, json={"items": items, "next": nxt})

    cursor = RestSource(
        RestConfig(
            url="https://a.test/x", records_path="items", pagination="cursor", cursor_path="next"
        ),
        lambda: _client(by_cursor),
    )
    assert [r["value"] for r in cursor.fetch({}, None).rows] == [1, 2, 3]

    def by_link(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/p2":
            return httpx.Response(200, json=[{"n": 2}])
        return httpx.Response(200, json=[{"n": 1}], headers={"Link": '</p2>; rel="next"'})

    link = RestSource(
        RestConfig(url="https://a.test/p1", pagination="link"), lambda: _client(by_link)
    )
    assert [r["n"] for r in link.fetch({}, None).rows] == [1, 2]


def test_build_source_uses_patchable_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stream, "default_http", lambda: _client(lambda req: httpx.Response(200, json=[{"a": 1}]))
    )
    src = stream.build_source("rest", {"url": "https://a.test/x"})
    assert src.fetch({}, None).rows == [{"a": 1}]
    with pytest.raises(ValidationError):
        stream.build_source("amqp", {})


def test_file_tail_and_buffer(tmp_path: Path) -> None:
    path = tmp_path / "stream ñ.jsonl"
    path.write_text(
        json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n" + '{"a": 3', encoding="utf-8"
    )
    src = FileTailSource(FileConfig(path=str(path)))
    first = src.fetch({}, None)
    assert [r["a"] for r in first.rows] == [1, 2]  # la línea incompleta espera
    with path.open("a", encoding="utf-8") as f:
        f.write("}\n")
    second = src.fetch(first.state, None)
    assert [r["a"] for r in second.rows] == [3]

    buf = StreamBuffer(tmp_path / "buffer")
    assert buf.append(first.rows) == 2
    mark = buf.batches()[-1].name
    buf.append(second.rows)
    assert buf.read().height == 3 and buf.read(after=mark)["a"].to_list() == [3]
    assert buf.stats()["rows"] == 3 and buf.stats()["batches"] == 2
    assert dig({"a": [{"b": 5}]}, "a.0.b") == 5
