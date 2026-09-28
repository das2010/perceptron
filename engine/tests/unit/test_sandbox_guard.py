"""Guardas del proceso sandbox (SPEC §13.2, ADR-0025).

Las guardas son irreversibles, así que cada caso corre en un subproceso aislado con el mismo
entorno mínimo que usa el Engine.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from perceptron.sandbox.process import python_args, sandbox_env

PROBE = r"""
import json, sys
from pathlib import Path
work = Path(sys.argv[1])
outside = Path(sys.argv[2])
from perceptron.sandbox.guard import Policy, install, runtime_roots
policy = Policy(write_roots=[work], read_roots=runtime_roots(), memory_mb=4096, cpu_seconds=60)
install(policy)
results = {}

def attempt(name, fn):
    try:
        fn()
        results[name] = "ok"
    except PermissionError as e:
        results[name] = "blocked"
    except Exception as e:
        results[name] = f"error: {type(e).__name__}: {e}"

attempt("write_inside", lambda: (work / "ok.txt").write_text("x"))
attempt("read_inside", lambda: (work / "ok.txt").read_text())
attempt("write_outside", lambda: (outside / "fuga.txt").write_text("x"))
attempt("read_outside", lambda: (outside / "secreto.txt").read_text())
attempt("list_outside", lambda: list(outside.iterdir()))
attempt("remove_outside", lambda: (outside / "secreto.txt").unlink())

def net():
    import socket
    socket.create_connection(("127.0.0.1", 9), timeout=1)
attempt("socket", net)

def proc():
    import subprocess
    subprocess.run([sys.executable, "-c", "print(1)"])
attempt("subprocess", proc)

def system():
    import os
    os.system("echo hola")
attempt("os_system", system)

def stdlib_import():
    import decimal  # lectura del runtime: permitida
    return decimal.Decimal(1)
attempt("stdlib_import", stdlib_import)
print(json.dumps({"results": results, "violations": len(policy.violations)}))
"""


@pytest.fixture
def probe(tmp_path: Path) -> dict[str, object]:
    work = tmp_path / "run con espacio ñ"
    work.mkdir()
    outside = tmp_path / "afuera"
    outside.mkdir()
    (outside / "secreto.txt").write_text("no", encoding="utf-8")
    script = work / "probe.py"
    script.write_text(PROBE, encoding="utf-8")
    args = python_args("perceptron.sandbox.static")  # intérprete + flags del sandbox
    cmd = [*args[:-2], str(script), str(work), str(outside)]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=work,
        env=sandbox_env(work),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out: dict[str, object] = json.loads(proc.stdout.strip().splitlines()[-1])
    assert (outside / "secreto.txt").exists()
    assert not (outside / "fuga.txt").exists()
    return out


def test_guards_block_io_network_and_processes(probe: dict[str, object]) -> None:
    results = probe["results"]
    assert isinstance(results, dict)
    assert results["write_inside"] == "ok" and results["read_inside"] == "ok"
    assert results["stdlib_import"] == "ok"
    for name in (
        "write_outside",
        "read_outside",
        "list_outside",
        "remove_outside",
        "socket",
        "subprocess",
        "os_system",
    ):
        assert results[name] == "blocked", (name, results[name])
    assert isinstance(probe["violations"], int) and probe["violations"] >= 7


def test_sandbox_env_is_minimal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secreto")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    env = sandbox_env(tmp_path)
    assert "ANTHROPIC_API_KEY" not in env and "HTTPS_PROXY" not in env
    assert env["TMP"].startswith(str(tmp_path)) and env["PERCEPTRON_OFFLINE"] == "1"
    assert python_args("x")[:5] == [sys.executable, "-I", "-u", "-X", "utf8"]
