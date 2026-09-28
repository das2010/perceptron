"""Guardas del proceso sandbox (SPEC §13.2; ADR-0025).

Se instalan dentro del proceso que va a ejecutar código experto, **después** de importar
torch/Lightning y **antes** de cargar el código del usuario:

- *audit hook* (no removible) que corta red, procesos, `ctypes` fuera del runtime y E/S de
  archivos fuera de las rutas permitidas (escritura: el run; lectura: run, dataset, runtime);
- `socket.socket` reemplazado por una versión que falla;
- límites del SO sobre el propio proceso: `setrlimit` en Linux, Job Object en Windows.

No es una frontera de seguridad absoluta (los audit hooks no lo son): es defensa en
profundidad para código accidental o inyectado; en el servidor se suma un contenedor.
"""

from __future__ import annotations

import os
import site
import socket
import sys
import sysconfig
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SANDBOX_ENV = "PERCEPTRON_SANDBOX"

_BLOCKED_EVENTS = frozenset(
    {
        "_winapi.CreateProcess",
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "os.startfile",
        "os.system",
        "pty.spawn",
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.sendto",
        "subprocess.Popen",
        "webbrowser.open",
    }
)
_BLOCKED_IMPORTS = frozenset(
    {
        "ftplib",
        "http.client",
        "httpx",
        "multiprocessing",
        "pty",
        "requests",
        "smtplib",
        "socketserver",
        "subprocess",
        "telnetlib",
        "urllib.request",
        "urllib3",
        "webbrowser",
    }
)
_PATH_EVENTS = frozenset(
    {
        "os.chmod",
        "os.listdir",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.scandir",
        "os.symlink",
        "os.truncate",
        "os.utime",
        "shutil.copyfile",
        "shutil.rmtree",
    }
)
_READ_ONLY_EVENTS = frozenset({"os.listdir", "os.scandir"})
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


class SandboxViolationError(PermissionError):
    """Operación prohibida dentro del sandbox."""


@dataclass
class Policy:
    write_roots: list[Path]
    read_roots: list[Path] = field(default_factory=list)
    memory_mb: int | None = None
    cpu_seconds: int | None = None
    max_file_mb: int = 4_096
    violations: list[str] = field(default_factory=list)


def _norm(p: str | os.PathLike[str] | bytes) -> str:
    # realpath: en Windows unifica nombres cortos (8.3) y largos; en Linux resuelve symlinks.
    return os.path.normcase(os.path.realpath(os.fsdecode(p)))


def _under(path: str, roots: Iterable[str]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:  # distinta unidad en Windows
            continue
    return False


def runtime_roots() -> list[Path]:
    """Rutas del intérprete y de lo importable: se pueden leer (torch carga módulos y datos)."""
    paths = {
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        *site.getsitepackages(),
        *(
            p
            for k in ("stdlib", "platstdlib", "purelib", "platlib")
            if (p := sysconfig.get_path(k))
        ),
    }
    # Todo lo importable (incluye instalaciones editables, p. ej. el Engine en desarrollo/CI).
    paths |= {p for p in sys.path if p and Path(p).is_dir()}
    roots = [Path(p) for p in paths if p]
    if sys.platform.startswith("linux"):
        # Información del sistema que consultan torch y el runtime (CPU, memoria, zona horaria).
        roots += [Path("/proc"), Path("/sys"), Path("/dev"), Path("/usr/share/zoneinfo")]
        roots += [Path("/etc/localtime"), Path("/usr/lib"), Path("/lib"), Path("/lib64")]
    return roots


def _mode_writes(mode: Any, flags: Any) -> bool:
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        return True
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _make_hook(policy: Policy) -> Any:
    writes = [_norm(p) for p in policy.write_roots]
    reads = writes + [_norm(p) for p in policy.read_roots]

    def deny(what: str) -> None:
        policy.violations.append(what)
        raise SandboxViolationError(f"sandbox: operación no permitida ({what})")

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event in _BLOCKED_EVENTS:
            deny(event)
        elif event == "os.kill":
            if not args or args[0] != os.getpid():
                deny("os.kill a otro proceso")
        elif event == "import":
            name = args[0] if args else ""
            if isinstance(name, str) and (
                name in _BLOCKED_IMPORTS or name.split(".")[0] in {"ctypes"}
            ):
                deny(f"import {name}")
        elif event == "ctypes.dlopen":
            lib = args[0] if args else None
            if (
                lib is None
                or not Path(os.fsdecode(lib)).is_absolute()
                or not _under(_norm(lib), reads)
            ):
                deny(f"ctypes.dlopen {lib!r}")
        elif event == "open":
            path, mode, flags = [*args, None, None, None][:3]
            if isinstance(path, int) or path is None:
                return  # descriptores ya abiertos
            p = _norm(path)
            if _mode_writes(mode, flags):
                if not _under(p, writes):
                    deny(f"escritura en {p}")
            elif not _under(p, reads):
                deny(f"lectura de {p}")
        elif event in _PATH_EVENTS:
            allowed = reads if event in _READ_ONLY_EVENTS else writes
            # Origen y destino (rename, symlink, copyfile): todas las rutas deben estar permitidas.
            for a in args:
                if isinstance(a, (str, bytes, os.PathLike)):
                    p = _norm(a)
                    if not _under(p, allowed):
                        deny(f"{event} {p}")

    return hook


class _NoSocket(socket.socket):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise SandboxViolationError("sandbox: sin red")


def _limit_linux(policy: Policy) -> None:
    if sys.platform == "win32":
        return
    import resource

    def cap(which: int, value: int) -> None:
        _, hard = resource.getrlimit(which)
        new = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(which, (new, new))

    if policy.memory_mb:
        cap(resource.RLIMIT_AS, policy.memory_mb * 1024 * 1024)
    if policy.cpu_seconds:
        cap(resource.RLIMIT_CPU, policy.cpu_seconds)
    cap(resource.RLIMIT_FSIZE, policy.max_file_mb * 1024 * 1024)


def _limit_windows(policy: Policy) -> None:
    """Job Object sobre el propio proceso: memoria, CPU, sin hijos y muerte al cerrar."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]  # fmt: skip

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    limit_process_time, limit_active, limit_memory, kill_on_close = 0x2, 0x8, 0x100, 0x2000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Firmas explícitas: sin argtypes, ctypes pasa los HANDLE como int de 32 bits y el
    # pseudo-handle de GetCurrentProcess (-1) desborda.
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW")
    info = ExtendedLimits()
    flags = limit_active | kill_on_close
    info.BasicLimitInformation.ActiveProcessLimit = 1
    if policy.cpu_seconds:
        flags |= limit_process_time
        info.BasicLimitInformation.PerProcessUserTimeLimit = policy.cpu_seconds * 10_000_000
    if policy.memory_mb:
        flags |= limit_memory
        info.ProcessMemoryLimit = policy.memory_mb * 1024 * 1024
    info.BasicLimitInformation.LimitFlags = flags
    extended_limit_information = 9
    if not kernel32.SetInformationJobObject(
        job, extended_limit_information, ctypes.byref(info), ctypes.sizeof(info)
    ):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
    # El handle del job queda abierto hasta que termina el proceso (kill on close).


def install(policy: Policy) -> None:
    """Aplica límites del SO, corta sockets e instala el audit hook (irreversible)."""
    if sys.platform == "win32":
        _limit_windows(policy)
    elif sys.platform.startswith("linux"):
        _limit_linux(policy)
    socket.socket = _NoSocket  # type: ignore[misc]
    sys.dont_write_bytecode = True  # no escribir .pyc fuera del run
    sys.addaudithook(_make_hook(policy))
    os.environ[SANDBOX_ENV] = "1"
