"""Manifiestos content-addressed de datasets (RF-ING-07, RF-MON-07).

Un manifiesto lista cada archivo (ruta relativa POSIX, tamaño, sha256). Su hash
(sha256 del JSON canónico) identifica la `DatasetVersion`: mismos bytes → mismo
hash, independientemente del SO o del orden de recorrido.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class ManifestEntry:
    path: str  # relativa, separador "/"
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Manifest:
    entries: tuple[ManifestEntry, ...]

    @property
    def total_size(self) -> int:
        return sum(e.size for e in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "entries": [{"path": e.path, "size": e.size, "sha256": e.sha256} for e in self.entries],
        }

    def canonical_json(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        if data.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(f"manifest_version no soportada: {data.get('manifest_version')}")
        return cls(tuple(sorted(ManifestEntry(**e) for e in data["entries"])))

    def diff(self, other: Manifest) -> dict[str, list[str]]:
        """Archivos agregados / eliminados / modificados de `self` a `other`."""
        a = {e.path: e.sha256 for e in self.entries}
        b = {e.path: e.sha256 for e in other.entries}
        return {
            "added": sorted(b.keys() - a.keys()),
            "removed": sorted(a.keys() - b.keys()),
            "changed": sorted(p for p in a.keys() & b.keys() if a[p] != b[p]),
        }


def build_manifest(root: Path, files: Iterable[Path] | None = None) -> Manifest:
    """Construye el manifiesto de `root` (todos los archivos, recursivo) o de `files`."""
    root = root.resolve()
    candidates = files if files is not None else (p for p in root.rglob("*") if p.is_file())
    entries = []
    for candidate in candidates:
        path = candidate.resolve()
        entries.append(
            ManifestEntry(
                path=path.relative_to(root).as_posix(),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return Manifest(tuple(sorted(entries)))
