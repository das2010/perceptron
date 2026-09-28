"""Rutas del workspace local (SPEC §6.2).

Todo se resuelve con `pathlib`; las rutas pueden contener espacios y caracteres
no ASCII (p. ej. carpetas de OneDrive corporativo, SPEC §13.5).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_path

APP_NAME = "Perceptron"


def within_roots(path: Path, roots: Sequence[Path] | None) -> bool:
    """`path` (resuelta, sin symlinks ni `..`) cae dentro de alguna raíz; `None` = sin límite."""
    if roots is None:
        return True
    target = path.expanduser().resolve()
    return any(target.is_relative_to(r.expanduser().resolve()) for r in roots)


def default_workspace_dir() -> Path:
    """`%LOCALAPPDATA%\\Perceptron` en Windows, `~/.local/share/perceptron` en Linux."""
    name = APP_NAME if sys.platform == "win32" else APP_NAME.lower()
    return user_data_path(name, appauthor=False)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Layout de `projects/<project_id>/`."""

    root: Path

    @property
    def project_file(self) -> Path:
        return self.root / "project.json"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def pipelines_dir(self) -> Path:
        return self.root / "pipelines"

    @property
    def archspecs_dir(self) -> Path:
        return self.root / "archspecs"

    @property
    def code_dir(self) -> Path:
        return self.root / "code"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    def dataset(self, version_hash: str) -> Path:
        return self.datasets_dir / version_hash

    def run(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def ensure(self) -> ProjectPaths:
        for d in (
            self.root,
            self.datasets_dir,
            self.labels_dir,
            self.pipelines_dir,
            self.archspecs_dir,
            self.code_dir,
            self.runs_dir,
            self.exports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """Layout de un workspace local."""

    root: Path

    @property
    def db_file(self) -> Path:
        return self.root / "perceptron.db"

    @property
    def mlflow_dir(self) -> Path:
        return self.root / "mlflow"

    @property
    def models_cache_dir(self) -> Path:
        return self.root / "cache" / "models"

    @property
    def projects_dir(self) -> Path:
        return self.root / "projects"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def project(self, project_id: str) -> ProjectPaths:
        return ProjectPaths(self.projects_dir / project_id)

    def ensure(self) -> WorkspacePaths:
        for d in (
            self.root,
            self.mlflow_dir,
            self.models_cache_dir,
            self.projects_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self
