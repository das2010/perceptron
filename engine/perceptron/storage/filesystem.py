"""Almacenamiento de archivos del proyecto en el workspace (SPEC §6.2).

`project.json` es una copia legible de la metadata del proyecto (la fuente de
verdad es la base de datos); nunca contiene secretos (SPEC §13.2).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from perceptron.core.paths import ProjectPaths, WorkspacePaths
from perceptron.domain.models import Project


def atomic_write_text(path: Path, text: str) -> None:
    """Escritura atómica (tmp + replace) para no dejar archivos a medias."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class ProjectFiles:
    def __init__(self, workspace: WorkspacePaths) -> None:
        self.workspace = workspace

    def paths(self, project_id: str) -> ProjectPaths:
        return self.workspace.project(project_id)

    def init_project(self, project: Project) -> ProjectPaths:
        paths = self.paths(project.id).ensure()
        self.write_project(project)
        return paths

    def write_project(self, project: Project) -> None:
        write_json(self.paths(project.id).project_file, project.model_dump(mode="json"))

    def read_project(self, project_id: str) -> Project:
        return Project.model_validate(read_json(self.paths(project_id).project_file))
