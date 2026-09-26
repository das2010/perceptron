from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from perceptron.core.errors import ConflictError, NotFoundError, ValidationError
from perceptron.core.paths import WorkspacePaths
from perceptron.domain.enums import DataSourceType, ProjectStatus
from perceptron.domain.models import DataSource, Project
from perceptron.storage.db import Database
from perceptron.storage.filesystem import ProjectFiles, atomic_write_text
from perceptron.storage.manifest import Manifest, build_manifest
from perceptron.storage.repositories import SqlRepository


@pytest.fixture
def db(workspace_dir: Path) -> Iterator[Database]:
    d = Database.for_file(workspace_dir / "perceptron.db")
    d.create_all()
    yield d
    d.dispose()


def test_crud_and_optimistic_locking(db: Database) -> None:
    repo = SqlRepository(db, Project)
    p = repo.add(Project(name="Churn ñandú"))
    assert repo.get(p.id) == p

    with pytest.raises(ConflictError):
        repo.add(p)

    updated = repo.update(p.model_copy(update={"status": ProjectStatus.ACTIVE}))
    assert updated.version == 2
    assert repo.get(p.id).status is ProjectStatus.ACTIVE

    # Cliente con versión vieja → conflicto
    with pytest.raises(ConflictError):
        repo.update(p.model_copy(update={"name": "otro"}))

    repo.delete(p.id)
    assert repo.find(p.id) is None
    with pytest.raises(NotFoundError):
        repo.get(p.id)
    with pytest.raises(NotFoundError):
        repo.delete(p.id)


def test_list_filters(db: Database) -> None:
    projects = SqlRepository(db, Project)
    sources = SqlRepository(db, DataSource)
    a = projects.add(Project(name="A"))
    b = projects.add(Project(name="B", status=ProjectStatus.ACTIVE))
    sources.add(DataSource(project_id=a.id, name="csv", type=DataSourceType.FILE))
    sources.add(DataSource(project_id=b.id, name="db", type=DataSourceType.DB))

    assert {p.name for p in projects.list()} == {"A", "B"}
    assert [p.name for p in projects.list(filters={"status": "active"})] == ["B"]
    assert [s.name for s in sources.list(filters={"project_id": a.id})] == ["csv"]
    assert len(projects.list(limit=1)) == 1
    with pytest.raises(ValidationError):
        projects.list(filters={"nope": 1})


def test_kinds_are_isolated(db: Database) -> None:
    p = SqlRepository(db, Project).add(Project(name="A"))
    assert SqlRepository(db, DataSource).find(p.id) is None


def test_manifest_hash_is_content_addressed(tmp_path: Path) -> None:
    root = tmp_path / "datos con espacio"
    (root / "sub dir").mkdir(parents=True)
    (root / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (root / "sub dir" / "imagen ñ.png").write_bytes(b"\x89PNG fake")

    m1 = build_manifest(root)
    assert [e.path for e in m1.entries] == ["a.csv", "sub dir/imagen ñ.png"]
    assert len(m1.content_hash) == 64
    assert m1.total_size == 8 + 9

    # Copia idéntica en otra ruta → mismo hash
    other = tmp_path / "copia"
    (other / "sub dir").mkdir(parents=True)
    (other / "a.csv").write_bytes((root / "a.csv").read_bytes())
    (other / "sub dir" / "imagen ñ.png").write_bytes(
        (root / "sub dir" / "imagen ñ.png").read_bytes()
    )
    assert build_manifest(other).content_hash == m1.content_hash

    # Modificar un archivo cambia el hash y aparece en el diff
    (other / "a.csv").write_text("x,y\n1,3\n", encoding="utf-8")
    (other / "nuevo.txt").write_text("n", encoding="utf-8")
    m2 = build_manifest(other)
    assert m2.content_hash != m1.content_hash
    assert m1.diff(m2) == {"added": ["nuevo.txt"], "removed": [], "changed": ["a.csv"]}

    assert Manifest.from_dict(m1.to_dict()).content_hash == m1.content_hash


def test_project_files(workspace_dir: Path) -> None:
    files = ProjectFiles(WorkspacePaths(workspace_dir))
    p = Project(name="Inspección visual")
    paths = files.init_project(p)
    assert paths.datasets_dir.is_dir()
    assert files.read_project(p.id) == p
    assert "Inspección" in paths.project_file.read_text(encoding="utf-8")


def test_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    target = tmp_path / "x" / "f.json"
    atomic_write_text(target, "{}")
    atomic_write_text(target, '{"a": 1}')
    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    assert [p.name for p in target.parent.iterdir()] == ["f.json"]
