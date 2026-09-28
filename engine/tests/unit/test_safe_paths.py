"""Rutas recibidas de clientes (subidas, sincronización): seguras en Windows y Linux (Capa 7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from perceptron.core.errors import ValidationError
from perceptron.core.paths import WorkspacePaths, ensure_within, safe_parts


@pytest.mark.parametrize(
    ("name", "parts"),
    [
        ("a/b.png", ("a", "b.png")),
        ("uc04 defects/ñandú.png", ("uc04 defects", "ñandú.png")),
        ("carpeta\\sub\\x.csv", ("carpeta", "sub", "x.csv")),
        ("./a//b", ("a", "b")),
    ],
)
def test_accepts_relative_paths_with_spaces_and_accents(name: str, parts: tuple[str, ...]) -> None:
    assert safe_parts(name) == parts


@pytest.mark.parametrize(
    "name",
    [
        "C:evil",  # letra de unidad: Path(root, "C:evil") sale de root en Windows
        "datos/C:\\Windows\\x",
        "a.csv:flujo",  # flujo alternativo de NTFS
        "con.txt",
        "datos/NUL",
        "lpt1",
        "archivo.",
        "archivo ",
        "a\x00b",
        "/etc/passwd",
        "../x",
        "a/../../x",
        "",
        "/",
    ],
)
def test_rejects_escapes_and_windows_specials(name: str) -> None:
    with pytest.raises(ValidationError):
        safe_parts(name)


def test_drop_parent_anchors_inside_root() -> None:
    assert safe_parts("../../fuera.png", drop_parent=True) == ("fuera.png",)
    assert safe_parts("/abs/x.png", drop_parent=True) == ("abs", "x.png")
    with pytest.raises(ValidationError):
        safe_parts("../C:evil", drop_parent=True)


def test_ensure_within_follows_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "proyecto"
    outside = tmp_path / "fuera"
    root.mkdir()
    outside.mkdir()
    assert ensure_within(root / "a" / "b.csv", root) == (root / "a" / "b.csv").resolve()
    link = root / "enlace"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("el sistema no permite crear symlinks")
    with pytest.raises(ValidationError):
        ensure_within(link / "x.csv", root)


def test_project_ids_never_build_other_paths(tmp_path: Path) -> None:
    paths = WorkspacePaths(tmp_path)
    assert paths.project("prj_01ABC").root == tmp_path / "projects" / "prj_01ABC"
    for bad in ("..", "a/b", "..\\x", "C:evil", ""):
        with pytest.raises(ValidationError):
            paths.project(bad)


def test_zip_limits_reject_bombs_and_huge_archives(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import zipfile

    from perceptron.data.sources import files

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(5):
            z.writestr(f"clase/{i}.txt", "0" * 100_000)
    with zipfile.ZipFile(buf) as z:
        files.check_zip_limits(z)  # normal
        with pytest.raises(ValidationError, match="demasiados"):
            files.check_zip_limits(z, max_files=4)
        with pytest.raises(ValidationError, match="supera"):
            files.check_zip_limits(z, max_bytes=400_000)
        monkeypatch.setattr(files, "ZIP_RATIO_MIN_BYTES", 1_000)
        monkeypatch.setattr(files, "ZIP_MAX_RATIO", 10)
        with pytest.raises(ValidationError, match="zip bomb"):
            files.check_zip_limits(z)
