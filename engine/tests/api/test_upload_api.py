"""Subida de archivos y carpetas desde la UI web (`POST /projects/{id}/uploads`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def test_upload_csv_then_ingest(client: TestClient, fixtures_dir: Path) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "subida"}), 201)["id"]
    data = (fixtures_dir / "uc01_churn" / "churn.csv").read_bytes()
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/uploads", files=[("files", ("churn.csv", data, "text/csv"))]
        ),
        201,
    )
    assert src["name"] == "churn.csv" and Path(src["config"]["path"]).is_file()
    preview = _ok(client.post(f"{API}/sources/{src['id']}/preview"))
    assert "churn" in preview["columns"]
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    assert dv["num_samples"] > 0


def test_upload_folder_keeps_structure_and_blocks_traversal(
    client: TestClient, fixtures_dir: Path
) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "imagenes"}), 201)["id"]
    images = sorted((fixtures_dir / "uc04_defects").rglob("*.png"))[:6]
    files = [
        ("files", (f"uc04_defects/{p.parent.name}/{p.name}", p.read_bytes(), "image/png"))
        for p in images
    ]
    files.append(("files", ("../../fuera.png", images[0].read_bytes(), "image/png")))
    src = _ok(client.post(f"{API}/projects/{pid}/uploads", files=files), 201)
    root = Path(src["config"]["path"])  # varias raíces: la carpeta de la subida
    assert src["config"]["files"] == 7
    assert (root / "uc04_defects" / images[0].parent.name / images[0].name).is_file()
    assert (root / "fuera.png").is_file()  # `..` se descarta: queda dentro de la subida
    assert not (root.parent.parent / "fuera.png").exists()
    only_folder = files[:-1]
    src2 = _ok(client.post(f"{API}/projects/{pid}/uploads", files=only_folder), 201)
    assert Path(src2["config"]["path"]).name == "uc04_defects"
    # Vista previa con clases y muestra, sin leer las imágenes.
    prev = _ok(client.post(f"{API}/sources/{src2['id']}/preview"))
    assert prev["kind"] == "image_folder" and prev["total"] == 6
    assert sum(prev["classes"].values()) == 6 and prev["rows"][0]["label"] in prev["classes"]


def test_upload_folder_with_root_file_uses_chosen_folder(
    client: TestClient, fixtures_dir: Path
) -> None:
    """UC-04 desde el navegador: subcarpetas por clase + `annotations_coco.json` en la raíz."""
    pid = _ok(client.post(f"{API}/projects", json={"name": "carpeta"}), 201)["id"]
    base = fixtures_dir / "uc04_defects"
    images = sorted(base.rglob("*.png"))[:4]
    files = [
        ("files", (f"uc04_defects/{p.parent.name}/{p.name}", p.read_bytes(), "image/png"))
        for p in images
    ]
    coco = (base / "annotations_coco.json").read_bytes()
    files.append(("files", ("uc04_defects/annotations_coco.json", coco, "application/json")))
    src = _ok(client.post(f"{API}/projects/{pid}/uploads", files=files), 201)
    root = Path(src["config"]["path"])
    assert root.name == "uc04_defects" and src["name"] == "uc04_defects"
    assert (root / "annotations_coco.json").is_file()
