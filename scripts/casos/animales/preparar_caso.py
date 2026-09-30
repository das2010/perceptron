"""Arma el caso de uso «Identificación de animales en cámaras del campo» (UC-11) a partir del
zip de Roboflow «Dataset Of animal Images» (CC BY 4.0).

Solo biblioteca estándar y determinístico (semilla fija). Deja cuatro conjuntos disjuntos,
sin copias exactas (se deduplica por contenido):

    entrenamiento/<Especie>/*.jpg   250 fotos por especie (8 especies): el dataset del proyecto
    sin_etiquetar/foto_NNNN.jpg     240 fotos nuevas sin etiqueta (etiquetado asistido)
    produccion_dia/foto_NNNN.jpg    200 fotos de día que llegan al modelo desplegado
    produccion_noche/foto_NNNN.jpg  120 fotos con visión nocturna (cambio de condiciones)
    respuestas/*.csv                especie real de cada foto sin etiqueta y de producción,
                                    para medir el etiquetado y enviar feedback (no se entrenan)

Uso:
    python scripts/casos/animales/preparar_caso.py "Dataset Of animal Images.zip" \
        fixtures/caso_animales
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
import zipfile
from pathlib import Path

SEED = 20260929
SPECIES = ["Cat", "Cow", "Deer", "Dog", "Goat", "Hen", "Rabbit", "Sheep"]
NIGHT = "NightVision"
PER_CLASS = {"entrenamiento": 250, "sin_etiquetar": 30, "produccion_dia": 25}
NIGHT_COUNT = 120


def images_by_class(zf: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    """Imágenes por carpeta de primer nivel (`<raíz>/<Clase>/train/images/*.jpg`)."""
    out: dict[str, list[zipfile.ZipInfo]] = {}
    for info in zf.infolist():
        parts = info.filename.split("/")
        if len(parts) >= 5 and parts[3] == "images" and info.filename.lower().endswith(".jpg"):
            out.setdefault(parts[1], []).append(info)
    return {k: sorted(v, key=lambda i: i.filename) for k, v in out.items()}


def unique(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo], seen: set[str]) -> list[bytes]:
    """Contenido de las imágenes sin repetir (el zip trae algunas copias exactas)."""
    out = []
    for info in infos:
        data = zf.read(info)
        digest = hashlib.sha256(data).hexdigest()
        if digest not in seen:
            seen.add(digest)
            out.append(data)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip", type=Path, help="Dataset Of animal Images.zip")
    parser.add_argument("destino", type=Path, help="carpeta del caso (se reemplaza)")
    args = parser.parse_args(argv)

    rng = random.Random(SEED)  # noqa: S311 - selección reproducible, no criptografía
    seen: set[str] = set()
    with zipfile.ZipFile(args.zip) as zf:
        by_class = images_by_class(zf)
        missing = [c for c in [*SPECIES, NIGHT] if c not in by_class]
        if missing:
            print(f"faltan carpetas en el zip: {missing}", file=sys.stderr)
            return 1
        pools: dict[str, list[bytes]] = {}
        for cls in [*SPECIES, NIGHT]:
            data = unique(zf, by_class[cls], seen)
            rng.shuffle(data)
            pools[cls] = data

    dest = args.destino
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "respuestas").mkdir(parents=True)

    flat: dict[str, list[tuple[str, bytes]]] = {"sin_etiquetar": [], "produccion_dia": []}
    for cls in SPECIES:
        pool = pools[cls]
        need = sum(PER_CLASS.values())
        if len(pool) < need:
            print(f"{cls}: hay {len(pool)} fotos y se necesitan {need}", file=sys.stderr)
            return 1
        start = 0
        for name, n in PER_CLASS.items():
            chunk = pool[start : start + n]
            start += n
            if name == "entrenamiento":
                folder = dest / name / cls
                folder.mkdir(parents=True)
                for i, data in enumerate(chunk):
                    (folder / f"{cls.lower()}_{i:03d}.jpg").write_bytes(data)
            else:
                flat[name] += [(cls, d) for d in chunk]

    # Las fotos sin etiqueta y las de producción llegan mezcladas y con nombres neutros.
    for name, items in flat.items():
        rng.shuffle(items)
        folder = dest / name
        folder.mkdir()
        with (dest / "respuestas" / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file_name", "especie"])
            for i, (cls, data) in enumerate(items):
                file_name = f"foto_{i:04d}.jpg"
                (folder / file_name).write_bytes(data)
                writer.writerow([file_name, cls])

    night = dest / "produccion_noche"
    night.mkdir()
    for i, data in enumerate(pools[NIGHT][:NIGHT_COUNT]):
        (night / f"foto_{i:04d}.jpg").write_bytes(data)

    (dest / "FUENTE.txt").write_text(
        "Imágenes: «Dataset Of animal Images» (exportado de Roboflow Universe, "
        "licencia CC BY 4.0).\nProyectos de origen: animal (dataset-0yd6z), sheep "
        "(sheep-h8rkr), nnightvision (sheep-h8rkr)\ny los demás indicados en cada data.yaml "
        "del zip original.\n",
        encoding="utf-8",
    )
    total = sum(1 for p in dest.rglob("*.jpg"))
    print(f"caso armado en {dest}: {total} imágenes, {len(seen)} únicas leídas del zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
