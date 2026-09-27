"""Datasets públicos del benchmark O2 (SPEC §15.4): descarga, checksum y preparación.

Se bajan en el job (no se commitean) y se convierten al formato de ingesta de Perceptron:
- Adult (UCI, CC BY 4.0): tabular, ¿ingreso > 50K?
- Fashion-MNIST (MIT): imágenes 28×28, 10 clases (subset).
- Free Spoken Digit Dataset (CC BY-SA 4.0): audio, dígitos hablados (subset). Reemplaza a
  Speech Commands (2,3 GB) para que el job quepa en un runner estándar.

Cada archivo descargado se verifica contra `SHA256` si está fijado; si no, se informa el
hash calculado para fijarlo (trust on first use).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import logging
import random
import struct
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
FASHION_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
FSDD_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/tags/v1.0.10.zip"
SHA256: dict[str, str] = {}  # se completan tras la primera descarga verificada

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
FASHION_CLASSES = [
    "remera",
    "pantalon",
    "buzo",
    "vestido",
    "abrigo",
    "sandalia",
    "camisa",
    "zapatilla",
    "bolso",
    "bota",
]


@dataclass(frozen=True)
class BenchDataset:
    key: str
    description: str
    license: str
    metric: str
    prepare: Callable[[Path], Path]
    reference: dict[str, Any] = field(default_factory=dict)
    epochs: int = 30


def download(url: str, dest: Path) -> Path:
    import httpx

    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
            tmp.replace(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    expected = SHA256.get(dest.name)
    if expected and expected != digest:
        raise ValueError(f"checksum inválido para {dest.name}: {digest} ≠ {expected}")
    if not expected:
        logger.warning("sha256 sin fijar", extra={"file": dest.name, "sha256": digest})
    return dest


# ---------------------------------------------------------------------- adult


def adult_rows(text: str) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(ADULT_COLUMNS) or "?" in parts:
            continue
        row = dict(zip(ADULT_COLUMNS, parts, strict=True))
        row["income"] = "alto" if row["income"].startswith(">50K") else "bajo"
        rows.append(row)
    return rows


def prepare_adult(cache: Path, *, n: int = 6000, seed: int = 0) -> Path:
    raw = download(ADULT_URL, cache / "adult.data")
    rows = adult_rows(raw.read_text(encoding="utf-8", errors="replace"))
    random.Random(seed).shuffle(rows)  # noqa: S311 - submuestreo reproducible
    out = cache / "adult" / "adult.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ADULT_COLUMNS)
        w.writeheader()
        w.writerows(rows[:n])
    return out


# ---------------------------------------------------------------------- fashion-mnist


def read_idx(data: bytes) -> tuple[list[int], bytes]:
    """Formato IDX de MNIST: devuelve (dimensiones, datos crudos)."""
    _zero, dtype, ndim = struct.unpack(">HBB", data[:4])
    if dtype != 0x08:
        raise ValueError("solo IDX de uint8")
    dims = list(struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim]))
    return dims, data[4 + 4 * ndim :]


def prepare_fashion(cache: Path, *, per_class: int = 150) -> Path:
    from PIL import Image

    images = download(f"{FASHION_URL}/train-images-idx3-ubyte.gz", cache / "fashion-images.gz")
    labels = download(f"{FASHION_URL}/train-labels-idx1-ubyte.gz", cache / "fashion-labels.gz")
    (n, h, w), pixels = read_idx(gzip.decompress(images.read_bytes()))
    _, ys = read_idx(gzip.decompress(labels.read_bytes()))
    root = cache / "fashion"
    counts = dict.fromkeys(range(10), 0)
    for i in range(n):
        y = ys[i]
        if counts[y] >= per_class:
            continue
        folder = root / FASHION_CLASSES[y]
        folder.mkdir(parents=True, exist_ok=True)
        img = Image.frombytes("L", (w, h), pixels[i * h * w : (i + 1) * h * w])
        img.save(folder / f"{i:05d}.png")
        counts[y] += 1
        if all(c >= per_class for c in counts.values()):
            break
    return root


# ---------------------------------------------------------------------- fsdd


def organize_fsdd(archive: bytes, root: Path, *, per_class: int = 50) -> Path:
    """`recordings/<dígito>_<hablante>_<n>.wav` → `root/<dígito>/…wav`."""
    counts: dict[str, int] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in sorted(z.namelist()):
            if "/recordings/" not in name or not name.endswith(".wav"):
                continue
            base = name.rsplit("/", 1)[-1]
            digit = base.split("_", 1)[0]
            if counts.get(digit, 0) >= per_class:
                continue
            folder = root / f"digito_{digit}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / base).write_bytes(z.read(name))
            counts[digit] = counts.get(digit, 0) + 1
    return root


def prepare_fsdd(cache: Path, *, per_class: int = 50) -> Path:
    archive = download(FSDD_URL, cache / "fsdd.zip")
    return organize_fsdd(archive.read_bytes(), cache / "fsdd", per_class=per_class)


# Configuraciones manuales de referencia: hiperparámetros fijos elegidos a mano y
# documentados (no SOTA). El agente compite con el mismo presupuesto de épocas.
DATASETS: dict[str, BenchDataset] = {
    "adult": BenchDataset(
        key="adult",
        description="Adult (UCI): ingreso > 50K, 6000 filas",
        license="CC BY 4.0",
        metric="roc_auc",
        prepare=prepare_adult,
        reference={"lr": 1e-3},
        epochs=30,
    ),
    "fashion": BenchDataset(
        key="fashion",
        description="Fashion-MNIST: 10 clases, 150 imágenes por clase",
        license="MIT",
        metric="accuracy",
        prepare=prepare_fashion,
        reference={"lr": 1e-3},
        epochs=15,
    ),
    "fsdd": BenchDataset(
        key="fsdd",
        description="Free Spoken Digit Dataset: 10 dígitos, 50 clips por clase",
        license="CC BY-SA 4.0",
        metric="accuracy",
        prepare=prepare_fsdd,
        reference={"lr": 2e-3},
        epochs=30,
    ),
}
