"""Fuentes de archivos locales (RF-ING-01, parte de RF-ING-09).

Tabulares: CSV, TSV, XLSX, Parquet, JSON y JSONL. Los formatos que Polars puede
escanear (CSV/TSV/Parquet/JSONL) se materializan en streaming con `sink_parquet`,
sin cargar el archivo completo en memoria.
Imágenes: carpeta `clase/archivo` (o ZIP con esa estructura).
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl
from PIL import Image, UnidentifiedImageError

from perceptron.core.errors import ValidationError
from perceptron.data.schema import IMAGE_EXTENSIONS

TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".ndjson"}


class SourceKind(StrEnum):
    TABLE = "table"
    IMAGE_FOLDER = "image_folder"
    TEXT_FOLDER = "text_folder"
    AUDIO_FOLDER = "audio_folder"
    SEGMENTATION_FOLDER = "segmentation_folder"
    OCR_FOLDER = "ocr_folder"


@dataclass(frozen=True, slots=True)
class DetectedSource:
    kind: SourceKind
    path: Path


def _sniff_separator(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        head = f.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _is_utf8(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            while f.read(1024 * 1024):
                pass
    except UnicodeDecodeError:
        return False
    return True


def scan_table(path: Path) -> pl.LazyFrame:
    """LazyFrame para el archivo tabular `path` (streaming cuando el formato lo permite)."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = _sniff_separator(path)
        if _is_utf8(path):
            return pl.scan_csv(
                path, separator=sep, infer_schema_length=10_000, try_parse_dates=True
            )
        # Archivos de Excel exportados en Windows suelen venir en cp1252.
        text = path.read_bytes().decode("cp1252")
        return pl.read_csv(text.encode("utf-8"), separator=sep, try_parse_dates=True).lazy()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pl.scan_ndjson(path, infer_schema_length=10_000)
    if suffix == ".json":
        return pl.read_json(path).lazy()
    if suffix in {".xlsx", ".xls"}:
        return pl.read_excel(path, engine="calamine").lazy()
    raise ValidationError(
        f"formato tabular no soportado: {path.suffix}", details={"path": str(path)}
    )


def write_table_parquet(lf: pl.LazyFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        lf.sink_parquet(target, compression="zstd", statistics=True)
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError):
        # Algunos planes no soportan streaming: se recolecta en memoria.
        lf.collect().write_parquet(target, compression="zstd", statistics=True)


@contextmanager
def open_source(path: Path) -> Iterator[DetectedSource]:
    """Detecta el tipo de fuente. Los ZIP se extraen a un directorio temporal."""
    path = path.resolve()
    if not path.exists():
        raise ValidationError(f"no existe: {path}", details={"path": str(path)})
    if path.is_file() and path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="perceptron-zip-") as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(path) as zf:
                _safe_extract(zf, root)
            entries = [p for p in root.iterdir() if not p.name.startswith(("__MACOSX", "."))]
            inner = entries[0] if len(entries) == 1 else root
            with open_source(inner) as detected:
                yield detected
        return
    if path.is_file():
        if path.suffix.lower() not in TABLE_SUFFIXES:
            raise ValidationError(
                f"formato no soportado: {path.suffix}", details={"path": str(path)}
            )
        yield DetectedSource(SourceKind.TABLE, path)
        return
    from perceptron.data.vision_tasks import is_segmentation_folder, ocr_labels_file

    if is_segmentation_folder(path):
        yield DetectedSource(SourceKind.SEGMENTATION_FOLDER, path)
        return
    if ocr_labels_file(path):
        yield DetectedSource(SourceKind.OCR_FOLDER, path)
        return
    if any(_iter_images(path)):
        yield DetectedSource(SourceKind.IMAGE_FOLDER, path)
        return
    if any(_iter_audio(path)):
        yield DetectedSource(SourceKind.AUDIO_FOLDER, path)
        return
    if any(_iter_texts(path)):
        yield DetectedSource(SourceKind.TEXT_FOLDER, path)
        return
    raise ValidationError(
        "la carpeta no contiene imágenes en formato clase/archivo", details={"path": str(path)}
    )


ZIP_MAX_FILES = 500_000
ZIP_MAX_BYTES = 50 * 1024**3  # descomprimido
ZIP_MAX_RATIO = 1000  # compresión sospechosa (zip bomb)…
ZIP_RATIO_MIN_BYTES = 64 * 1024**2  # …en archivos de más de 64 MB


def check_zip_limits(
    zf: zipfile.ZipFile, *, max_files: int = ZIP_MAX_FILES, max_bytes: int = ZIP_MAX_BYTES
) -> None:
    """Rechaza ZIPs que al extraerse llenarían el disco (cantidad, tamaño total o ratio)."""
    members = zf.infolist()
    if len(members) > max_files:
        raise ValidationError(f"el ZIP tiene demasiados archivos (máximo {max_files})")
    total = 0
    for m in members:
        total += m.file_size
        big = m.file_size > ZIP_RATIO_MIN_BYTES
        if big and m.file_size > ZIP_MAX_RATIO * max(m.compress_size, 1):
            raise ValidationError("el ZIP tiene una compresión sospechosa (zip bomb)")
    if total > max_bytes:
        raise ValidationError(f"el ZIP descomprimido supera {max_bytes // 1024**3} GB")


def _safe_extract(zf: zipfile.ZipFile, root: Path) -> None:
    """Extrae evitando rutas que escapen del destino (zip slip) y zip bombs."""
    check_zip_limits(zf)
    root = root.resolve()
    for member in zf.infolist():
        dest = (root / member.filename).resolve()
        if not dest.is_relative_to(root):
            raise ValidationError("ZIP con rutas inválidas", details={"member": member.filename})
    zf.extractall(root)  # rutas validadas arriba (zip slip)


PREVIEW_ANNOTATIONS = {".csv", ".json", ".xml"}  # anotaciones: se decide con la detección completa


@dataclass
class FolderPreview:
    kind: SourceKind
    total: int
    classes: dict[str, int]
    samples: list[dict[str, str | None]]


def folder_preview(names: Iterable[str], limit: int = 20) -> FolderPreview | None:
    """Vista previa de una carpeta (o zip) de imágenes/audio a partir de las rutas relativas,
    sin leer ni extraer archivos: clases (primer nivel de carpeta) con su cantidad y una muestra
    repartida entre clases. None si hace falta la detección completa (anotaciones, máscaras,
    tablas): la decide `open_source`."""
    from perceptron.data.audio import AUDIO_EXTENSIONS

    files = [
        n.replace("\\", "/")
        for n in names
        if n
        and not n.endswith("/")
        and not any(p.startswith((".", "__MACOSX")) for p in n.replace("\\", "/").split("/"))
    ]
    if not files:
        return None
    tops = {f.split("/")[0] for f in files}
    if len(tops) == 1 and all("/" in f for f in files):  # un zip con una sola carpeta adentro
        prefix = next(iter(tops)) + "/"
        files = [f[len(prefix) :] for f in files]
    suffixes = {Path(f).suffix.lower() for f in files}
    if suffixes & PREVIEW_ANNOTATIONS or {"images", "masks"} <= {f.split("/")[0] for f in files}:
        return None
    images = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
    audio = [f for f in files if Path(f).suffix.lower() in AUDIO_EXTENSIONS]
    chosen, kind = (images, SourceKind.IMAGE_FOLDER) if images else (audio, SourceKind.AUDIO_FOLDER)
    if not chosen:
        return None
    by_class: dict[str, list[str]] = {}
    for f in sorted(chosen):
        by_class.setdefault(f.split("/")[0] if "/" in f else "", []).append(f)
    samples: list[dict[str, str | None]] = []
    depth = 0
    while len(samples) < limit and any(depth < len(v) for v in by_class.values()):
        for label, items in by_class.items():
            if depth < len(items) and len(samples) < limit:
                samples.append({"path": items[depth], "label": label or None})
        depth += 1
    return FolderPreview(
        kind=kind,
        total=len(chosen),
        classes={k or "(sin etiqueta)": len(v) for k, v in sorted(by_class.items())},
        samples=samples,
    )


def _iter_images(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def _iter_audio(root: Path) -> Iterator[Path]:
    from perceptron.data.audio import AUDIO_EXTENSIONS

    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            yield p


def audio_folder_index(root: Path, files_dir: Path) -> pl.DataFrame:
    """Copia/enlaza los audios a `files_dir` y devuelve el índice (etiqueta = carpeta)."""
    from perceptron.data.audio import info

    rows: list[dict[str, object]] = []
    for src in _iter_audio(root):
        rel = src.relative_to(root)
        link_or_copy(src, files_dir / rel)
        try:
            meta = info(src)
            row = {
                "sample_rate": meta.sample_rate,
                "channels": meta.channels,
                "duration": meta.duration,
                "format": meta.format,
                "corrupt": False,
            }
        except Exception:  # archivo ilegible: se marca y se excluye del entrenamiento
            row = {
                "sample_rate": None,
                "channels": None,
                "duration": None,
                "format": None,
                "corrupt": True,
            }
        rows.append(
            {"path": rel.as_posix(), "label": rel.parts[0] if len(rel.parts) > 1 else None, **row}
        )
    schema = {
        "path": pl.String,
        "label": pl.String,
        "sample_rate": pl.Int32,
        "channels": pl.Int8,
        "duration": pl.Float64,
        "format": pl.String,
        "corrupt": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)


def _iter_texts(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*.txt")):
        if p.is_file():
            yield p


def text_folder_table(root: Path) -> pl.DataFrame:
    """Carpeta `clase/archivo.txt` → tabla (`path`, `text`, `label`)."""
    rows = []
    for p in _iter_texts(root):
        rel = p.relative_to(root)
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
        rows.append(
            {
                "path": rel.as_posix(),
                "text": text.strip(),
                "label": rel.parts[0] if len(rel.parts) > 1 else None,
            }
        )
    return pl.DataFrame(rows, schema={"path": pl.String, "text": pl.String, "label": pl.String})


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink (sin duplicar espacio) o copia si está en otro volumen."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def image_folder_index(root: Path, files_dir: Path) -> pl.DataFrame:
    """Copia/enlaza las imágenes a `files_dir` y devuelve el índice.

    La etiqueta es el primer nivel de carpeta (`clase/archivo`). Imágenes en la
    raíz quedan sin etiqueta (para etiquetar luego, Capa 4).
    """
    rows: list[dict[str, object]] = []
    for src in _iter_images(root):
        rel = src.relative_to(root)
        label = rel.parts[0] if len(rel.parts) > 1 else None
        link_or_copy(src, files_dir / rel)
        width = height = channels = None
        fmt = None
        corrupt = False
        try:
            with Image.open(src) as im:
                im.verify()
            with Image.open(src) as im:
                width, height = im.size
                channels = len(im.getbands())
                fmt = im.format
        except (UnidentifiedImageError, OSError, SyntaxError):
            corrupt = True
        rows.append(
            {
                "path": rel.as_posix(),
                "label": label,
                "width": width,
                "height": height,
                "channels": channels,
                "format": fmt,
                "corrupt": corrupt,
            }
        )
    schema = {
        "path": pl.String,
        "label": pl.String,
        "width": pl.Int32,
        "height": pl.Int32,
        "channels": pl.Int8,
        "format": pl.String,
        "corrupt": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)
