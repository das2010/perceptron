"""Visión avanzada: detección, segmentación y OCR (RF-ING-02, §8).

Estructuras de carpeta soportadas:
- detección: carpeta de imágenes + anotaciones COCO (`*.json`), Pascal VOC (`*.xml`
  junto a cada imagen) o YOLO (`*.txt` junto a cada imagen + `classes.txt`);
- segmentación: `images/` + `masks/` (PNG con el mismo nombre; valor = clase, 255 = 1);
- OCR: `images/` + un CSV con columnas de archivo y texto.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from perceptron.core.errors import ValidationError
from perceptron.data.schema import IMAGE_EXTENSIONS
from perceptron.data.sources.files import link_or_copy
from perceptron.domain.enums import TaskType

_TEXT_COLUMNS = ("texto", "text", "label", "transcripcion", "transcription")
_FILE_COLUMNS = ("file_name", "filename", "file", "path", "imagen", "image")


def _images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _meta(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return {
                "width": im.size[0],
                "height": im.size[1],
                "channels": len(im.getbands()),
                "format": im.format,
                "corrupt": False,
            }
    except (UnidentifiedImageError, OSError, SyntaxError):
        return {"width": None, "height": None, "channels": None, "format": None, "corrupt": True}


def is_segmentation_folder(root: Path) -> bool:
    return (root / "images").is_dir() and (root / "masks").is_dir()


def ocr_labels_file(root: Path) -> Path | None:
    if not (root / "images").is_dir():
        return None
    for f in sorted(root.glob("*.csv")):
        with f.open(encoding="utf-8", newline="") as fh:
            header = [h.strip().lower() for h in next(csv.reader(fh), [])]
        if any(c in header for c in _FILE_COLUMNS) and any(c in header for c in _TEXT_COLUMNS):
            return f
    return None


def find_annotations(root: Path) -> Path | None:
    for f in sorted(root.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and {"images", "annotations"} <= set(data):
            return f
    return None


# ------------------------------------------------------------------ anotaciones


def load_coco(path: Path) -> tuple[dict[str, list[tuple[list[float], str]]], list[str]]:
    """{archivo: [(caja xyxy, clase)]}, clases."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cats = {c["id"]: str(c["name"]) for c in data.get("categories", [])}
    files = {img["id"]: str(img["file_name"]) for img in data["images"]}
    out: dict[str, list[tuple[list[float], str]]] = {f: [] for f in files.values()}
    for a in data["annotations"]:
        x, y, w, h = (float(v) for v in a["bbox"])
        out[files[a["image_id"]]].append(
            ([x, y, x + w, y + h], cats.get(a["category_id"], str(a["category_id"])))
        )
    return out, sorted(set(cats.values()))


def load_voc(image: Path) -> list[tuple[list[float], str]]:
    xml = image.with_suffix(".xml")
    if not xml.is_file():
        return []
    root = ET.parse(xml).getroot()  # noqa: S314 - archivo local del usuario
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        name = obj.findtext("name", "objeto")
        if bb is not None:
            vals = [float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")]
            boxes.append((vals, name))
    return boxes


def load_yolo(
    image: Path, classes: list[str], size: tuple[int, int]
) -> list[tuple[list[float], str]]:
    txt = image.with_suffix(".txt")
    if not txt.is_file():
        return []
    w, h = size
    boxes = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        k, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
        name = classes[k] if k < len(classes) else str(k)
        boxes.append(
            ([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], name)
        )
    return boxes


# ------------------------------------------------------------------ índices por tarea


def detection_index(
    root: Path, files_dir: Path, annotations: Path | None
) -> tuple[pl.DataFrame, list[str]]:
    coco = load_coco(annotations) if annotations else None
    yolo_classes_file = root / "classes.txt"
    yolo_classes = (
        yolo_classes_file.read_text(encoding="utf-8").split() if yolo_classes_file.is_file() else []
    )
    rows, classes = [], set(coco[1] if coco else [])
    for img in _images(root):
        rel = img.relative_to(root).as_posix()
        meta = _meta(img)
        if coco:
            boxes = coco[0].get(rel, coco[0].get(img.name, []))
        else:
            boxes = load_voc(img) or load_yolo(
                img, yolo_classes, (meta["width"] or 1, meta["height"] or 1)
            )
        link_or_copy(img, files_dir / rel)
        classes.update(n for _, n in boxes)
        rows.append(
            {
                "path": rel,
                **meta,
                "boxes": [b for b, _ in boxes],
                "box_labels": [n for _, n in boxes],
            }
        )
    if not rows:
        raise ValidationError("no se encontraron imágenes para detección")
    schema = {
        "path": pl.String,
        "width": pl.Int32,
        "height": pl.Int32,
        "channels": pl.Int8,
        "format": pl.String,
        "corrupt": pl.Boolean,
        "boxes": pl.List(pl.List(pl.Float64)),
        "box_labels": pl.List(pl.String),
    }
    return pl.DataFrame(rows, schema=schema), sorted(classes)


def segmentation_index(root: Path, files_dir: Path) -> tuple[pl.DataFrame, list[str]]:
    images_dir, masks_dir = root / "images", root / "masks"
    rows, values = [], set()
    for img in _images(images_dir):
        mask = masks_dir / img.relative_to(images_dir).with_suffix(".png")
        if not mask.is_file():
            continue
        rel_img = img.relative_to(root).as_posix()
        rel_mask = mask.relative_to(root).as_posix()
        link_or_copy(img, files_dir / rel_img)
        link_or_copy(mask, files_dir / rel_mask)
        with Image.open(mask) as m:
            values.update(np.unique(np.asarray(m.convert("L"))).tolist())
        rows.append({"path": rel_img, "mask_path": rel_mask, **_meta(img)})
    if not rows:
        raise ValidationError("no se encontraron pares imagen/máscara (images/ + masks/)")
    # Máscaras binarias 0/255 → clases fondo/objeto; si no, cada valor es una clase.
    ids = sorted(1 if v == 255 else int(v) for v in values)
    k = max(ids) + 1 if ids else 2
    classes = ["fondo"] + [f"clase_{i}" for i in range(1, k)] if k > 2 else ["fondo", "objeto"]
    schema = {
        "path": pl.String,
        "mask_path": pl.String,
        "width": pl.Int32,
        "height": pl.Int32,
        "channels": pl.Int8,
        "format": pl.String,
        "corrupt": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema), classes


def ocr_index(root: Path, files_dir: Path, labels: Path) -> tuple[pl.DataFrame, list[str]]:
    with labels.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = {f.strip().lower(): f for f in reader.fieldnames or []}
        fcol = next(fields[c] for c in _FILE_COLUMNS if c in fields)
        tcol = next(fields[c] for c in _TEXT_COLUMNS if c in fields)
        pairs = [(r[fcol], r[tcol]) for r in reader]
    rows, alphabet = [], set()
    for name, text in pairs:
        img = root / "images" / name
        if not img.is_file():
            img = root / name
        if not img.is_file():
            continue
        rel = img.relative_to(root).as_posix()
        link_or_copy(img, files_dir / rel)
        alphabet.update(text)
        rows.append({"path": rel, "text": text, **_meta(img)})
    if not rows:
        raise ValidationError("el CSV de OCR no referencia imágenes existentes")
    schema = {
        "path": pl.String,
        "text": pl.String,
        "width": pl.Int32,
        "height": pl.Int32,
        "channels": pl.Int8,
        "format": pl.String,
        "corrupt": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema), sorted(alphabet)


# ------------------------------------------------------------------ profiling


class VisionTaskProfile(BaseModel):
    task: TaskType
    classes: list[str] = Field(default_factory=list)
    objects_per_image_p50: float | None = None
    objects_per_image_max: int | None = None
    images_without_objects: int | None = None
    box_area_fraction_p50: float | None = None
    foreground_fraction: float | None = None
    text_length_p50: float | None = None
    text_length_p95: float | None = None
    alphabet_size: int | None = None


def profile_vision_task(
    task: TaskType, index: pl.DataFrame, files_dir: Path, classes: list[str]
) -> VisionTaskProfile:
    from perceptron.data.profiling.card import sig

    ok = index.filter(~pl.col("corrupt"))
    if task is TaskType.OBJECT_DETECTION:
        counts = ok["boxes"].list.len().to_numpy()
        areas = []
        for boxes, w, h in ok.select("boxes", "width", "height").iter_rows():
            areas += [((b[2] - b[0]) * (b[3] - b[1])) / max(w * h, 1) for b in boxes or []]
        return VisionTaskProfile(
            task=task,
            classes=classes,
            objects_per_image_p50=sig(float(np.median(counts))) if len(counts) else None,
            objects_per_image_max=int(counts.max()) if len(counts) else None,
            images_without_objects=int((counts == 0).sum()),
            box_area_fraction_p50=sig(float(np.median(areas))) if areas else None,
        )
    if task is TaskType.SEGMENTATION:
        fg = []
        for rel in ok["mask_path"].head(200).to_list():
            with Image.open(files_dir / rel) as m:
                fg.append(float((np.asarray(m.convert("L")) > 0).mean()))
        return VisionTaskProfile(
            task=task, classes=classes, foreground_fraction=sig(float(np.mean(fg))) if fg else None
        )
    lengths = ok["text"].str.len_chars().to_numpy()
    return VisionTaskProfile(
        task=task,
        alphabet_size=len(classes),
        text_length_p50=sig(float(np.quantile(lengths, 0.5))) if len(lengths) else None,
        text_length_p95=sig(float(np.quantile(lengths, 0.95))) if len(lengths) else None,
    )
