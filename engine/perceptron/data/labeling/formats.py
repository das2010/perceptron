"""Import/export de etiquetas (RF-LBL-06): CSV, JSONL, COCO, YOLO y Pascal VOC.

- CSV/JSONL (clase o multi-etiqueta): `sample_id` o una columna clave (`path` en imágenes,
  la primera columna del dataset en tablas) + `label` (multi-etiqueta separada por `;`).
- COCO/YOLO/VOC (cajas): se asocian por nombre de archivo de la imagen.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from xml.etree import ElementTree as ET

import polars as pl

from perceptron.core.errors import ValidationError
from perceptron.data.sources.files import check_zip_limits
from perceptron.domain.enums import LabelKind

if TYPE_CHECKING:
    from perceptron.domain.models import LabelSet
    from perceptron.services.labeling import LabelItem, LabelUpdate

Format = Literal["csv", "jsonl", "coco", "yolo", "voc", "events"]
# CSV de eventos de audio (RF-ING-02): una fila por evento, como `eventos.csv` de UC-09.
EVENT_COLUMNS = ("file_name", "inicio_s", "fin_s", "etiqueta")
_EVENT_ALIASES = {
    "file_name": ("file_name", "file", "archivo", "path", "filename"),
    "inicio_s": ("inicio_s", "start_s", "start", "onset", "inicio"),
    "fin_s": ("fin_s", "end_s", "end", "offset", "fin"),
    "etiqueta": ("etiqueta", "label", "event_label", "clase"),
}


def _sid(i: int) -> str:
    return f"row:{i}"


def _key_column(df: pl.DataFrame) -> str | None:
    if "path" in df.columns:
        return "path"
    candidates = [c for c in df.columns if not c.startswith("__")]
    return candidates[0] if candidates else None


def _label_str(label: Any) -> str:
    return ";".join(label) if isinstance(label, list) else ("" if label is None else str(label))


def export_labels(
    ls: LabelSet, df: pl.DataFrame, items: dict[str, LabelItem], fmt: Format
) -> tuple[bytes, str]:
    key = _key_column(df)
    keys = {_sid(i): (str(df[key][i]) if key else _sid(i)) for i in df["__i__"].to_list()}
    if fmt == "jsonl":
        # JSONL lleva todo: etiqueta y formas (cajas, polígonos, segmentos; RF-LBL-01).
        lines = []
        for sid, it in items.items():
            row: dict[str, Any] = {"sample_id": sid, **({key: keys[sid]} if key else {})}
            if it.label is not None:
                row["label"] = _label_str(it.label)
            for field in ("boxes", "polygons", "segments"):
                shapes = getattr(it, field)
                if shapes:
                    row[field] = [sh.model_dump() for sh in shapes]
            if len(row) > (2 if key else 1):
                lines.append(json.dumps(row, ensure_ascii=False) + "\n")
        return "".join(lines).encode("utf-8"), "application/x-ndjson"
    if fmt == "csv":
        rows = [
            {"sample_id": sid, **({key: keys[sid]} if key else {}), "label": _label_str(it.label)}
            for sid, it in items.items()
            if it.label is not None
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["sample_id", *([key] if key else []), "label"])
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8"
    if fmt == "events":
        if ls.kind is not LabelKind.TEMPORAL_EVENT or key != "path":
            raise ValidationError("el CSV de eventos exporta segmentos de audio")
        buf_ev = io.StringIO()
        ev_writer = csv.writer(buf_ev)
        ev_writer.writerow(EVENT_COLUMNS)
        for sid, it in items.items():
            for seg in it.segments:
                ev_writer.writerow([keys[sid], f"{seg.start_s:.3f}", f"{seg.end_s:.3f}", seg.label])
        return buf_ev.getvalue().encode("utf-8"), "text/csv; charset=utf-8"
    if ls.kind is not LabelKind.BOX or key != "path":
        raise ValidationError(f"{fmt.upper()} exporta cajas de imágenes")
    classes = ls.classes
    if fmt == "coco":
        coco: dict[str, Any] = {
            "images": [],
            "annotations": [],
            "categories": [{"id": k + 1, "name": c} for k, c in enumerate(classes)],
        }
        for n, (sid, it) in enumerate(items.items(), start=1):
            coco["images"].append({"id": n, "file_name": keys[sid], "width": 1, "height": 1})
            for b in it.boxes:
                coco["annotations"].append(
                    {
                        "id": len(coco["annotations"]) + 1,
                        "image_id": n,
                        "category_id": classes.index(b.label) + 1,
                        # Normalizadas (ancho/alto 1): el consumidor las escala a su tamaño.
                        "bbox": [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1],
                        "iscrowd": 0,
                    }
                )
        return json.dumps(coco, ensure_ascii=False).encode("utf-8"), "application/json"
    buf_zip = io.BytesIO()
    with zipfile.ZipFile(buf_zip, "w", zipfile.ZIP_DEFLATED) as z:
        if fmt == "yolo":
            z.writestr("classes.txt", "\n".join(classes) + "\n")
        for sid, it in items.items():
            stem = Path(keys[sid]).stem
            if fmt == "yolo":
                lines = [
                    f"{classes.index(b.label)} {(b.x1 + b.x2) / 2:.6f} {(b.y1 + b.y2) / 2:.6f} "
                    f"{b.x2 - b.x1:.6f} {b.y2 - b.y1:.6f}"
                    for b in it.boxes
                ]
                z.writestr(f"{stem}.txt", "\n".join(lines) + "\n")
            else:
                root = ET.Element("annotation")
                ET.SubElement(root, "filename").text = Path(keys[sid]).name
                for b in it.boxes:
                    obj = ET.SubElement(root, "object")
                    ET.SubElement(obj, "name").text = b.label
                    box = ET.SubElement(obj, "bndbox")
                    for tag, v in (("xmin", b.x1), ("ymin", b.y1), ("xmax", b.x2), ("ymax", b.y2)):
                        ET.SubElement(box, tag).text = f"{v:.6f}"
                z.writestr(f"{stem}.xml", ET.tostring(root, encoding="unicode"))
    return buf_zip.getvalue(), "application/zip"


def _by_key(df: pl.DataFrame) -> dict[str, str]:
    key = _key_column(df)
    if key is None:
        return {}
    out: dict[str, str] = {}
    for i, v in zip(df["__i__"].to_list(), df[key].to_list(), strict=True):
        out[str(v)] = _sid(i)
        if key == "path":
            out.setdefault(Path(str(v)).name, _sid(i))
            out.setdefault(Path(str(v)).stem, _sid(i))
    return out


def import_labels(ls: LabelSet, df: pl.DataFrame, data: bytes, fmt: Format) -> list[LabelUpdate]:
    from perceptron.services.labeling import Box, LabelUpdate

    keys = _by_key(df)
    multilabel = ls.kind is LabelKind.MULTILABEL

    def resolve(row: dict[str, Any]) -> str | None:
        sid = row.get("sample_id")
        if sid:
            return str(sid)
        for v in row.values():
            if str(v) in keys:
                return keys[str(v)]
        return None

    def label_of(raw: Any) -> str | list[str]:
        text = str(raw)
        return [x for x in text.split(";") if x] if multilabel else text

    updates: list[LabelUpdate] = []
    if fmt == "events":
        return _import_events(ls, keys, data)
    if fmt in ("csv", "jsonl"):
        text = data.decode("utf-8-sig")
        rows = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if fmt == "jsonl"
            else list(csv.DictReader(io.StringIO(text)))
        )
        for r in rows:
            sid = resolve(r)
            if sid is None:
                continue
            label = label_of(r["label"]) if r.get("label") not in (None, "") else None
            shapes = {f: r[f] for f in ("boxes", "polygons", "segments") if r.get(f)}
            if label is not None or shapes:
                updates.append(
                    LabelUpdate.model_validate({"sample_id": sid, "label": label, **shapes})
                )
        return updates
    if fmt == "coco":
        coco = json.loads(data.decode("utf-8"))
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        images = {im["id"]: im for im in coco.get("images", [])}
        boxes: dict[str, list[Box]] = {}
        for a in coco.get("annotations", []):
            im = images.get(a["image_id"])
            sid = keys.get(Path(im["file_name"]).name) if im else None
            if sid is None or im is None:
                continue
            w, h = float(im.get("width") or 1), float(im.get("height") or 1)
            x, y, bw, bh = (float(v) for v in a["bbox"])
            boxes.setdefault(sid, []).append(
                Box(
                    x1=x / w,
                    y1=y / h,
                    x2=(x + bw) / w,
                    y2=(y + bh) / h,
                    label=cats[a["category_id"]],
                )
            )
        return [LabelUpdate(sample_id=s, boxes=b) for s, b in boxes.items()]
    # YOLO/VOC: un zip con un archivo por imagen (y classes.txt en YOLO).
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        check_zip_limits(z, max_files=200_000, max_bytes=2 * 1024**3)
        names = z.namelist()
        classes = ls.classes
        if fmt == "yolo" and "classes.txt" in names:
            classes = [c for c in z.read("classes.txt").decode("utf-8").splitlines() if c.strip()]
        for name in names:
            stem = Path(name).stem
            sid = keys.get(stem)
            if sid is None or name == "classes.txt":
                continue
            got: list[Box] = []
            if fmt == "yolo" and name.endswith(".txt"):
                for line in z.read(name).decode("utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    c, cx, cy, bw, bh = int(parts[0]), *(float(p) for p in parts[1:])
                    got.append(
                        Box(
                            x1=cx - bw / 2,
                            y1=cy - bh / 2,
                            x2=cx + bw / 2,
                            y2=cy + bh / 2,
                            label=classes[c],
                        )
                    )
            elif fmt == "voc" and name.endswith(".xml"):
                raw = z.read(name)
                if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
                    # Sin DTD ni entidades: evita la expansión de entidades (billion laughs).
                    raise ValidationError(f"{name}: XML con DTD/entidades no permitido")
                root = ET.fromstring(raw)  # noqa: S314 - sin DTD ni entidades (verificado arriba)
                for obj in root.iter("object"):
                    bb = obj.find("bndbox")
                    if bb is None:
                        continue
                    vals = [float(bb.findtext(t) or 0) for t in ("xmin", "ymin", "xmax", "ymax")]
                    size = root.find("size")
                    w = float(size.findtext("width") or 1) if size is not None else 1.0
                    h = float(size.findtext("height") or 1) if size is not None else 1.0
                    norm = (
                        [vals[0] / w, vals[1] / h, vals[2] / w, vals[3] / h]
                        if max(vals) > 1
                        else vals
                    )
                    got.append(
                        Box(
                            x1=norm[0],
                            y1=norm[1],
                            x2=norm[2],
                            y2=norm[3],
                            label=obj.findtext("name") or "",
                        )
                    )
            if got:
                updates.append(LabelUpdate(sample_id=sid, boxes=got))
    return updates


def _import_events(ls: LabelSet, keys: dict[str, str], data: bytes) -> list[LabelUpdate]:
    """CSV de eventos → segmentos por muestra (el archivo se busca por ruta, nombre o stem)."""
    from perceptron.services.labeling import LabelUpdate, Segment

    if ls.kind is not LabelKind.TEMPORAL_EVENT:
        raise ValidationError("el CSV de eventos se importa en conjuntos de segmentos")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    fields = {c.strip().lower(): c for c in reader.fieldnames or []}
    cols: dict[str, str] = {}
    for canon, aliases in _EVENT_ALIASES.items():
        found = next((fields[a] for a in aliases if a in fields), None)
        if found is None:
            raise ValidationError(f"falta la columna {canon} en el CSV de eventos")
        cols[canon] = found
    by_sample: dict[str, list[Segment]] = {}
    unknown: set[str] = set()
    for row in reader:
        name = str(row[cols["file_name"]]).strip().replace("\\", "/")
        sid = keys.get(name) or keys.get(Path(name).name) or keys.get(Path(name).stem)
        if sid is None:
            unknown.add(name)
            continue
        try:
            seg = Segment(
                start_s=float(row[cols["inicio_s"]]),
                end_s=float(row[cols["fin_s"]]),
                label=str(row[cols["etiqueta"]]).strip(),
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"evento inválido para {name}: {exc}") from None
        by_sample.setdefault(sid, []).append(seg)
    if not by_sample:
        raise ValidationError(
            "ningún evento coincide con los archivos del dataset",
            details={"ejemplos": sorted(unknown)[:5]},
        )
    return [LabelUpdate(sample_id=sid, segments=segs) for sid, segs in by_sample.items()]
