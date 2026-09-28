"""Etiquetado asistido (SPEC §7.5, RF-LBL-01..06; ADR-0029).

Un `LabelSet` anota las muestras de una versión de dataset (inmutable). Cada muestra tiene un
id estable, `row:<i>`: su posición en la tabla de la versión. Los ítems guardan etiqueta,
origen (humano, modelo o LLM), confianza y estado (sugerida o aceptada), en un JSONL por
conjunto (`labels/<id>.jsonl`).

Flujo:
1. Crear el conjunto (se siembran como aceptadas las etiquetas que ya traía el dataset).
2. Pre-etiquetar con el modelo del proyecto o con el LLM.
3. Revisar por cola de active learning (incertidumbre o diversidad) y aceptar en lote las
   sugerencias muy confiadas.
4. Aplicar: se crea una versión nueva del dataset (`parent_id`) con perfil y splits propios,
   lista para reentrenar.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from perceptron.core.errors import NotFoundError, ValidationError
from perceptron.data.splits import FOLD_COLUMN, SPLIT_COLUMN
from perceptron.data.view import DatasetView, Purpose
from perceptron.domain.enums import DataSourceType, LabelKind, LabelOrigin, Modality
from perceptron.domain.models import DatasetVersion, DataSource, LabelSet, Run, utcnow

Status = Literal["suggested", "accepted"]
QueueStrategy = Literal["uncertainty", "diversity", "random"]
_INTERNAL = {SPLIT_COLUMN, FOLD_COLUMN, "corrupt"}


class Box(BaseModel):
    """Caja normalizada a [0, 1] (x1, y1) – (x2, y2)."""

    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)
    label: str


class Polygon(BaseModel):
    """Polígono normalizado a [0, 1] (máscaras de segmentación, RF-LBL-01)."""

    points: list[list[float]] = Field(min_length=3, max_length=2000, description="[[x, y], ...]")
    label: str

    @field_validator("points")
    @classmethod
    def _in_unit(cls, pts: list[list[float]]) -> list[list[float]]:
        if any(len(p) != 2 or not all(0 <= v <= 1 for v in p) for p in pts):
            raise ValueError("cada punto es [x, y] normalizado a [0, 1]")
        return pts


class Segment(BaseModel):
    """Segmento temporal en segundos (eventos de audio, RF-LBL-01)."""

    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    label: str

    @model_validator(mode="after")
    def _ordered(self) -> Segment:
        if self.end_s <= self.start_s:
            raise ValueError("el segmento termina antes de empezar")
        return self


class LabelItem(BaseModel):
    sample_id: str
    label: str | list[str] | None = None
    boxes: list[Box] = Field(default_factory=list)
    polygons: list[Polygon] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    origin: LabelOrigin = LabelOrigin.HUMAN
    confidence: float | None = None
    status: Status = "accepted"
    updated_at: str = Field(default_factory=lambda: utcnow().isoformat())


class LabelUpdate(BaseModel):
    sample_id: str
    label: str | list[str] | None = None
    boxes: list[Box] = Field(default_factory=list)
    polygons: list[Polygon] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)

    def empty(self) -> bool:
        return self.label is None and not (self.boxes or self.polygons or self.segments)


class Sample(BaseModel):
    sample_id: str
    split: str | None
    path: str | None = None
    text: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    item: LabelItem | None = None


class LabelingSummary(BaseModel):
    labelset: LabelSet
    total: int
    accepted: int
    suggested: int
    unlabeled: int
    by_class: dict[str, int]


class LabelQuality(BaseModel):
    compared: int
    agreement: float | None = Field(
        description="Humano vs modelo, sobre las aceptadas con sugerencia"
    )
    confusing_pairs: list[dict[str, Any]]
    suspected_errors: list[str] = Field(description="sample_id con etiqueta humana dudosa")


def _row_id(i: int) -> str:
    return f"row:{i}"


def _row_index(sample_id: str) -> int:
    kind, _, num = sample_id.partition(":")
    if kind != "row" or not num.isdigit():
        raise ValidationError(f"sample_id inválido: {sample_id!r}")
    return int(num)


def _json_value(v: Any) -> Any:
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    return str(v)


class Labeling:
    def __init__(self, workflow: Any) -> None:
        self.wf = workflow
        self.ctx = workflow.ctx

    # ---------------------------------------------------------------- almacenamiento

    def _labelset(self, labelset_id: str) -> LabelSet:
        ls: LabelSet = self.ctx.repo(LabelSet).get(labelset_id)
        return ls

    def _dataset(self, ls: LabelSet) -> DatasetVersion:
        dv: DatasetVersion = self.wf.dataset(ls.dataset_version_id)
        return dv

    def _items_path(self, ls: LabelSet) -> Path:
        dv = self._dataset(ls)
        root: Path = self.ctx.settings.paths.project(dv.project_id).root
        return root / (ls.path or f"labels/{ls.id}.jsonl")

    def items(self, labelset_id: str) -> dict[str, LabelItem]:
        path = self._items_path(self._labelset(labelset_id))
        if not path.is_file():
            return {}
        items = [
            LabelItem.model_validate_json(line)
            for line in path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        return {i.sample_id: i for i in items}

    def _save(self, ls: LabelSet, items: dict[str, LabelItem]) -> None:
        path = self._items_path(ls)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            "".join(i.model_dump_json() + "\n" for i in items.values()), encoding="utf-8"
        )
        tmp.replace(path)

    def frame(self, dv: DatasetVersion) -> pl.DataFrame:
        """Todas las muestras de la versión (incluido el test: la persona anota, no se elige)."""
        view: DatasetView = self.wf.view(dv)
        df = view.read(None, purpose=Purpose.LABELING)
        return df.with_row_index("__i__")

    # ---------------------------------------------------------------- alta y consulta

    def create(
        self,
        dataset_version_id: str,
        *,
        kind: LabelKind = LabelKind.CLASS,
        classes: list[str] | None = None,
        name: str | None = None,
        target: str | None = None,
    ) -> LabelSet:
        dv: DatasetVersion = self.wf.dataset(dataset_version_id)
        df = self.frame(dv)
        target_col = target or dv.target or "label"
        known: list[str] = []
        items: dict[str, LabelItem] = {}
        if target_col in df.columns and kind in (LabelKind.CLASS, LabelKind.MULTILABEL):
            # Las etiquetas que ya trae el dataset cuentan como humanas y aceptadas.
            for i, v in zip(df["__i__"].to_list(), df[target_col].to_list(), strict=True):
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    continue
                label = str(v)
                items[_row_id(i)] = LabelItem(sample_id=_row_id(i), label=label)
                if label not in known:
                    known.append(label)
        all_classes = list(dict.fromkeys([*(classes or []), *sorted(known)]))
        if not all_classes:
            raise ValidationError("indicá las clases a etiquetar")
        ls = LabelSet(
            dataset_version_id=dv.id,
            kind=kind,
            classes=all_classes,
            name=name,
            target=target_col,
        )
        ls.path = f"labels/{ls.id}.jsonl"
        self.ctx.repo(LabelSet).add(ls)
        self._save(ls, items)
        return ls

    def list_sets(self, dataset_version_id: str) -> list[LabelSet]:
        sets: list[LabelSet] = self.ctx.repo(LabelSet).list(
            filters={"dataset_version_id": dataset_version_id}, limit=200
        )
        return sets

    def summary(self, labelset_id: str) -> LabelingSummary:
        ls = self._labelset(labelset_id)
        total = self.frame(self._dataset(ls)).height
        items = self.items(labelset_id).values()
        accepted = [i for i in items if i.status == "accepted"]
        by_class: dict[str, int] = {}
        for i in accepted:
            for lab in i.label if isinstance(i.label, list) else [i.label]:
                if lab is not None:
                    by_class[lab] = by_class.get(lab, 0) + 1
            for shape in [*i.boxes, *i.polygons, *i.segments]:
                by_class[shape.label] = by_class.get(shape.label, 0) + 1
        suggested = sum(1 for i in items if i.status == "suggested")
        return LabelingSummary(
            labelset=ls,
            total=total,
            accepted=len(accepted),
            suggested=suggested,
            unlabeled=total - len(accepted),
            by_class=by_class,
        )

    def _sample(
        self, dv: DatasetVersion, row: dict[str, Any], item: LabelItem | None, target: str
    ) -> Sample:
        view: DatasetView = self.wf.view(dv)
        text_col = view.text_column
        fields = {
            k: _json_value(v)
            for k, v in row.items()
            if k not in _INTERNAL and k not in {"__i__", target, "path"}
        }
        return Sample(
            sample_id=_row_id(int(row["__i__"])),
            split=row.get(SPLIT_COLUMN),
            path=row.get("path"),
            text=str(row[text_col]) if text_col and row.get(text_col) is not None else None,
            fields=fields if "path" not in row else {},
            item=item,
        )

    def queue(
        self, labelset_id: str, *, strategy: QueueStrategy = "uncertainty", limit: int = 20
    ) -> list[Sample]:
        """Próximas muestras a revisar (RF-LBL-03): sin etiqueta aceptada, priorizadas."""
        ls = self._labelset(labelset_id)
        dv = self._dataset(ls)
        items = self.items(labelset_id)
        df = self.frame(dv)
        pending = [
            r
            for r in df.iter_rows(named=True)
            if items.get(_row_id(r["__i__"]), None) is None
            or items[_row_id(r["__i__"])].status != "accepted"
        ]
        if strategy == "uncertainty":
            # Menor confianza primero; las que no tienen sugerencia van después, en orden.
            def conf(r: dict[str, Any]) -> float:
                it = items.get(_row_id(r["__i__"]))
                return it.confidence if it and it.confidence is not None else 2.0

            pending.sort(key=conf)
        elif strategy == "diversity":
            # Ronda por clase sugerida (y sin sugerencia): cubre el espacio antes de profundizar.
            groups: dict[str, list[dict[str, Any]]] = {}
            for r in pending:
                it = items.get(_row_id(r["__i__"]))
                groups.setdefault(str(it.label) if it else "∅", []).append(r)
            rng = random.Random(0)  # noqa: S311 - orden de revisión reproducible, no criptografía
            for g in groups.values():
                rng.shuffle(g)
            mixed: list[dict[str, Any]] = []
            while any(groups.values()):
                for key in list(groups):
                    if groups[key]:
                        mixed.append(groups[key].pop())
            pending = mixed
        else:
            random.Random(0).shuffle(pending)  # noqa: S311 - orden reproducible
        return [
            self._sample(dv, r, items.get(_row_id(r["__i__"])), ls.target) for r in pending[:limit]
        ]

    def sample_file(self, labelset_id: str, sample_id: str) -> Path:
        """Archivo (imagen/audio) de una muestra, para mostrarlo en la herramienta."""
        ls = self._labelset(labelset_id)
        dv = self._dataset(ls)
        df = self.frame(dv)
        i = _row_index(sample_id)
        if i >= df.height or "path" not in df.columns:
            raise NotFoundError(f"la muestra {sample_id} no tiene archivo")
        view: DatasetView = self.wf.view(dv)
        path = view.files_dir / str(df["path"][i])
        if not path.resolve().is_relative_to(view.files_dir.resolve()):
            raise NotFoundError("ruta de muestra inválida")
        return path

    # ---------------------------------------------------------------- escritura

    def _check_label(self, ls: LabelSet, update: LabelUpdate) -> None:
        labels = update.label if isinstance(update.label, list) else [update.label]
        if ls.kind is LabelKind.MULTILABEL and not isinstance(update.label, list):
            raise ValidationError("en multi-etiqueta, label es una lista")
        unknown = [lab for lab in labels if lab is not None and lab not in ls.classes]
        shapes = [*update.boxes, *update.polygons, *update.segments]
        unknown += [sh.label for sh in shapes if sh.label not in ls.classes]
        if unknown:
            raise ValidationError(f"clases desconocidas: {sorted(set(unknown))}")

    def set_labels(self, labelset_id: str, updates: list[LabelUpdate]) -> int:
        """Etiquetas de la persona (aceptadas). Sin etiqueta ni formas, se borra la muestra."""
        ls = self._labelset(labelset_id)
        total = self.frame(self._dataset(ls)).height
        items = self.items(labelset_id)
        for u in updates:
            if _row_index(u.sample_id) >= total:
                raise ValidationError(f"{u.sample_id} no existe en el dataset")
            self._check_label(ls, u)
            if u.empty():
                items.pop(u.sample_id, None)
                continue
            items[u.sample_id] = LabelItem(
                sample_id=u.sample_id,
                label=u.label,
                boxes=u.boxes,
                polygons=u.polygons,
                segments=u.segments,
            )
        self._save(ls, items)
        return len(updates)

    def accept_suggestions(self, labelset_id: str, *, min_confidence: float = 0.9) -> int:
        """Acepta en lote las sugerencias con confianza ≥ umbral (una sola acción)."""
        ls = self._labelset(labelset_id)
        items = self.items(labelset_id)
        n = 0
        for key, it in items.items():
            if it.status == "suggested" and (it.confidence or 0.0) >= min_confidence:
                items[key] = it.model_copy(
                    update={"status": "accepted", "updated_at": utcnow().isoformat()}
                )
                n += 1
        self._save(ls, items)
        return n

    def add_classes(self, labelset_id: str, classes: list[str]) -> LabelSet:
        ls = self._labelset(labelset_id)
        merged = list(dict.fromkeys([*ls.classes, *classes]))
        updated: LabelSet = self.ctx.repo(LabelSet).update(
            ls.model_copy(update={"classes": merged})
        )
        return updated

    # ---------------------------------------------------------------- pre-etiquetado (RF-LBL-02)

    def prelabel_with_model(self, labelset_id: str, run_id: str) -> int:
        """Sugerencias del modelo del proyecto para las muestras sin etiqueta aceptada."""
        from perceptron.training.data import make_dataset_from_frame
        from perceptron.training.inference import load_trained, predict

        ls = self._labelset(labelset_id)
        if ls.kind is not LabelKind.CLASS:
            raise ValidationError("el pre-etiquetado con el modelo cubre etiquetas de clase")
        run: Run = self.ctx.repo(Run).get(run_id)
        trained = load_trained(self.wf._run_dir(run))
        if trained.task.value != "classification":
            raise ValidationError("el modelo del run no es de clasificación")
        dv = self._dataset(ls)
        items = self.items(labelset_id)
        df = self.frame(dv)
        pending = df.filter(
            ~pl.col("__i__").map_elements(
                lambda i: _row_id(i) in items and items[_row_id(i)].status == "accepted",
                return_dtype=pl.Boolean,
            )
        )
        if "corrupt" in pending.columns:
            # El dataset de imágenes descarta las corruptas: sin esto se desalinearían las filas.
            pending = pending.filter(~pl.col("corrupt"))
        if pending.is_empty():
            return 0
        view: DatasetView = self.wf.view(dv)
        ds = make_dataset_from_frame(view, trained.pipeline, pending)
        preds = predict(trained, ds)
        classes = trained.pipeline.classes or []
        proba = np.asarray(preds.proba) if preds.proba is not None else None
        n = 0
        for k, i in enumerate(pending["__i__"].to_list()):
            label = classes[int(preds.y_pred[k])] if classes else str(int(preds.y_pred[k]))
            conf = float(proba[k].max()) if proba is not None else None
            items[_row_id(i)] = LabelItem(
                sample_id=_row_id(i),
                label=label,
                origin=LabelOrigin.MODEL,
                confidence=conf,
                status="suggested",
            )
            n += 1
        new_classes = [c for c in classes if c not in ls.classes]
        if new_classes:
            ls = self.add_classes(labelset_id, new_classes)
        self._save(ls, items)
        self.snapshot_model_opinions(labelset_id)  # para medir el acuerdo humano-modelo luego
        return n

    def prelabel_with_llm(self, labelset_id: str, guide: Any, *, limit: int = 200) -> int:
        """Sugerencias del LLM (texto) con la guía de etiquetado (RF-LBL-04), L2/L3."""
        ls = self._labelset(labelset_id)
        dv = self._dataset(ls)
        view: DatasetView = self.wf.view(dv)
        column = view.text_column
        if column is None:
            raise ValidationError("el pre-etiquetado con LLM necesita una columna de texto")
        roles = self.wf.roles
        items = self.items(labelset_id)
        df = self.frame(dv)
        pending = [
            (int(r["__i__"]), str(r[column]))
            for r in df.iter_rows(named=True)
            if not (
                items.get(_row_id(r["__i__"])) and items[_row_id(r["__i__"])].status == "accepted"
            )
        ][:limit]
        suggestions = roles.prelabel_texts(dv, guide, {_row_id(i): t for i, t in pending}, column)
        for s in suggestions:
            items[s["sample_id"]] = LabelItem(
                sample_id=s["sample_id"],
                label=s["label"],
                origin=LabelOrigin.LLM,
                confidence=s.get("confidence"),
                status="suggested",
            )
        self._save(ls, items)
        return len(suggestions)

    def prelabel_zero_shot(
        self, labelset_id: str, *, model: str | None = None, limit: int = 200
    ) -> int:
        """Sugerencias zero-shot locales con los nombres de clase (RF-LBL-02), sin entrenar."""
        from perceptron.data.labeling.zero_shot import classify

        ls = self._labelset(labelset_id)
        if ls.kind not in (LabelKind.CLASS, LabelKind.MULTILABEL):
            raise ValidationError("el zero-shot cubre etiquetas de clase y multi-etiqueta")
        dv = self._dataset(ls)
        view: DatasetView = self.wf.view(dv)
        items = self.items(labelset_id)
        df = self.frame(dv)
        if "corrupt" in df.columns:
            df = df.filter(~pl.col("corrupt"))
        rows = [
            r
            for r in df.iter_rows(named=True)
            if not (
                items.get(_row_id(r["__i__"])) and items[_row_id(r["__i__"])].status == "accepted"
            )
        ][:limit]
        if not rows:
            return 0
        modality = view.modality
        column = view.text_column
        if column and modality not in (Modality.IMAGE, Modality.AUDIO):
            modality, inputs = Modality.TEXT, [str(r[column] or "") for r in rows]
        elif modality in (Modality.IMAGE, Modality.AUDIO) and "path" in df.columns:
            inputs = [str(view.files_dir / str(r["path"])) for r in rows]
        else:
            raise ValidationError("el zero-shot cubre imagen, texto y audio")
        multi = ls.kind is LabelKind.MULTILABEL
        scores = classify(modality, inputs, ls.classes, model=model, multi_label=multi)
        for r, sc in zip(rows, scores, strict=True):
            if not sc:
                continue
            best = max(sc, key=lambda c: sc[c])
            label: str | list[str] = [c for c, v in sc.items() if v >= 0.5] or [best]
            if not multi:
                label = best
            items[_row_id(r["__i__"])] = LabelItem(
                sample_id=_row_id(r["__i__"]),
                label=label,
                origin=LabelOrigin.MODEL,
                confidence=float(sc[best]),
                status="suggested",
            )
        self._save(ls, items)
        return len(rows)

    # ---------------------------------------------------------------- calidad (RF-LBL-05)

    def quality(self, labelset_id: str, *, error_confidence: float = 0.9) -> LabelQuality:
        """Acuerdo humano-modelo y posibles errores: donde el modelo discrepa con mucha confianza.

        Usa las sugerencias guardadas en el historial de cada muestra aceptada (`suggested_*`).
        """
        items = self.items(labelset_id)
        path = self._items_path(self._labelset(labelset_id)).with_suffix(".model.jsonl")
        model: dict[str, LabelItem] = {}
        if path.is_file():
            model = {
                it.sample_id: it
                for it in (
                    LabelItem.model_validate_json(line)
                    for line in path.read_text("utf-8").splitlines()
                    if line.strip()
                )
            }
        pairs: dict[tuple[str, str], int] = {}
        compared = agree = 0
        suspected: list[str] = []
        for sid, it in items.items():
            m = model.get(sid)
            if (
                it.status != "accepted"
                or it.origin is not LabelOrigin.HUMAN
                or m is None
                or it.label is None
            ):
                continue
            compared += 1
            if m.label == it.label:
                agree += 1
            else:
                key = (str(it.label), str(m.label))
                pairs[key] = pairs.get(key, 0) + 1
                if (m.confidence or 0.0) >= error_confidence:
                    suspected.append(sid)
        return LabelQuality(
            compared=compared,
            agreement=(agree / compared) if compared else None,
            confusing_pairs=[
                {"human": h, "model": mo, "count": c}
                for (h, mo), c in sorted(pairs.items(), key=lambda x: -x[1])[:10]
            ],
            suspected_errors=suspected[:200],
        )

    def snapshot_model_opinions(self, labelset_id: str) -> None:
        """Guarda las sugerencias actuales del modelo aparte, para medir acuerdo después."""
        ls = self._labelset(labelset_id)
        items = self.items(labelset_id)
        path = self._items_path(ls).with_suffix(".model.jsonl")
        prev: dict[str, str] = {}
        if path.is_file():
            prev = {
                json.loads(line)["sample_id"]: line
                for line in path.read_text("utf-8").splitlines()
                if line.strip()
            }
        for sid, it in items.items():
            if it.origin is LabelOrigin.MODEL:
                prev[sid] = it.model_dump_json()
        path.write_text("".join(v + "\n" for v in prev.values()), encoding="utf-8")

    # ---------------------------------------------------------------- import/export (RF-LBL-06)

    def export(
        self, labelset_id: str, fmt: Literal["csv", "jsonl", "coco", "yolo", "voc", "events"]
    ) -> tuple[bytes, str]:
        from perceptron.data.labeling.formats import export_labels

        ls = self._labelset(labelset_id)
        dv = self._dataset(ls)
        df = self.frame(dv)
        accepted = {k: v for k, v in self.items(labelset_id).items() if v.status == "accepted"}
        return export_labels(ls, df, accepted, fmt)

    def import_labels(
        self,
        labelset_id: str,
        data: bytes,
        fmt: Literal["csv", "jsonl", "coco", "yolo", "voc", "events"],
    ) -> int:
        from perceptron.data.labeling.formats import import_labels

        ls = self._labelset(labelset_id)
        df = self.frame(self._dataset(ls))
        updates = import_labels(ls, df, data, fmt)
        return self.set_labels(labelset_id, updates)

    # ---------------------------------------------------------------- aplicar

    def apply(self, labelset_id: str) -> DatasetVersion:
        """Versión nueva del dataset con las etiquetas aceptadas (lista para reentrenar)."""
        ls = self._labelset(labelset_id)
        dv = self._dataset(ls)
        view: DatasetView = self.wf.view(dv)
        df = self.frame(dv)
        items = {k: v for k, v in self.items(labelset_id).items() if v.status == "accepted"}
        if not items:
            raise ValidationError("no hay etiquetas aceptadas para aplicar")
        project_root: Path = self.ctx.settings.paths.project(dv.project_id).root
        work = project_root / "labels" / ls.id / "applied"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        labeled = df.filter(
            pl.col("__i__").map_elements(lambda i: _row_id(i) in items, return_dtype=pl.Boolean)
        )
        if ls.kind is LabelKind.BOX:
            source = self._apply_boxes(view, labeled, items, work)
            task = "object_detection"
        elif ls.kind is LabelKind.MASK:
            source = self._apply_masks(ls, view, labeled, items, work)
            task = "segmentation"
        elif ls.kind is LabelKind.TEMPORAL_EVENT:
            source = self._apply_segments(view, labeled, items, work)
            task = None
        elif "path" in df.columns and view.modality in (Modality.IMAGE, Modality.AUDIO):
            source = work / "dataset"
            for r in labeled.iter_rows(named=True):
                it = items[_row_id(r["__i__"])]
                label = it.label[0] if isinstance(it.label, list) else it.label
                dest = source / str(label) / Path(str(r["path"])).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(view.files_dir / str(r["path"]), dest)
            task = None
        else:
            labels = [items[_row_id(i)].label for i in labeled["__i__"].to_list()]
            flat = [";".join(v) if isinstance(v, list) else v for v in labels]
            keep = [
                c for c in labeled.columns if c not in _INTERNAL and c not in {"__i__", ls.target}
            ]
            out = labeled.select(keep).with_columns(pl.Series(ls.target, flat, dtype=pl.String))
            source = work / "dataset.parquet"
            out.write_parquet(source)
            task = None
        src = DataSource(
            project_id=dv.project_id,
            name=f"etiquetas {ls.name or ls.id}",
            type=DataSourceType.FOLDER if source.is_dir() else DataSourceType.FILE,
            config={"path": str(source), "labelset_id": ls.id},
        )
        self.ctx.repo(DataSource).add(src)
        new: DatasetVersion = self.wf.ingest(
            dv.project_id,
            source,
            target=None if source.is_dir() else ls.target,
            source_record=src,
            **({"task": task} if task else {}),
        )
        if new.parent_id is None:
            new = self.ctx.repo(DatasetVersion).update(
                new.model_copy(update={"parent_id": dv.id, "transformation": f"labels:{ls.id}"})
            )
        self.ctx.repo(LabelSet).update(
            self._labelset(labelset_id).model_copy(update={"applied_version_id": new.id})
        )
        return new

    def _apply_boxes(
        self, view: DatasetView, labeled: pl.DataFrame, items: dict[str, LabelItem], work: Path
    ) -> Path:
        from PIL import Image

        folder = work / "dataset"
        images_dir = folder
        images_dir.mkdir(parents=True, exist_ok=True)
        categories: dict[str, int] = {}
        coco: dict[str, Any] = {"images": [], "annotations": [], "categories": []}
        for k, r in enumerate(labeled.iter_rows(named=True)):
            it = items[_row_id(r["__i__"])]
            src = view.files_dir / str(r["path"])
            name = f"{k:06d}_{Path(str(r['path'])).name}"
            shutil.copy2(src, images_dir / name)
            with Image.open(src) as im:
                w, h = im.size
            coco["images"].append({"id": k + 1, "file_name": name, "width": w, "height": h})
            for b in it.boxes:
                cid = categories.setdefault(b.label, len(categories) + 1)
                x, y = b.x1 * w, b.y1 * h
                bw, bh = (b.x2 - b.x1) * w, (b.y2 - b.y1) * h
                coco["annotations"].append(
                    {
                        "id": len(coco["annotations"]) + 1,
                        "image_id": k + 1,
                        "category_id": cid,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )
        coco["categories"] = [{"id": i, "name": n} for n, i in categories.items()]
        (folder / "annotations_coco.json").write_text(json.dumps(coco), encoding="utf-8")
        return folder

    def _apply_masks(
        self,
        ls: LabelSet,
        view: DatasetView,
        labeled: pl.DataFrame,
        items: dict[str, LabelItem],
        work: Path,
    ) -> Path:
        """Polígonos → `images/` + `masks/` (PNG, valor = índice de clase + 1; 0 = fondo)."""
        from PIL import Image, ImageDraw

        folder = work / "dataset"
        images_dir, masks_dir = folder / "images", folder / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        for k, r in enumerate(labeled.iter_rows(named=True)):
            it = items[_row_id(r["__i__"])]
            if not it.polygons:
                continue
            src = view.files_dir / str(r["path"])
            name = f"{k:06d}_{Path(str(r['path'])).name}"
            shutil.copy2(src, images_dir / name)
            with Image.open(src) as im:
                w, h = im.size
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            # En orden: el último polígono queda arriba donde se superponen.
            for poly in it.polygons:
                pts = [(x * (w - 1), y * (h - 1)) for x, y in poly.points]
                draw.polygon(pts, fill=ls.classes.index(poly.label) + 1)
            mask.save(masks_dir / Path(name).with_suffix(".png"))
        if not any(masks_dir.iterdir()):
            raise ValidationError("no hay polígonos aceptados para generar máscaras")
        (folder / "classes.txt").write_text("\n".join(ls.classes) + "\n", encoding="utf-8")
        return folder

    def _apply_segments(
        self, view: DatasetView, labeled: pl.DataFrame, items: dict[str, LabelItem], work: Path
    ) -> Path:
        """Segmentos → un clip por evento en `<clase>/` (clasificación de audio) + `eventos.csv`."""
        import csv

        import soundfile as sf

        folder = work / "dataset"
        folder.mkdir(parents=True)
        rows: list[dict[str, Any]] = []
        for r in labeled.iter_rows(named=True):
            it = items[_row_id(r["__i__"])]
            if not it.segments:
                continue
            src = view.files_dir / str(r["path"])
            data, sr = sf.read(str(src), dtype="float32", always_2d=True)
            stem = Path(str(r["path"])).stem
            for n, seg in enumerate(it.segments):
                a, b = int(seg.start_s * sr), min(int(seg.end_s * sr), data.shape[0])
                if b - a < 1:
                    continue  # el segmento cae fuera del audio
                rel = Path(seg.label) / f"{stem}_{n:03d}.wav"
                (folder / rel).parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(folder / rel), data[a:b], sr)
                rows.append(
                    {
                        "file_name": str(r["path"]),
                        "inicio_s": seg.start_s,
                        "fin_s": seg.end_s,
                        "etiqueta": seg.label,
                        "clip": rel.as_posix(),
                    }
                )
        if not rows:
            raise ValidationError("no hay segmentos aceptados dentro de los audios")
        # Las anotaciones de eventos quedan junto al dataset (no son audio).
        with (work / "eventos.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return folder
