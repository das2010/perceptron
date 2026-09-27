"""Ingesta → `DatasetVersion` inmutable y content-addressed (RF-ING-07, ADR-0016)."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from perceptron.core.errors import ValidationError
from perceptron.core.ids import IdPrefix, new_id
from perceptron.core.paths import ProjectPaths
from perceptron.data.schema import SemanticType, TableSchema, infer_schema
from perceptron.data.sources.files import (
    SourceKind,
    image_folder_index,
    open_source,
    scan_table,
    text_folder_table,
    write_table_parquet,
)
from perceptron.data.splits import SplitRequest, assign_splits, split_counts
from perceptron.data.view import (
    FILES_DIR,
    INDEX_FILE,
    MANIFEST_FILE,
    META_FILE,
    SCHEMA_FILE,
    TABLE_FILE,
)
from perceptron.domain.enums import Modality, SplitStrategy
from perceptron.domain.models import DatasetVersion, Split
from perceptron.storage.filesystem import write_json
from perceptron.storage.manifest import build_manifest

logger = logging.getLogger(__name__)

IMAGE_TARGET = "label"


@dataclass(frozen=True)
class IngestRequest:
    project_id: str
    source: Path
    target: str | None = None
    split: SplitRequest | None = None
    overrides: dict[str, SemanticType] | None = None
    source_id: str | None = None
    modality: Modality | None = None


def _materialize_table(
    src: Path, staging: Path, req: IngestRequest
) -> tuple[TableSchema, pl.DataFrame]:
    raw = staging / "_raw.parquet"
    write_table_parquet(scan_table(src), raw)
    df = pl.read_parquet(raw)
    raw.unlink()
    if df.height == 0:
        raise ValidationError("el archivo no tiene filas")
    schema = infer_schema(df, target=req.target)
    if req.overrides:
        schema = schema.with_overrides(req.overrides)
    if schema.target is None and schema.target_candidates:
        schema = schema.model_copy(update={"target": schema.target_candidates[0]})
    return schema, df


def _materialize_images(src: Path, staging: Path) -> tuple[TableSchema, pl.DataFrame]:
    df = image_folder_index(src, staging / FILES_DIR)
    if df.height == 0:
        raise ValidationError("no se encontraron imágenes")
    has_labels = df["label"].null_count() < df.height
    schema = infer_schema(df.select("path", "label"), target=IMAGE_TARGET if has_labels else None)
    schema = schema.with_overrides(
        {"path": SemanticType.FILEPATH, "label": SemanticType.CATEGORICAL}
    )
    return schema, df


MAX_TEXT_SIDE_COLUMNS = 2


def text_column(schema: TableSchema) -> str | None:
    """La columna de texto principal si la tabla es 'texto + pocas columnas más'."""
    texts = [
        c.name
        for c in schema.columns
        if c.semantic is SemanticType.TEXT and c.name != schema.target
    ]
    others = [
        c
        for c in schema.columns
        if c.name != schema.target and c.semantic not in (SemanticType.ID, SemanticType.TEXT)
    ]
    if len(texts) == 1 and len(others) <= MAX_TEXT_SIDE_COLUMNS:
        return texts[0]
    return None


def _materialize_text_folder(src: Path, req: IngestRequest) -> tuple[TableSchema, pl.DataFrame]:
    df = text_folder_table(src)
    if df.height == 0:
        raise ValidationError("no se encontraron archivos .txt")
    has_labels = df["label"].null_count() < df.height
    schema = infer_schema(df, target="label" if has_labels else None).with_overrides(
        {"path": SemanticType.ID, "text": SemanticType.TEXT, "label": SemanticType.CATEGORICAL}
    )
    return schema, df


def _default_split(schema: TableSchema, modality: Modality) -> SplitRequest:
    """Estratificado si hay target categórico; temporal si hay una columna de fecha."""
    if modality is Modality.TABULAR:
        dates = [c.name for c in schema.columns if c.semantic is SemanticType.DATETIME]
        target_col = schema.column(schema.target) if schema.target else None
        if target_col and target_col.semantic is SemanticType.NUMERIC and dates:
            return SplitRequest(strategy=SplitStrategy.TEMPORAL, time_column=dates[0])
    return SplitRequest(strategy=SplitStrategy.STRATIFIED)


def ingest(paths: ProjectPaths, req: IngestRequest) -> DatasetVersion:
    """Materializa la fuente en `datasets/<hash>/` y devuelve la versión (sin persistirla).

    Si ya existe una versión con el mismo hash se reutiliza su directorio.
    """
    paths.ensure()
    staging = paths.datasets_dir / f".staging-{new_id(IdPrefix.DATASET_VERSION)}"
    staging.mkdir(parents=True)
    try:
        extra_meta: dict[str, object] = {}
        with open_source(req.source) as detected:
            if detected.kind is SourceKind.TABLE:
                schema, df = _materialize_table(detected.path, staging, req)
                text_col = text_column(schema)
                modality = req.modality or (Modality.TEXT if text_col else Modality.TABULAR)
                if modality is Modality.TEXT:
                    text_col = text_col or next(
                        (c.name for c in schema.columns if c.semantic is SemanticType.TEXT), None
                    )
                    if text_col is None:
                        raise ValidationError("modalidad texto sin ninguna columna de texto")
                    extra_meta["text_column"] = text_col
                data_file = TABLE_FILE
            elif detected.kind is SourceKind.TEXT_FOLDER:
                modality = Modality.TEXT
                schema, df = _materialize_text_folder(detected.path, req)
                extra_meta["text_column"] = "text"
                data_file = TABLE_FILE
            else:
                modality = Modality.IMAGE
                schema, df = _materialize_images(detected.path, staging)
                data_file = INDEX_FILE

        split_req = req.split or _default_split(schema, modality)
        df = assign_splits(df, split_req, schema.target)
        df.write_parquet(staging / data_file, compression="zstd", statistics=True)
        write_json(staging / SCHEMA_FILE, schema.model_dump(mode="json"))
        write_json(
            staging / META_FILE,
            {"modality": modality.value, "split": split_req.model_dump(mode="json"), **extra_meta},
        )

        manifest = build_manifest(staging)
        content_hash = manifest.content_hash
        final = paths.dataset(content_hash)
        if final.exists():
            logger.info("dataset ya existente, se reutiliza", extra={"content_hash": content_hash})
            shutil.rmtree(staging)
        else:
            write_json(staging / MANIFEST_FILE, manifest.to_dict())
            staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    counts = split_counts(df)
    return DatasetVersion(
        project_id=req.project_id,
        source_id=req.source_id,
        content_hash=content_hash,
        modality=modality,
        target=schema.target,
        path=final.relative_to(paths.root).as_posix(),
        num_samples=df.height,
        size_bytes=manifest.total_size,
        split=Split(
            strategy=split_req.strategy,
            train=counts["train"],
            val=counts["val"],
            test=counts["test"],
            folds=split_req.folds,
            seed=split_req.seed,
            group_column=split_req.group_column,
            time_column=split_req.time_column,
        ),
    )


def read_schema(dataset_dir: Path) -> TableSchema:
    return TableSchema.model_validate(json.loads((dataset_dir / SCHEMA_FILE).read_text("utf-8")))
