"""Profiling de una `DatasetVersion` → `ProfileCard` (RF-PRF-01, 02, 06, 07).

Se perfilan solo train + val: el test sellado no se inspecciona.
"""

from __future__ import annotations

import polars as pl

from perceptron.data.profiling.alerts import (
    column_alerts,
    duplicate_alert,
    image_alerts,
    target_alerts,
)
from perceptron.data.profiling.audio import audio_alerts, profile_audio
from perceptron.data.profiling.card import ProfileCard
from perceptron.data.profiling.images import profile_images
from perceptron.data.profiling.tabular import profile_columns, profile_target
from perceptron.data.profiling.text import profile_text, text_alerts
from perceptron.data.schema import SemanticType
from perceptron.data.series import profile_series
from perceptron.data.splits import FOLD_COLUMN, SPLIT_COLUMN, TEST
from perceptron.data.view import DatasetView, Purpose
from perceptron.data.vision_tasks import profile_vision_task
from perceptron.domain.enums import Modality

_INTERNAL = {SPLIT_COLUMN, FOLD_COLUMN}


def profile_dataset(
    view: DatasetView, *, dataset_version_id: str | None = None, content_hash: str | None = None
) -> ProfileCard:
    df = view.read(purpose=Purpose.PROFILING)
    counts = dict(
        pl.scan_parquet(view.data_file).group_by(SPLIT_COLUMN).len().collect().iter_rows()
    )
    total = int(sum(counts.values()))

    schema = view.schema
    if view.task is not None:
        feature_df = df.select("path")
    elif view.modality in (Modality.IMAGE, Modality.AUDIO):
        feature_df = df.select("path", "label")
    else:
        feature_df = df.drop([c for c in _INTERNAL if c in df.columns])

    target = profile_target(schema, feature_df) if view.task is None else None
    # Detección, segmentación y OCR: el target es una estructura (cajas, máscaras, texto).
    columns = profile_columns(schema, feature_df) if view.task is None else []
    alerts = target_alerts(target, df.height) if view.task is None else []

    images = None
    dup_fraction: float | None = None
    audio = None
    if view.modality is Modality.IMAGE:
        images = profile_images(df, view.files_dir)
        alerts += image_alerts(images)
        num_features = 1
    elif view.modality is Modality.AUDIO:
        audio = profile_audio(df, view.files_dir)
        alerts += audio_alerts(audio)
        num_features = 1
    else:
        alerts += column_alerts(schema, columns, feature_df)
        non_id = [c.name for c in schema.columns if c.semantic is not SemanticType.ID]
        rows = feature_df.select(non_id)
        dup_fraction = round(1 - rows.unique().height / max(rows.height, 1), 4)
        alerts += duplicate_alert(dup_fraction)
        num_features = len(schema.feature_columns)
    series = None
    if view.modality is Modality.TIMESERIES and view.series:
        series = profile_series(feature_df, view.series)
    text = None
    if view.modality is Modality.TEXT and view.text_column:
        text = profile_text(feature_df, view.text_column)
        alerts += text_alerts(text)

    return ProfileCard(
        dataset_version_id=dataset_version_id,
        content_hash=content_hash,
        modality=view.modality,
        num_samples=total,
        profiled_samples=df.height,
        split_counts={k: int(counts.get(k, 0)) for k in ("train", "val", TEST)},
        num_features=num_features,
        target=target,
        columns=[] if view.modality in (Modality.IMAGE, Modality.AUDIO) else columns,
        images=images,
        text=text,
        audio=audio,
        vision_task=profile_vision_task(view.task, df, view.files_dir, view.classes)
        if view.task is not None
        else None,
        series=series,
        duplicate_row_fraction=dup_fraction,
        alerts=alerts,
    )
