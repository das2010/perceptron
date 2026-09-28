"""Registro de predicciones y de feedback de un deployment (RF-MON-01).

Parquet por lote bajo `projects/<id>/monitoring/<deployment>/`. Se guardan solo las features
que usa el modelo (columnas de la firma), con prefijo `x:`, más la clave de negocio si el
deployment la configuró para asociar el feedback. Nada de otras columnas del request.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from perceptron.domain.models import utcnow

FEATURE_PREFIX = "x:"


class PredictionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.predictions_dir = root / "predictions"
        self.feedback_dir = root / "feedback"

    @staticmethod
    def _write(folder: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        folder.mkdir(parents=True, exist_ok=True)
        stamp = utcnow().strftime("%Y%m%dT%H%M%S%f")
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(
            folder / f"{stamp}-{secrets.token_hex(4)}.parquet"
        )

    @staticmethod
    def _read(folder: Path) -> pl.DataFrame:
        parts = sorted(folder.glob("*.parquet")) if folder.is_dir() else []
        if not parts:
            return pl.DataFrame()
        return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")

    # ------------------------------------------------------------------ escritura

    def log_predictions(
        self,
        model_version_id: str,
        features: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        keys: list[str | None],
        ids: list[str],
    ) -> None:
        now = utcnow()
        rows = []
        for pid, feats, out, key in zip(ids, features, outputs, keys, strict=True):
            row: dict[str, Any] = {
                "prediction_id": pid,
                "ts": now,
                "model_version_id": model_version_id,
                "key": key,
                "prediction": str(out.get("prediction")),
                "confidence": out.get("confidence"),
                "probabilities": json.dumps(out.get("probabilities") or {}),
            }
            row.update({f"{FEATURE_PREFIX}{k}": v for k, v in feats.items()})
            rows.append(row)
        self._write(self.predictions_dir, rows)

    def log_feedback(self, items: list[dict[str, Any]]) -> None:
        now = utcnow()
        self._write(
            self.feedback_dir,
            [
                {
                    "prediction_id": i.get("prediction_id"),
                    "key": i.get("key"),
                    "label": str(i["label"]),
                    "ts": now,
                }
                for i in items
            ],
        )

    # ------------------------------------------------------------------ lectura

    def predictions(
        self, *, last: int | None = None, since: datetime | None = None
    ) -> pl.DataFrame:
        df = self._read(self.predictions_dir)
        if df.is_empty():
            return df
        df = df.sort("ts")
        if since is not None:
            df = df.filter(pl.col("ts") >= since)
        if last is not None:
            df = df.tail(last)
        return df

    def count(self) -> int:
        return self.predictions().height

    def features(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
        return df.select(cols).rename({c: c[len(FEATURE_PREFIX) :] for c in cols})

    def labeled(self, *, last: int | None = None) -> pl.DataFrame:
        """Predicciones con su etiqueta real (por prediction_id o, si no, por clave)."""
        preds = self.predictions()
        fb = self._read(self.feedback_dir)
        if preds.is_empty() or fb.is_empty():
            return pl.DataFrame()
        fb = fb.sort("ts").unique(subset=["prediction_id", "key"], keep="last")
        by_id = preds.join(
            fb.filter(pl.col("prediction_id").is_not_null()).select("prediction_id", "label"),
            on="prediction_id",
            how="inner",
        )
        by_key = pl.DataFrame()
        if "key" in fb.columns and fb["key"].drop_nulls().len():
            keyed = fb.filter(pl.col("key").is_not_null()).select(
                pl.col("key").cast(pl.Utf8), "label"
            )
            by_key = (
                preds.filter(
                    ~pl.col("prediction_id").is_in(by_id["prediction_id"].to_list())
                    & pl.col("key").is_not_null()
                )
                .with_columns(pl.col("key").cast(pl.Utf8))
                .join(keyed, on="key", how="inner")
            )
        out = pl.concat([by_id, by_key], how="diagonal_relaxed") if by_key.height else by_id
        out = out.sort("ts")
        return out.tail(last) if last is not None else out
