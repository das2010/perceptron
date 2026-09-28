"""Inferencia con un modelo exportado (RF-EXP-02, RF-EXP-03; ADR-0027).

Carga `model.onnx` + `pipeline.json` + `signature.json` de un directorio de export y predice
con ONNX Runtime. El preprocesamiento es el del pipeline ajustado del Engine (el mismo
código que en el entrenamiento), así que no hay desvío entre entrenar y servir. En tabular
y texto no necesita torch; en imagen usa las transformaciones de evaluación (torchvision,
CPU) y en audio las mismas features (log-mel/MFCC) normalizadas con las estadísticas de train.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from perceptron.data.pipeline.pipeline import (
    FittedPipeline,
    audio_features,
    decode_regression,
    image_transforms,
    transform_tabular,
    transform_text,
)

MODEL_FILE = "model.onnx"


class InputError(ValueError):
    """La entrada no respeta la firma del modelo."""


@dataclass
class Prediction:
    prediction: Any
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"prediction": self.prediction}
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.probabilities:
            out["probabilities"] = self.probabilities
        return out


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


class InferenceModel:
    """Modelo exportado listo para predecir (sin Lightning ni el resto del Engine)."""

    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort

        self.dir = model_dir
        self.signature: dict[str, Any] = json.loads(
            (model_dir / "signature.json").read_text(encoding="utf-8")
        )
        self.pipeline = FittedPipeline.model_validate_json(
            (model_dir / "pipeline.json").read_text(encoding="utf-8")
        )
        self.session = ort.InferenceSession(
            str(model_dir / MODEL_FILE), providers=ort.get_available_providers()
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    @property
    def kind(self) -> str:
        return str(self.signature["inputs"]["kind"])

    @property
    def task(self) -> str:
        return str(self.signature["outputs"]["task"])

    @property
    def required_columns(self) -> list[str]:
        cols: list[str] = self.signature["inputs"].get("columns", [])
        return cols

    # ---------------------------------------------------------------- salida

    def _postprocess(self, out: np.ndarray) -> list[Prediction]:
        if self.task == "regression":
            y = decode_regression(self.pipeline, out.reshape(len(out), -1)[:, 0])
            return [Prediction(prediction=float(v)) for v in y]
        classes = self.pipeline.classes or [str(i) for i in range(max(out.shape[1], 2))]
        if out.ndim == 1 or out.shape[1] == 1:  # binaria con una sola salida (BCE)
            p1 = 1.0 / (1.0 + np.exp(-out.reshape(-1)))
            proba = np.stack([1.0 - p1, p1], axis=1)
        else:
            proba = _softmax(out.astype(np.float64))
        preds = []
        for row in proba:
            k = int(row.argmax())
            preds.append(
                Prediction(
                    prediction=classes[k],
                    confidence=float(row[k]),
                    probabilities={c: float(p) for c, p in zip(classes, row, strict=False)},
                )
            )
        return preds

    def _run(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(self.session.run(None, feed)[0], dtype=np.float32)

    # ---------------------------------------------------------------- entradas

    def predict_rows(self, rows: list[dict[str, Any]]) -> list[Prediction]:
        """Tabular: filas como dicts con las columnas originales del dataset."""
        if self.kind != "tabular":
            raise InputError(f"el modelo espera {self.kind}, no filas de tabla")
        if not rows:
            return []
        missing = sorted({c for c in self.required_columns if any(c not in r for r in rows)})
        if missing:
            raise InputError(f"faltan columnas: {', '.join(missing)}")
        # Celdas vacías (CSV, formularios) son valores faltantes, no texto: el pipeline las imputa.
        clean = [{k: (None if v == "" else v) for k, v in r.items()} for r in rows]
        df = pl.DataFrame(clean, infer_schema_length=None)
        arrays = transform_tabular(self.pipeline, df)
        feed = {"x_num": arrays.x_num, "x_cat": arrays.x_cat}
        return self._postprocess(
            self._run({k: v for k, v in feed.items() if k in self.input_names})
        )

    def predict_images(self, images: list[Any]) -> list[Prediction]:
        """Imagen: objetos `PIL.Image` (se aplican las transformaciones de evaluación)."""
        if self.kind != "image":
            raise InputError(f"el modelo espera {self.kind}, no imágenes")
        import torch

        tf = image_transforms(self.pipeline, train=False)
        spec = self.pipeline.spec.image
        mode = "L" if spec is not None and spec.channels == 1 else "RGB"
        batch = torch.stack([tf(im.convert(mode)) for im in images]).numpy()
        return self._postprocess(self._run({self.input_names[0]: batch.astype(np.float32)}))

    def predict_texts(self, texts: list[str]) -> list[Prediction]:
        """Texto: se normaliza y tokeniza igual que en el entrenamiento."""
        if self.kind != "tokens":
            raise InputError(f"el modelo espera {self.kind}, no texto")
        spec = self.pipeline.spec.text
        if spec is None:
            raise InputError("el modelo no tiene pipeline de texto")
        if not texts:
            return []
        ids = transform_text(self.pipeline, pl.DataFrame({spec.column: texts}))
        return self._postprocess(self._run({self.input_names[0]: ids}))

    def predict_audio(self, files: list[Path]) -> list[Prediction]:
        """Audio (WAV, FLAC, OGG, MP3): se remuestrea y se calculan las features de train."""
        if self.kind != "spectrogram":
            raise InputError(f"el modelo espera {self.kind}, no audio")
        from perceptron.data.audio import load

        spec = self.pipeline.spec.audio
        if spec is None:
            raise InputError("el modelo no tiene pipeline de audio")
        mean = self.pipeline.audio_mean or 0.0
        std = self.pipeline.audio_std or 1.0
        feats = []
        for path in files:
            try:
                wav, _ = load(path, sample_rate=spec.sample_rate)
            except Exception as exc:  # formato no soportado o archivo roto
                raise InputError(f"no se pudo leer el audio: {exc}") from exc
            feats.append(((audio_features(spec, wav) - mean) / std).numpy())
        batch = np.stack(feats).astype(np.float32)
        return self._postprocess(self._run({self.input_names[0]: batch}))
