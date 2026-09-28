"""Inferencia con un modelo exportado (RF-EXP-02, RF-EXP-03; ADR-0027).

Carga `model.onnx` + `pipeline.json` + `signature.json` de un directorio de export y predice
con ONNX Runtime. El preprocesamiento es el del pipeline ajustado del Engine (el mismo
código que en el entrenamiento), así que no hay desvío entre entrenar y servir. En tabular
y texto no necesita torch; en imagen usa las transformaciones de evaluación (torchvision,
CPU) y en audio las mismas features (log-mel/MFCC) normalizadas con las estadísticas de train.

Embeddings internos (RF-MON-02): la entrada de la última capa lineal (la cabeza) se expone como
salida extra del grafo ONNX al cargarlo, sin volver a exportar. Sirven para medir drift en
imagen, texto y audio.
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
_HEAD_OPS = ("Gemm", "MatMul")


def penultimate_tensor(graph: Any) -> str | None:
    """Nombre del tensor que entra a la cabeza lineal final (o None si no hay una)."""
    producer = {out: node for node in graph.node for out in node.output}
    # Pesos: inicializadores y constantes (según el exportador, la matriz llega de una u otra).
    weights = {init.name for init in graph.initializer} | {
        out for node in graph.node if node.op_type == "Constant" for out in node.output
    }
    current = graph.output[0].name
    for _ in range(64):  # sube desde la salida por activaciones, bias y reshapes
        node = producer.get(current)
        if node is None:
            return None
        acts = [i for i in node.input if i and i not in weights and i in producer]
        if node.op_type in _HEAD_OPS:
            return acts[0] if acts else None
        if not acts:
            return None
        current = acts[0]
    return None


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
        self._embedding: tuple[Any, list[str]] | bool | None = None

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

    def _embedding_session(self) -> tuple[Any, list[str]] | None:
        if self._embedding is None:
            self._embedding = False
            try:
                import onnx
                import onnxruntime as ort

                proto = onnx.load(str(self.dir / MODEL_FILE))
                name = penultimate_tensor(proto.graph)
                if name is not None:
                    out = proto.graph.output[0].name
                    proto.graph.output.append(
                        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
                    )
                    session = ort.InferenceSession(
                        proto.SerializeToString(), providers=ort.get_available_providers()
                    )
                    self._embedding = (session, [out, name])
            except Exception:  # sin embeddings se monitorea solo la salida
                self._embedding = False
        return self._embedding if isinstance(self._embedding, tuple) else None

    def infer(
        self, feed: dict[str, np.ndarray], *, embeddings: bool = False
    ) -> tuple[list[Prediction], np.ndarray | None]:
        """Predicciones y, si se piden y el grafo lo permite, embeddings [N, D] de la cabeza."""
        emb = self._embedding_session() if embeddings else None
        if emb is None:
            return self._postprocess(self._run(feed)), None
        session, names = emb
        out, vec = session.run(names, feed)
        vec = np.asarray(vec, dtype=np.float32)
        return self._postprocess(np.asarray(out, dtype=np.float32)), vec.reshape(len(vec), -1)

    # ---------------------------------------------------------------- entradas

    def predict_rows(self, rows: list[dict[str, Any]]) -> list[Prediction]:
        """Tabular: filas como dicts con las columnas originales del dataset."""
        if not rows:
            return []
        return self.infer(self.feed_rows(rows))[0]

    def feed_rows(self, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if self.kind != "tabular":
            raise InputError(f"el modelo espera {self.kind}, no filas de tabla")
        missing = sorted({c for c in self.required_columns if any(c not in r for r in rows)})
        if missing:
            raise InputError(f"faltan columnas: {', '.join(missing)}")
        # Celdas vacías (CSV, formularios) son valores faltantes, no texto: el pipeline las imputa.
        clean = [{k: (None if v == "" else v) for k, v in r.items()} for r in rows]
        df = pl.DataFrame(clean, infer_schema_length=None)
        arrays = transform_tabular(self.pipeline, df)
        feed = {"x_num": arrays.x_num, "x_cat": arrays.x_cat}
        return {k: v for k, v in feed.items() if k in self.input_names}

    def predict_images(self, images: list[Any]) -> list[Prediction]:
        """Imagen: objetos `PIL.Image` (se aplican las transformaciones de evaluación)."""
        return self.infer(self.feed_images(images))[0]

    def feed_images(self, images: list[Any]) -> dict[str, np.ndarray]:
        if self.kind != "image":
            raise InputError(f"el modelo espera {self.kind}, no imágenes")
        import torch

        tf = image_transforms(self.pipeline, train=False)
        spec = self.pipeline.spec.image
        mode = "L" if spec is not None and spec.channels == 1 else "RGB"
        batch = torch.stack([tf(im.convert(mode)) for im in images]).numpy()
        return {self.input_names[0]: batch.astype(np.float32)}

    @property
    def text_column(self) -> str | None:
        spec = self.pipeline.spec.text
        return spec.column if spec is not None else None

    def predict_texts(self, texts: list[str]) -> list[Prediction]:
        """Texto: se normaliza y tokeniza igual que en el entrenamiento."""
        if not texts:
            return []
        return self.infer(self.feed_texts(texts))[0]

    def feed_texts(self, texts: list[str]) -> dict[str, np.ndarray]:
        if self.kind != "tokens":
            raise InputError(f"el modelo espera {self.kind}, no texto")
        spec = self.pipeline.spec.text
        if spec is None:
            raise InputError("el modelo no tiene pipeline de texto")
        ids = transform_text(self.pipeline, pl.DataFrame({spec.column: texts}))
        return {self.input_names[0]: ids}

    def predict_audio(self, files: list[Path]) -> list[Prediction]:
        """Audio (WAV, FLAC, OGG, MP3): se remuestrea y se calculan las features de train."""
        return self.infer(self.feed_audio(files))[0]

    def feed_audio(self, files: list[Path]) -> dict[str, np.ndarray]:
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
        return {self.input_names[0]: batch}
