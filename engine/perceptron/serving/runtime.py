"""Inferencia con un modelo exportado (RF-EXP-02, RF-EXP-03; ADR-0027).

Carga `model.onnx` + `pipeline.json` + `signature.json` de un directorio de export y predice
con ONNX Runtime. El preprocesamiento es el del pipeline ajustado del Engine (el mismo
código que en el entrenamiento), así que no hay desvío entre entrenar y servir. En tabular,
texto y series no necesita torch; en imagen usa las transformaciones de evaluación
(torchvision, CPU) y en audio las mismas features (log-mel/MFCC) normalizadas con las
estadísticas de train.

Visión avanzada: detección (CenterNet) devuelve las cajas relativas a la imagen original,
segmentación la máscara de clases (PNG) y la fracción por clase, y OCR el texto decodificado
(CTC greedy). El decodificado está en numpy, igual al de `tasks.vision` (un test lo compara).

Series: `predict_series` recibe la historia reciente con las columnas del entrenamiento y
aplica el mismo escalado y ventaneo. Forecasting devuelve los próximos `horizon` valores y
anomalías marca los puntos cuyo error de reconstrucción supera el umbral calibrado en
validación (guardado en la firma al exportar).

Embeddings internos (RF-MON-02): la entrada de la última capa lineal (la cabeza) se expone como
salida extra del grafo ONNX al cargarlo, sin volver a exportar. Sirven para medir drift en
imagen, texto y audio.
"""

from __future__ import annotations

import base64
import io
import itertools
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
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
    vision_task_array,
    vision_task_input,
)
from perceptron.data.series import SeriesConfig, history_columns

MODEL_FILE = "model.onnx"
_HEAD_OPS = ("Gemm", "MatMul")

# Detección: puntaje mínimo para contar un objeto si la evaluación no fijó otro.
DEFAULT_SCORE_THRESHOLD = 0.3
# Iguales a `tasks.vision` (candidatos del decodificado de CenterNet).
DECODE_MIN_SCORE = 0.05
MAX_DETECTIONS = 50
# Ventanas por llamada a ONNX Runtime en series (acota la memoria con historias largas).
SERIES_CHUNK = 512
_DAY = 86_400


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
    # Detalle propio de la tarea (cajas, máscara, fechas del pronóstico, puntajes...).
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"prediction": self.prediction}
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.probabilities:
            out["probabilities"] = self.probabilities
        if self.details:
            out["details"] = self.details
        return out


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return out


def _max_pool3(x: np.ndarray) -> np.ndarray:
    """Max pooling 3×3 con paso 1 y relleno 1 sobre las dos últimas dimensiones."""
    h, w = x.shape[-2:]
    pad = [(0, 0)] * (x.ndim - 2) + [(1, 1), (1, 1)]
    p = np.pad(x, pad, constant_values=-np.inf)
    out: np.ndarray = np.max(
        np.stack([p[..., i : i + h, j : j + w] for i in range(3) for j in range(3)]), axis=0
    )
    return out


def decode_detections(
    out: np.ndarray, k: int, image_size: int
) -> list[list[tuple[int, float, list[float]]]]:
    """CenterNet [B, K+4, h, w] → por imagen, (clase, puntaje, caja xyxy en píxeles de entrada).

    Mismo algoritmo que `tasks.vision.decode_centernet`: picos locales del heatmap, los
    `MAX_DETECTIONS` mejores con puntaje > `DECODE_MIN_SCORE`, tamaño y desplazamiento regresados.
    """
    b, _, h, w = out.shape
    stride = image_size / w
    heat = _sigmoid(out[:, :k].astype(np.float64))
    peaks = heat * (heat == _max_pool3(heat))
    results = []
    for i in range(b):
        flat = peaks[i].reshape(-1)
        n = min(MAX_DETECTIONS, flat.size)
        idx = np.argsort(-flat, kind="stable")[:n]
        idx = idx[flat[idx] > DECODE_MIN_SCORE]
        labels, pix = idx // (h * w), idx % (h * w)
        ys, xs = (pix // w).astype(np.float64), (pix % w).astype(np.float64)
        reg = out[i, k:].reshape(4, -1)[:, pix].astype(np.float64)
        bw, bh = np.clip(reg[0], 0, None), np.clip(reg[1], 0, None)
        cx, cy = (xs + reg[2]) * stride, (ys + reg[3]) * stride
        boxes = np.stack(
            [
                cx - bw * stride / 2,
                cy - bh * stride / 2,
                cx + bw * stride / 2,
                cy + bh * stride / 2,
            ],
            axis=1,
        ).clip(0, image_size)
        results.append(
            [
                (int(lbl), float(flat[j]), [float(v) for v in box])
                for lbl, j, box in zip(labels, idx, boxes, strict=True)
            ]
        )
    return results


def greedy_ctc_texts(logits: np.ndarray, alphabet: list[str]) -> list[str]:
    """CTC greedy sobre [B, T, V]: colapsa repetidos y quita el blank (índice 0)."""
    texts = []
    for row in logits.argmax(-1).tolist():
        chars, prev = [], 0
        for i in row:
            if i not in (prev, 0) and i - 1 < len(alphabet):
                chars.append(alphabet[i - 1])
            prev = i
        texts.append("".join(chars))
    return texts


def _png_base64(mask: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _as_datetime(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day)
    raise TypeError(f"fecha inválida: {v!r}")


def _iso(t: datetime, date_only: bool) -> str:
    return t.date().isoformat() if date_only else t.isoformat()


def _number(column: str, v: Any) -> float | None:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, int | float):
        return float(v)
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        msg = f"la columna «{column}» tiene un valor que no es un número: {v!r}"
        raise InputError(msg) from None


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
    def series_config(self) -> SeriesConfig | None:
        spec = self.pipeline.spec.series
        return spec.config if spec is not None else None

    @property
    def required_columns(self) -> list[str]:
        """Las columnas que el modelo usa. Las que la preparación descarta (identificadores,
        columnas constantes) no hacen falta: se completan vacías. En series, las de la
        historia reciente (tiempo, serie, target y covariables)."""
        cfg = self.series_config
        if self.kind == "sequence" and cfg is not None:
            return history_columns(cfg)
        inputs = self.signature["inputs"]
        cols: list[str] = inputs.get("columns", [])
        used = {*inputs.get("numeric_features", []), *inputs.get("categorical_features", [])}
        if not used:
            return cols
        return [c for c in cols if c in used or any(u.startswith(f"{c}.") for u in used)]

    # ---------------------------------------------------------------- salida

    def _postprocess(
        self, out: np.ndarray, sizes: list[tuple[int, int]] | None = None
    ) -> list[Prediction]:
        if self.task == "object_detection":
            return self._detections(out)
        if self.task == "segmentation":
            return self._segmentation(out, sizes)
        if self.task == "ocr":
            return [Prediction(prediction=t) for t in greedy_ctc_texts(out, self._classes())]
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

    def _classes(self) -> list[str]:
        return list(self.pipeline.classes or [])

    def _image_size(self) -> int:
        shape = self.signature["inputs"].get("shape") or []
        if shape:
            return int(shape[-1])
        vision = vision_task_input(self.pipeline)
        return vision[0][1] if vision else 224

    def _detections(self, out: np.ndarray) -> list[Prediction]:
        """Objetos con puntaje ≥ el umbral de la firma; cajas relativas (0..1) a la imagen.

        La imagen se redimensiona sin conservar la proporción, así que la caja relativa a la
        entrada del modelo es también relativa a la imagen original.
        """
        classes = self._classes()
        size = self._image_size()
        stored = self.signature["outputs"].get("score_threshold")
        threshold = float(stored) if stored is not None else DEFAULT_SCORE_THRESHOLD
        preds = []
        for dets in decode_detections(out, out.shape[1] - 4, size):
            objects = [
                {
                    "label": classes[lbl] if lbl < len(classes) else str(lbl),
                    "score": round(score, 4),
                    "box": [round(min(max(v / size, 0.0), 1.0), 4) for v in box],
                }
                for lbl, score, box in dets
                if score >= threshold
            ]
            preds.append(Prediction(prediction=len(objects), details={"objects": objects}))
        return preds

    def _segmentation(
        self, out: np.ndarray, sizes: list[tuple[int, int]] | None
    ) -> list[Prediction]:
        """Porcentaje de la imagen cubierto por clases que no son el fondo (índice 0)."""
        from PIL import Image

        k = out.shape[1]
        names = self._classes()
        labels = [names[i] if i < len(names) else str(i) for i in range(k)]
        preds = []
        for i, small in enumerate(out.argmax(1).astype(np.uint8)):
            mask = small
            if sizes is not None and i < len(sizes):
                resized = Image.fromarray(small).resize(sizes[i], Image.Resampling.NEAREST)
                mask = np.asarray(resized, dtype=np.uint8)
            counts = np.bincount(mask.reshape(-1), minlength=k)[:k]
            total = max(int(mask.size), 1)
            fraction = {c: round(float(n) / total, 6) for c, n in zip(labels, counts, strict=True)}
            covered = 100.0 * (1.0 - float(counts[0]) / total)
            preds.append(
                Prediction(
                    prediction=round(covered, 4),
                    details={"fraction": fraction, "mask_png": _png_base64(mask)},
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
        self,
        feed: dict[str, np.ndarray],
        *,
        embeddings: bool = False,
        sizes: list[tuple[int, int]] | None = None,
    ) -> tuple[list[Prediction], np.ndarray | None]:
        """Predicciones y, si se piden y el grafo lo permite, embeddings [N, D] de la cabeza.

        `sizes` (ancho, alto) de las imágenes originales: la máscara de segmentación se
        devuelve en ese tamaño.
        """
        emb = self._embedding_session() if embeddings else None
        if emb is None:
            return self._postprocess(self._run(feed), sizes), None
        session, names = emb
        out, vec = session.run(names, feed)
        vec = np.asarray(vec, dtype=np.float32)
        preds = self._postprocess(np.asarray(out, dtype=np.float32), sizes)
        return preds, vec.reshape(len(vec), -1)

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
        columns = self.signature["inputs"].get("columns", [])
        clean = [
            {**dict.fromkeys(columns), **{k: (None if v == "" else v) for k, v in r.items()}}
            for r in rows
        ]
        df = pl.DataFrame(clean, infer_schema_length=None)
        arrays = transform_tabular(self.pipeline, df)
        feed = {"x_num": arrays.x_num, "x_cat": arrays.x_cat}
        return {k: v for k, v in feed.items() if k in self.input_names}

    def predict_images(self, images: list[Any]) -> list[Prediction]:
        """Imagen: objetos `PIL.Image` (se aplican las transformaciones de evaluación)."""
        if not images:
            return []
        sizes = [(int(im.size[0]), int(im.size[1])) for im in images]
        return self.infer(self.feed_images(images), sizes=sizes)[0]

    def feed_images(self, images: list[Any]) -> dict[str, np.ndarray]:
        if self.kind != "image":
            raise InputError(f"el modelo espera {self.kind}, no imágenes")
        vision = vision_task_input(self.pipeline)
        if vision is not None:  # detección, segmentación, OCR: sin torch
            size, mode = vision
            batch = np.stack([vision_task_array(self.pipeline, im, size, mode) for im in images])
            return {self.input_names[0]: batch.astype(np.float32)}
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

    # ---------------------------------------------------------------- series

    def predict_series(self, rows: list[dict[str, Any]]) -> list[Prediction]:
        """Series: la historia reciente (mismas columnas que en el entrenamiento).

        Una predicción por serie. Forecasting: los próximos `horizon` valores en la escala
        original. Anomalías: cuántos puntos superan el umbral calibrado, con sus puntajes.
        """
        cfg = self.series_config
        if self.kind != "sequence" or cfg is None or self.pipeline.series_state is None:
            raise InputError(f"el modelo espera {self.kind}, no la historia de una serie")
        if not rows:
            return []
        groups = self._series_groups(cfg, self._history(cfg, rows))
        if self.task == "anomaly_detection":
            return [self._anomalies(cfg, *g) for g in groups]
        return self._forecast(cfg, groups)

    def _history(self, cfg: SeriesConfig, rows: list[dict[str, Any]]) -> pl.DataFrame:
        cols = history_columns(cfg)
        missing = sorted({c for c in cols if any(c not in r for r in rows)})
        if missing:
            raise InputError(f"faltan columnas: {', '.join(missing)}")
        raw_times = [r[cfg.time_column] for r in rows]
        texts = [
            None
            if v is None or v == ""
            else (v.isoformat() if isinstance(v, date | datetime) else str(v).strip())
            for v in raw_times
        ]
        if any(t is None for t in texts):
            raise InputError(f"hay filas sin fecha en la columna «{cfg.time_column}»")
        try:
            times = pl.Series(cfg.time_column, texts, dtype=pl.String).str.to_datetime(strict=False)
        except Exception:  # formatos mezclados que Polars no puede inferir
            times = pl.Series(cfg.time_column, [None] * len(texts), dtype=pl.Datetime)
        bad = [t for t, parsed in zip(texts, times.to_list(), strict=True) if parsed is None]
        if bad:
            raise InputError(
                f"no se pudo leer la fecha {bad[0]!r} de la columna «{cfg.time_column}»"
            )
        if getattr(times.dtype, "time_zone", None):
            times = times.dt.replace_time_zone(None)
        data: dict[str, pl.Series] = {cfg.time_column: times}
        if cfg.series_id:
            ids = [None if r[cfg.series_id] is None else str(r[cfg.series_id]) for r in rows]
            data[cfg.series_id] = pl.Series(cfg.series_id, ids, dtype=pl.String)
        for c in cols:
            if c not in data:
                data[c] = pl.Series(c, [_number(c, r[c]) for r in rows], dtype=pl.Float64)
        return pl.DataFrame(data)

    def _series_groups(
        self, cfg: SeriesConfig, df: pl.DataFrame
    ) -> list[tuple[str | None, list[datetime], np.ndarray, np.ndarray, np.ndarray]]:
        """Por serie: nombre, tiempos, features escaladas [n, C] y media/desvío por canal."""
        from perceptron.data.pipeline.series_windows import prepare_series

        sspec = self.pipeline.spec.series
        state = self.pipeline.series_state or {}
        calendar = sspec.calendar if sspec is not None else True
        groups = []
        for key, part, _values, feats, mean, std in prepare_series(cfg, state, df, calendar):
            name = key if cfg.series_id else None
            if part.height < cfg.lookback:
                who = f"la serie «{name}» tiene" if name is not None else "llegaron"
                raise InputError(
                    f"Hacen falta al menos {cfg.lookback} valores recientes por serie para "
                    f"predecir; {who} {part.height}."
                )
            times = [_as_datetime(t) for t in part[cfg.time_column].to_list()]
            groups.append((name, times, feats, mean, std))
        return groups

    def _run_chunks(self, x: np.ndarray) -> np.ndarray:
        name = self.input_names[0]
        outs = [
            self._run({name: x[i : i + SERIES_CHUNK].astype(np.float32)})
            for i in range(0, len(x), SERIES_CHUNK)
        ]
        return np.concatenate(outs)

    @staticmethod
    def _date_only(times: list[datetime], freq: float | None) -> bool:
        midnight = all(t.hour == t.minute == t.second == t.microsecond == 0 for t in times)
        return midnight and (freq is None or freq % _DAY == 0)

    @staticmethod
    def _freq(cfg: SeriesConfig, times: list[datetime]) -> float | None:
        if cfg.freq_seconds:
            return float(cfg.freq_seconds)
        diffs = [(b - a).total_seconds() for a, b in itertools.pairwise(times)]
        positive = [d for d in diffs if d > 0]
        return float(np.median(positive)) if positive else None

    def _forecast(
        self,
        cfg: SeriesConfig,
        groups: list[tuple[str | None, list[datetime], np.ndarray, np.ndarray, np.ndarray]],
    ) -> list[Prediction]:
        lookback, horizon = cfg.lookback, cfg.horizon
        windows, levels = [], []
        for _name, _times, feats, _mean, _std in groups:
            # Igual que en el entrenamiento: relativo al último valor observado (RevIN).
            x = feats[-lookback:].copy()
            level = float(x[-1, 0])
            x[:, 0] -= level
            windows.append(x)
            levels.append(level)
        out = self._run_chunks(np.stack(windows)).reshape(len(windows), -1)[:, :horizon]
        preds = []
        for (name, times, _feats, mean, std), level, row in zip(groups, levels, out, strict=True):
            values = (row + level) * float(std[0]) + float(mean[0])
            freq = self._freq(cfg, times)
            last = times[-1]
            date_only = self._date_only(times, freq)
            future = (
                [_iso(last + timedelta(seconds=freq * (h + 1)), date_only) for h in range(horizon)]
                if freq
                else []
            )
            preds.append(
                Prediction(
                    prediction=[float(v) for v in values],
                    details={"series": name, "times": future, "last_time": _iso(last, date_only)},
                )
            )
        return preds

    def _anomalies(
        self,
        cfg: SeriesConfig,
        name: str | None,
        times: list[datetime],
        feats: np.ndarray,
        _mean: np.ndarray,
        _std: np.ndarray,
    ) -> Prediction:
        lookback = cfg.lookback
        ends = range(lookback - 1, len(feats))
        x = np.stack([feats[e - lookback + 1 : e + 1] for e in ends])
        rec = self._run_chunks(x)
        # Como en la calibración: el error del último punto de cada ventana.
        scores = ((rec[:, -1] - x[:, -1]) ** 2).mean(axis=1)
        stored = self.signature["outputs"].get("threshold")
        # Sin umbral guardado (export viejo): el cuantil 0,99 de los puntajes recibidos.
        threshold = float(stored) if stored is not None else float(np.quantile(scores, 0.99))
        point_times = [times[e] for e in ends]
        date_only = self._date_only(times, self._freq(cfg, times))
        iso = [_iso(t, date_only) for t in point_times]
        flagged = [t for t, s in zip(iso, scores, strict=True) if s >= threshold]
        return Prediction(
            prediction=len(flagged),
            details={
                "series": name,
                "threshold": threshold,
                "times": iso,
                "scores": [round(float(s), 6) for s in scores],
                "anomalies": flagged,
            },
        )
