"""Pre-etiquetado zero-shot local (RF-LBL-02): sin entrenar, solo con los nombres de clase.

- Imagen: SigLIP (similitud imagen–texto).
- Texto: NLI multilingüe (cada clase es una hipótesis).
- Audio: CLAP (similitud audio–texto).

Los modelos del catálogo tienen licencia apta para uso comercial (se revisan en la auditoría de
licencias) y se descargan a la caché del workspace (RF-TRN-11); sin conexión deben estar
descargados de antemano. El backend es reemplazable (tests, otros runtimes).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from perceptron.core.errors import ValidationError
from perceptron.domain.enums import Modality


class ZeroShotModelInfo(BaseModel):
    model: str
    modality: Modality
    license: str
    commercial_ok: bool
    params_m: float
    languages: list[str]


ZERO_SHOT_MODELS: dict[str, ZeroShotModelInfo] = {
    z.model: z
    for z in [
        ZeroShotModelInfo(
            model="google/siglip-base-patch16-224",
            modality=Modality.IMAGE,
            license="Apache-2.0",
            commercial_ok=True,
            params_m=203,
            languages=["en"],
        ),
        ZeroShotModelInfo(
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            modality=Modality.TEXT,
            license="MIT",
            commercial_ok=True,
            params_m=279,
            languages=["multi"],
        ),
        ZeroShotModelInfo(
            model="laion/clap-htsat-unfused",
            modality=Modality.AUDIO,
            license="Apache-2.0",
            commercial_ok=True,
            params_m=153,
            languages=["en"],
        ),
    ]
}
DEFAULT_ZERO_SHOT: dict[Modality, str] = {z.modality: z.model for z in ZERO_SHOT_MODELS.values()}
# Plantillas de hipótesis: el NLI multilingüe entiende español; SigLIP y CLAP, inglés.
TEMPLATES: dict[Modality, str] = {
    Modality.TEXT: "Este texto trata sobre {}.",
    Modality.IMAGE: "This is a photo of {}.",
    Modality.AUDIO: "This is the sound of {}.",
}

# (modelo, modalidad, entradas, clases, multi-etiqueta) → puntaje por clase de cada entrada.
Backend = Callable[[str, Modality, Sequence[Any], list[str], bool], list[dict[str, float]]]


def _pipeline(task: str, model: str) -> Any:
    # Sin conexión no hace falta nada acá: `weights_cache.configure` ya fijó HF_HUB_OFFLINE.
    # (`model_kwargs={"local_files_only": …}` choca con el mismo argumento que transformers le
    # pasa a AutoConfig y rompía la carga.)
    try:
        from transformers import pipeline
    except ImportError:
        raise ValidationError("el zero-shot necesita el extra `ml` (transformers)") from None
    factory: Any = pipeline  # la tarea llega como str: sin sobrecarga tipada
    try:
        return factory(task, model=model)
    except OSError as e:
        raise ValidationError(
            f"el modelo {model} no está descargado: bajalo desde la caché de modelos"
        ) from e
    except ImportError as e:  # p. ej. tokenizadores que necesitan sentencepiece
        raise ValidationError(f"falta una dependencia para {model}: {e}".strip()[:300]) from e


def transformers_backend(
    model: str, modality: Modality, inputs: Sequence[Any], labels: list[str], multi: bool
) -> list[dict[str, float]]:
    template = TEMPLATES[modality]
    if modality is Modality.TEXT:
        clf = _pipeline("zero-shot-classification", model)
        out = clf(
            list(inputs), candidate_labels=labels, multi_label=multi, hypothesis_template=template
        )
        rows = out if isinstance(out, list) else [out]
        return [dict(zip(r["labels"], map(float, r["scores"]), strict=True)) for r in rows]
    if modality is Modality.IMAGE:
        from PIL import Image

        clf = _pipeline("zero-shot-image-classification", model)
        result = []
        for path in inputs:
            with Image.open(Path(path)) as im:
                preds = clf(
                    im.convert("RGB"), candidate_labels=labels, hypothesis_template=template
                )
            result.append({p["label"]: float(p["score"]) for p in preds})
        return result
    if modality is Modality.AUDIO:
        from perceptron.data.audio import load

        clf = _pipeline("zero-shot-audio-classification", model)
        sr = int(clf.feature_extractor.sampling_rate)
        result = []
        for path in inputs:
            wave, _ = load(Path(path), sample_rate=sr)
            preds = clf(wave[0], candidate_labels=labels, hypothesis_template=template)
            result.append({p["label"]: float(p["score"]) for p in preds})
        return result
    raise ValidationError(f"no hay zero-shot para la modalidad {modality.value}")


_BACKEND: dict[str, Backend] = {"active": transformers_backend}


def set_backend(backend: Backend | None) -> None:
    """Reemplaza el backend (None restaura transformers)."""
    _BACKEND["active"] = backend or transformers_backend


def resolve_model(modality: Modality, model: str | None) -> ZeroShotModelInfo:
    name = model or DEFAULT_ZERO_SHOT.get(modality)
    info = ZERO_SHOT_MODELS.get(name or "")
    if info is None:
        raise ValidationError(f"no hay un modelo zero-shot curado para {modality.value}")
    if info.modality is not modality:
        raise ValidationError(f"{info.model} es para {info.modality.value}, no {modality.value}")
    if not info.commercial_ok:
        raise ValidationError(f"la licencia de {info.model} no permite uso comercial")
    return info


def classify(
    modality: Modality,
    inputs: Sequence[Any],
    labels: list[str],
    *,
    model: str | None = None,
    multi_label: bool = False,
    batch: int = 16,
) -> list[dict[str, float]]:
    if len(labels) < 2 and not multi_label:
        raise ValidationError("el zero-shot necesita al menos dos clases")
    info = resolve_model(modality, model)
    out: list[dict[str, float]] = []
    for i in range(0, len(inputs), batch):
        out.extend(
            _BACKEND["active"](info.model, modality, inputs[i : i + batch], labels, multi_label)
        )
    return out
