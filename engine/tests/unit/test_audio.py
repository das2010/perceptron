"""Audio (Capa 1b, sub-hito 3): IO, DSP, ingesta, profiling, pipeline, dataset y reglas."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from perceptron.archspec.builder import build_model
from perceptron.archspec.validate import validate_archspec
from perceptron.catalog.rules import recommend
from perceptron.core.paths import ProjectPaths
from perceptron.data.audio import (
    fix_length,
    frames_for,
    load,
    log_mel,
    mel_filterbank,
    mfcc,
    quality,
    resample,
    spec_augment,
)
from perceptron.data.pipeline.pipeline import fit_pipeline
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView
from perceptron.domain.enums import Modality
from perceptron.training.data import make_dataset


def _tone(freq: float, sr: int = 8000, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def test_load_resample_and_mono(tmp_path: Path) -> None:
    stereo = np.stack([_tone(440), _tone(440)], axis=1)
    path = tmp_path / "estéreo ñ.wav"
    sf.write(path, stereo, 8000)
    x, sr = load(path)
    assert x.shape == (1, 8000) and sr == 8000
    y, sr2 = load(path, sample_rate=16000)
    assert sr2 == 16000 and y.shape[1] == pytest.approx(16000, abs=2)
    assert resample(x, 8000, 4000).shape[1] == 4000


def test_features_shapes_and_peak() -> None:
    x = _tone(1000)[None, :]
    lm = log_mel(x, 8000, n_mels=64)
    assert lm.shape == (1, 64, frames_for(1.0, 8000))
    # La energía se concentra en la banda del tono
    fb = mel_filterbank(8000, 256, 64)
    assert fb.shape[0] == 64 and (fb >= 0).all()
    peak_band = int(lm[0].mean(dim=1).argmax())
    assert 20 < peak_band < 50  # 1 kHz a 8 kHz de sample rate
    assert mfcc(x, 8000, n_mfcc=13).shape[1] == 13
    assert fix_length(x, 100).shape == (1, 100)
    assert np.abs(fix_length(x[:, :10], 20)[0, 10:]).sum() == 0


def test_quality_metrics() -> None:
    silent = np.zeros((1, 8000), dtype=np.float32)
    assert quality(silent)["silence_fraction"] == 1.0
    clipped = np.clip(_tone(200) * 4, -1, 1)[None, :]
    assert quality(clipped)["clipping_fraction"] > 0.1


def test_spec_augment_masks() -> None:
    s = torch.randn(1, 32, 50)
    g = torch.Generator().manual_seed(0)
    out = spec_augment(s, freq_mask=8, time_mask=10, rng=g)
    assert out.shape == s.shape


@pytest.fixture
def paths(workspace_dir: Path) -> ProjectPaths:
    return ProjectPaths(workspace_dir / "projects" / "prj_au").ensure()


def test_uc09_end_to_end_prep(
    paths: ProjectPaths, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    v = ingest(paths, IngestRequest(project_id="prj_au", source=fixtures_dir / "uc09_motor_audio"))
    assert v.modality is Modality.AUDIO
    assert v.target == "label" and v.num_samples == 120
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    assert card.audio is not None
    assert card.audio.corrupt == 0
    assert [c.value for c in card.audio.sample_rates] == ["8000"]
    assert card.audio.duration_quantiles["p50"] == pytest.approx(1.0)
    assert card.audio.snr_db is not None

    spec = propose_pipeline(card)
    assert spec.audio is not None and spec.audio.sample_rate == 8000
    fitted = fit_pipeline(spec, view.read("train"), view.files_dir)
    assert fitted.audio_mean is not None and fitted.audio_std and fitted.audio_std > 0
    assert fitted.classes == ["cavitacion", "desbalance", "normal", "rodamiento"]

    ds = make_dataset(view, fitted, "train", train=True)
    x, _ = ds[0]
    assert x.shape == (1, 64, frames_for(spec.audio.duration_s, 8000))
    assert abs(float(x.mean())) < 3

    rec = recommend(card, fitted)
    assert rec.template == "crnn"
    assert rec.spec.modality is Modality.AUDIO
    assert rec.spec.input.kind == "spectrogram"
    assert validate_archspec(rec.spec).valid
    out = build_model(rec.spec).model(x.unsqueeze(0).repeat(2, 1, 1, 1))
    assert out.shape == (2, 4)


def test_image_dataset_normalization_stats_are_fitted(
    paths: ProjectPaths, fixtures_dir: Path
) -> None:
    """Regresión: con normalize='dataset' antes se usaban en silencio las de ImageNet."""
    v = ingest(paths, IngestRequest(project_id="prj_au", source=fixtures_dir / "uc04_defects"))
    view = DatasetView(paths.dataset(v.content_hash))
    spec = propose_pipeline(profile_dataset(view))
    assert spec.image is not None and spec.image.normalize == "dataset"
    fitted = fit_pipeline(spec, view.read("train"), view.files_dir)
    assert fitted.image_mean is not None and len(fitted.image_mean) == 3
    assert 0.4 < fitted.image_mean[0] < 0.7  # piezas grises


def test_audio_crnn_forward_and_code() -> None:
    from typing import Any

    from perceptron.archspec.to_code import archspec_to_code
    from perceptron.catalog.templates import audio_template
    from perceptron.domain.enums import TaskType

    spec = audio_template("crnn", task=TaskType.CLASSIFICATION, num_classes=4, bins=64, frames=101)
    assert validate_archspec(spec).valid, validate_archspec(spec).feedback()
    built = build_model(spec)
    x = torch.randn(2, 1, 64, 101)
    assert built.model(x).shape == (2, 4)
    ns: dict[str, Any] = {"__name__": "gen"}
    exec(compile(archspec_to_code(spec), "<gen>", "exec"), ns)  # noqa: S102 - código propio
    gen = ns["Model"]()
    gen.load_state_dict({k.removeprefix("blocks."): v for k, v in built.model.state_dict().items()})
    built.model.eval()
    gen.eval()
    with torch.no_grad():
        assert torch.allclose(built.model(x), gen(x), atol=1e-5)
