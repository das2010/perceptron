"""Modalidad texto (Capa 1b, sub-hito 1): utilidades, ingesta, profiling, pipeline y catálogo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch

from perceptron.archspec.builder import build_model
from perceptron.archspec.to_code import archspec_to_code
from perceptron.archspec.validate import Stage, validate_archspec
from perceptron.catalog.rules import recommend
from perceptron.catalog.templates import text_template
from perceptron.core.paths import ProjectPaths
from perceptron.data.pipeline.pipeline import fit_pipeline, transform_text
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.profiling.card import AlertCode
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.text import (
    PAD,
    UNK,
    build_vocab,
    encode,
    guess_language,
    normalize,
    strip_accents,
    tokenize,
)
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView
from perceptron.domain.enums import Modality, TaskType


def test_normalize_and_tokenize() -> None:
    assert normalize("  Hola   MUNDO https://x.com/a  ") == "hola mundo <url>"
    assert normalize("Pagué 1.500 pesos", numbers=True) == "pagué <num> pesos"
    assert strip_accents("Ñandú está acá") == "Ñandu esta aca"
    assert tokenize("hola, ¿qué tal? <url>") == ["hola", ",", "¿", "qué", "tal", "?", "<url>"]


def test_vocab_is_deterministic_and_encodes_unknowns() -> None:
    texts = ["b a", "a c", "a b"]
    vocab = build_vocab(texts, min_freq=1)
    assert vocab[:2] == ["<pad>", "<unk>"]
    assert vocab[2] == "a"  # la más frecuente
    assert build_vocab(texts, min_freq=2) == ["<pad>", "<unk>", "a", "b"]
    index = {w: i for i, w in enumerate(vocab)}
    assert encode("a zzz", index, 4) == [index["a"], UNK, PAD, PAD]
    assert len(encode("a " * 10, index, 3)) == 3


def test_guess_language() -> None:
    assert guess_language(["No puedo descargar la factura de marzo"])[0] == "es"
    assert guess_language(["I can not download the invoice for this month"])[0] == "en"
    assert guess_language(["12345"]) == (None, 0.0)


@pytest.fixture
def paths(workspace_dir: Path) -> ProjectPaths:
    return ProjectPaths(workspace_dir / "projects" / "prj_txt").ensure()


def _uc03(paths: ProjectPaths, fixtures_dir: Path) -> Any:
    v = ingest(
        paths,
        IngestRequest(
            project_id="prj_txt", source=fixtures_dir / "uc03_tickets_es" / "tickets.jsonl"
        ),
    )
    return v, DatasetView(paths.dataset(v.content_hash))


def test_ingest_detects_text_modality(paths: ProjectPaths, fixtures_dir: Path) -> None:
    v, view = _uc03(paths, fixtures_dir)
    assert v.modality is Modality.TEXT
    assert v.target == "categoria"
    assert view.text_column == "texto"

    # UC-01 sigue siendo tabular; se puede forzar la modalidad
    churn = ingest(
        paths, IngestRequest(project_id="prj_txt", source=fixtures_dir / "uc01_churn" / "churn.csv")
    )
    assert churn.modality is Modality.TABULAR


def test_text_folder_source(paths: ProjectPaths, tmp_path: Path) -> None:
    root = tmp_path / "reseñas"
    for label, texts in {
        "positivo": ["me encantó el producto", "excelente atención"] * 5,
        "negativo": ["llegó roto y tarde", "pésimo servicio"] * 5,
    }.items():
        (root / label).mkdir(parents=True)
        for i, t in enumerate(texts):
            (root / label / f"r{i}.txt").write_text(t, encoding="utf-8")
    v = ingest(paths, IngestRequest(project_id="prj_txt", source=root))
    assert v.modality is Modality.TEXT
    assert v.target == "label"
    assert DatasetView(paths.dataset(v.content_hash)).text_column == "text"


def test_profile_and_pipeline(paths: ProjectPaths, fixtures_dir: Path) -> None:
    _, view = _uc03(paths, fixtures_dir)
    card = profile_dataset(view)
    assert card.text is not None
    assert card.text.language == "es"
    assert card.text.tokens_p95 and card.text.tokens_p95 > 3
    assert card.text.vocabulary_size > 20
    assert not card.alerts_by(AlertCode.DUPLICATE_TEXTS) or card.text.duplicate_fraction > 0.05
    # Privacidad L1: ningún texto crudo en el card
    raw = view.read()["texto"].to_list()
    card_json = card.model_dump_json()
    assert not any(t in card_json for t in raw)

    spec = propose_pipeline(card)
    assert spec.text is not None and spec.text.tokenizer == "word"
    fitted = fit_pipeline(spec, view.read("train"))
    assert fitted.vocab and fitted.vocab[:2] == ["<pad>", "<unk>"]
    assert fitted.classes == sorted(view.read("train")["categoria"].unique().to_list())
    ids = transform_text(fitted, view.read("val"))
    assert ids.dtype == np.int64 and ids.shape[1] == spec.text.max_length
    assert (ids[:, 0] > 1).mean() > 0.8  # casi todos los textos empiezan con palabras conocidas

    rec = recommend(card, fitted)
    assert rec.template == "textcnn"
    assert rec.spec.input.vocab_size == len(fitted.vocab)
    assert validate_archspec(rec.spec).valid


@pytest.mark.parametrize("backbone", ["textcnn", "bilstm"])
def test_text_templates_forward_and_code(backbone: str) -> None:
    spec = text_template(
        backbone, task=TaskType.CLASSIFICATION, num_classes=4, max_length=12, vocab_size=50
    )
    assert validate_archspec(spec).valid
    built = build_model(spec)
    ids = torch.randint(0, 50, (3, 12))
    assert built.model(ids).shape == (3, 4)

    ns: dict[str, Any] = {"__name__": "gen"}
    exec(compile(archspec_to_code(spec), "<gen>", "exec"), ns)  # noqa: S102 - código propio
    gen = ns["Model"]()
    gen.load_state_dict({k.removeprefix("blocks."): v for k, v in built.model.state_dict().items()})
    built.model.eval()
    gen.eval()
    with torch.no_grad():
        assert torch.allclose(built.model(ids), gen(ids), atol=1e-5)


def test_hf_encoder_validates_without_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    spec = text_template(
        "hf",
        task=TaskType.CLASSIFICATION,
        num_classes=3,
        max_length=32,
        hf_model="distilbert-base-multilingual-cased",
    )
    report = validate_archspec(spec, cache_dir=tmp_path)
    assert report.valid, report.feedback()  # la shape sale de la tabla curada (sin descargar)
    assert report.issues[0].stage is Stage.WEIGHTS  # advertencia: no está en caché


@pytest.mark.network
def test_hf_tiny_encoder_trains_a_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Descarga un modelo de prueba minúsculo (~1 MB) y verifica tokenizador + encoder."""
    from perceptron.data.pipeline.pipeline import PipelineSpec, TargetSpec, TextSpec

    name = "hf-internal-testing/tiny-random-bert"
    df = pl.DataFrame(
        {"t": ["hola mundo", "chau mundo", "hola otra vez", "nada"], "y": ["a", "b", "a", "b"]}
    )
    spec = PipelineSpec(
        modality=Modality.TEXT,
        target=TargetSpec(name="y", task=TaskType.CLASSIFICATION),
        text=TextSpec(column="t", tokenizer="hf", hf_model=name, max_length=8),
    )
    fitted = fit_pipeline(spec, df)
    ids = torch.tensor(transform_text(fitted, df))
    arch = text_template(
        "hf",
        task=TaskType.CLASSIFICATION,
        num_classes=2,
        max_length=8,
        pad_id=fitted.pad_id,
        hf_model=name,
    )
    model = build_model(arch).model
    loss = torch.nn.functional.cross_entropy(model(ids), torch.tensor([0, 1, 0, 1]))
    loss.backward()
    assert loss.item() > 0
