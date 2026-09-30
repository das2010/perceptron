"""El informe del LLM debe traer las secciones como encabezados Markdown."""

from __future__ import annotations

from perceptron.llm.schemas import ModelCard, Report
from perceptron.services.llm_roles import REPORT_SECTIONS, report_validator


def _report(md: str) -> Report:
    card = ModelCard(intended_use="x", data="y", training="z")
    return Report(title="t", summary="s", markdown=md, model_card=card)


def test_plain_text_sections_are_rejected() -> None:
    check = report_validator("español")
    error = check(_report("Resumen\n\nTodo bien.\n\nDatos\n\n400 filas."))
    assert error and "'## Resumen'" in error and "'## Recomendaciones'" in error


def test_markdown_headings_pass_in_both_languages() -> None:
    for language, sections in REPORT_SECTIONS.items():
        md = "\n\n".join(f"## {name}\n\ntexto" for name in sections)
        assert report_validator(language)(_report(md)) is None


def test_model_card_accepts_per_class_metrics() -> None:
    """Con muchas clases el LLM detalla métricas por clase; no debe invalidar el informe."""
    from perceptron.llm.schemas import ModelCard

    card = ModelCard.model_validate(
        {
            "intended_use": "clasificar especies",
            "data": "fotos",
            "training": "EfficientNet-B0",
            "metrics": {
                "accuracy": 0.82,
                "per_class": [{"label": "Cat", "recall": 0.86}, {"label": "Dog", "recall": 0.68}],
                "recall_por_clase": {"Cat": 0.86},
            },
        }
    )
    assert card.metrics["per_class"][1] == {"label": "Dog", "recall": 0.68}
