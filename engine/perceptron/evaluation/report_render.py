"""Informe final exportable con marca Preteco (RF-EVL-06; ADR-0028).

Formatos: HTML autocontenido (Titillium Web embebida, gráficos en SVG), PDF (reportlab, sin
dependencias del sistema) y Markdown. Toma el informe (LLM o plantilla), la evaluación del
test sellado y, si ya se calcularon, la explicación global, la equidad y la robustez.

El Markdown del informe se convierte sin HTML crudo: si el LLM devolviera etiquetas, se
muestran como texto (defensa ante prompt injection, §13.2).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt

import perceptron
from perceptron.domain.models import utcnow
from perceptron.evaluation.evaluate import EvaluationReport
from perceptron.llm.schemas import Report

HERE = Path(__file__).parent
DARK = "#334000"
LIME = "#cbff00"
MUTED = "#5a6330"


@dataclass
class ReportInputs:
    report: Report
    evaluation: EvaluationReport
    run_id: str
    project: str
    explanation: dict[str, Any] | None = None
    fairness: list[dict[str, Any]] = field(default_factory=list)
    robustness: dict[str, Any] | None = None


def _markdown() -> MarkdownIt:
    return MarkdownIt("commonmark", {"html": False}).enable("table")


def _confusion(ev: EvaluationReport) -> dict[str, Any] | None:
    cls = ev.classification
    if cls is None or not cls.confusion_matrix:
        return None
    top = max((v for row in cls.confusion_matrix for v in row), default=1) or 1
    return {
        "labels": cls.labels,
        "rows": [
            {
                "label": label,
                "cells": [{"value": v, "alpha": round(0.15 + 0.7 * v / top, 3)} for v in row],
            }
            for label, row in zip(cls.labels, cls.confusion_matrix, strict=False)
        ],
    }


def _explanation_view(exp: dict[str, Any] | None) -> dict[str, Any] | None:
    if not exp or not exp.get("features"):
        return None
    top = exp["features"][:10]
    return {**exp, "top": top, "max": max(f["importance"] for f in top) or 1.0}


def render_markdown(inp: ReportInputs) -> str:
    return inp.report.markdown


def render_html(inp: ReportInputs) -> str:
    env = Environment(
        loader=FileSystemLoader(HERE / "templates"),
        autoescape=select_autoescape(["html", "j2"]),
    )
    fonts = {
        w: base64.b64encode(
            (HERE / "assets" / f"titillium-web-latin-{w}-normal.woff2").read_bytes()
        ).decode("ascii")
        for w in ("400", "600")
    }
    return env.get_template("report.html.j2").render(
        lang="es",
        report=inp.report,
        project=inp.project,
        run_id=inp.run_id,
        created=utcnow().date().isoformat(),
        version=perceptron.__version__,
        metrics=sorted(inp.evaluation.metrics.items()),
        confusion=_confusion(inp.evaluation),
        body=_markdown().render(inp.report.markdown),
        explanation=_explanation_view(inp.explanation),
        fairness=inp.fairness,
        robustness=inp.robustness,
        font400=fonts["400"],
        font600=fonts["600"],
    )


# ------------------------------------------------------------------ PDF


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _markdown_flowables(md: str, styles: Any) -> list[Any]:
    from reportlab.platypus import ListFlowable, ListItem, Paragraph

    out: list[Any] = []
    tokens = _markdown().parse(md)
    heading: str | None = None
    items: list[Any] = []
    in_list = 0
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            heading = {"h1": "Heading2", "h2": "Heading2"}.get(tok.tag, "Heading3")
        elif tok.type in ("bullet_list_open", "ordered_list_open"):
            in_list += 1
        elif tok.type in ("bullet_list_close", "ordered_list_close"):
            in_list -= 1
            if not in_list and items:
                out.append(ListFlowable(items, bulletType="bullet", leftIndent=12))
                items = []
        elif tok.type == "inline":
            text = _escape(tok.content)
            if heading:
                out.append(Paragraph(text, styles[heading]))
                heading = None
            elif in_list:
                items.append(ListItem(Paragraph(text, styles["BodyText"])))
            elif i and tokens[i - 1].type == "paragraph_open":
                out.append(Paragraph(text, styles["BodyText"]))
    return out


def _table(rows: list[list[Any]], widths: list[float] | None = None) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    t = Table([[str(c) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f7e8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9dfc2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def _bar_chart(features: list[dict[str, Any]]) -> Any:
    from reportlab.graphics.charts.barcharts import HorizontalBarChart
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors

    top = list(reversed(features[:10]))
    d = Drawing(460, 18 * len(top) + 20)
    chart = HorizontalBarChart()
    chart.x, chart.y, chart.width, chart.height = 150, 10, 290, 18 * len(top)
    chart.data = [[f["importance"] for f in top]]
    chart.categoryAxis.categoryNames = [f["feature"][:26] for f in top]
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = colors.HexColor(DARK)
    d.add(chart)
    return d


def render_pdf(inp: ReportInputs) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "Brand", parent=styles["Normal"], textColor=colors.HexColor(MUTED), fontSize=9
        )
    )
    styles["Title"].textColor = colors.HexColor(DARK)
    for name in ("Heading2", "Heading3"):
        styles[name].textColor = colors.HexColor(DARK)
    r = inp.report
    story: list[Any] = [
        Paragraph("PRETECO · PERCEPTRON", styles["Brand"]),
        Paragraph(_escape(r.title), styles["Title"]),
        Paragraph(
            _escape(f"{inp.project} · run {inp.run_id} · {utcnow().date().isoformat()}"),
            styles["Brand"],
        ),
        HRFlowable(width="100%", thickness=3, color=colors.HexColor(LIME)),
        Spacer(1, 8),
        Paragraph(f"<b>Resumen.</b> {_escape(r.summary)}", styles["BodyText"]),
        Paragraph("Métricas en el test sellado", styles["Heading2"]),
        _table(
            [
                ["Métrica", "Valor"],
                *[[k, f"{v:.4g}"] for k, v in sorted(inp.evaluation.metrics.items())],
            ]
        ),
    ]
    conf = _confusion(inp.evaluation)
    if conf:
        story += [
            Paragraph("Matriz de confusión", styles["Heading3"]),
            _table(
                [
                    ["Real \\ Predicho", *conf["labels"]],
                    *[[row["label"], *[c["value"] for c in row["cells"]]] for row in conf["rows"]],
                ]
            ),
        ]
    story += _markdown_flowables(r.markdown, styles)
    exp = _explanation_view(inp.explanation)
    if exp:
        story += [
            Paragraph("Qué variables pesan más", styles["Heading2"]),
            Paragraph(
                f"Importancia global (Shapley por muestreo, {exp['samples']} filas de validación).",
                styles["Brand"],
            ),
            _bar_chart(exp["features"]),
        ]
    for f in inp.fairness:
        story.append(Paragraph(f"Equidad: {_escape(f['attribute'])}", styles["Heading2"]))
        if f["task"] == "regression":
            rows = [["Grupo", "Casos", "MAE"]] + [
                [g["group"], g["support"], f"{g['mae']:.4g}"] for g in f["groups"]
            ]
        else:
            rows = [
                ["Grupo", "Casos", f"Tasa «{f['positive_class']}»", "Accuracy", "TPR", "FPR"]
            ] + [
                [
                    g["group"],
                    g["support"],
                    *(f"{(g[k] or 0):.3f}" for k in ("selection_rate", "accuracy", "tpr", "fpr")),
                ]
                for g in f["groups"]
            ]
        story.append(_table(rows))
        story += [Paragraph(f"⚠ {_escape(a)}", styles["BodyText"]) for a in f["alerts"]]
    rob = inp.robustness
    if rob:
        story += [
            Paragraph("Robustez ante perturbaciones", styles["Heading2"]),
            _table(
                [["Perturbación", "Severidad", rob["metric"], "Degradación"]]
                + [
                    [x["kind"], x["severity"], f"{x['metric']:.4g}", f"{x['degradation']:.4g}"]
                    for x in rob["results"]
                ]
            ),
        ]
    mc = r.model_card
    story += [
        Paragraph("Model card", styles["Heading2"]),
        _table(
            [
                ["Campo", "Detalle"],
                ["Uso previsto", Paragraph(_escape(mc.intended_use), styles["BodyText"])],
                ["Datos", Paragraph(_escape(mc.data), styles["BodyText"])],
                ["Entrenamiento", Paragraph(_escape(mc.training), styles["BodyText"])],
                [
                    "Limitaciones",
                    Paragraph(_escape("; ".join(mc.limitations) or "—"), styles["BodyText"]),
                ],
            ],
            widths=[3.5 * cm, 13 * cm],
        ),
    ]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, title=r.title, author="Perceptron", leftMargin=2 * cm, rightMargin=2 * cm
    )
    doc.build(story)
    return buf.getvalue()
