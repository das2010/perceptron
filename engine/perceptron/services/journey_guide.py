"""Conocimiento del asistente guiado, determinístico y sin efectos (RF-WIZ-01/03).

El asistente conversa con el LLM, pero lo que define el recorrido es esto:

- **qué hace falta saber** de cada caso (`missing`), según el tipo de datos y de problema;
- **qué datos pedir y cómo prepararlos**, con un ejemplo armado con las palabras de la persona
  (`data_guide`) y una plantilla descargable;
- **cómo leer el resultado** según la tarea (`summarize`) y **cómo validarlo** con casos nuevos;
- **qué campos necesita el modelo** para una predicción (`input_form`).

Se apoya en la ficha del caso del wizard (`UseCaseBrief`): al confirmar, los hechos pasan a esa
ficha y el arquitecto, los requisitos de diseño y el plan los usan como en el modo avanzado.
"""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from perceptron.data.profiling.card import ProfileCard
    from perceptron.evaluation.evaluate import EvaluationReport

DataKind = Literal["table", "images", "text", "audio", "other"]
ProblemKind = Literal["category", "value", "rule", "forecast", "anomaly", "other"]
ErrorCosts = Literal["symmetric", "false_negative_worse", "false_positive_worse"]
FactField = Literal[
    "data_kind",
    "problem",
    "prediction",
    "labels_available",
    "classes",
    "error_costs",
    "positive_class",
    "extrapolate",
    "grouped",
    "success_target",
    "target_name",
    "input_names",
]
# Lo que el recorrido guiado sabe preparar, entrenar, evaluar y servir de punta a punta.
SUPPORTED: frozenset[tuple[str, str]] = frozenset(
    {
        ("table", "category"),
        ("table", "value"),
        ("table", "rule"),
        ("images", "category"),
        ("text", "category"),
        ("audio", "category"),
    }
)
MIN_ROWS = 20
MIN_PER_CLASS = 5
FEW_ROWS = 200
VALIDATION_MIN = 10


class Facts(BaseModel):
    """Lo que el asistente sabe del caso. Todo opcional: se completa conversando."""

    model_config = ConfigDict(extra="forbid")

    data_kind: DataKind | None = None
    problem: ProblemKind | None = None
    prediction: str | None = Field(default=None, max_length=300)
    labels_available: bool | None = None
    classes: list[str] | None = Field(default=None, min_length=2, max_length=100)
    error_costs: ErrorCosts | None = None
    positive_class: str | None = Field(default=None, max_length=100)
    extrapolate: bool | None = None
    grouped: bool | None = None
    success_target: float | None = Field(
        default=None,
        gt=0,
        le=100,
        description="Categoría: % mínimo de aciertos. Valor: error típico máximo aceptable (%)",
    )
    target_name: str | None = Field(default=None, max_length=100)
    input_names: list[str] | None = Field(default=None, max_length=40)


def unsupported_reason(f: Facts) -> str | None:
    """Por qué el recorrido guiado no puede resolver el caso (lo resuelve el modo avanzado)."""
    if f.data_kind == "other":
        return (
            "Por ahora el asistente trabaja con tablas, imágenes, textos o audios. Para otros "
            "datos usá el wizard de las opciones avanzadas."
        )
    if f.problem == "forecast":
        return (
            "Los pronósticos de series de tiempo todavía no tienen recorrido guiado: usá el "
            "wizard de las opciones avanzadas, que sí los entrena."
        )
    if f.problem == "anomaly":
        return (
            "La detección de anomalías todavía no tiene recorrido guiado: usá el wizard de las "
            "opciones avanzadas."
        )
    if f.problem == "other":
        return "No es un problema de predicción que el asistente sepa resolver todavía."
    if f.data_kind and f.problem and (f.data_kind, f.problem) not in SUPPORTED:
        return (
            "Esa combinación de datos y objetivo todavía no tiene recorrido guiado (por ahora: "
            "categorías con tablas, imágenes, textos o audios; valores y reglas con tablas). "
            "Usá el wizard de las opciones avanzadas."
        )
    if f.labels_available is False:
        return (
            "Para aprender hacen falta ejemplos con la respuesta correcta. Etiquetalos primero "
            "en «Etiquetado» (opciones avanzadas) y volvé a este recorrido."
        )
    return None


def _valid_classes(f: Facts) -> bool:
    c = f.classes or []
    return len(c) >= 2 and len(set(c)) == len(c) and all(x.strip() for x in c)


def missing(f: Facts) -> list[FactField]:
    """Lo esencial que falta saber, en el orden en que conviene preguntarlo."""
    out: list[FactField] = []
    if f.data_kind is None:
        out.append("data_kind")
    if f.problem is None:
        out.append("problem")
    if out or unsupported_reason(f):
        return out
    if not (f.prediction or "").strip():
        out.append("prediction")
    if f.labels_available is None:
        out.append("labels_available")
    if f.problem == "category":
        if f.data_kind in ("images", "audio") and not _valid_classes(f):
            out.append("classes")
        if f.error_costs is None:
            out.append("error_costs")
        elif (f.error_costs != "symmetric" and not (f.positive_class or "").strip()) or (
            f.positive_class and f.classes and f.positive_class not in f.classes
        ):
            out.append("positive_class")
    if f.problem in ("value", "rule") and f.extrapolate is None:
        out.append("extrapolate")
    if f.data_kind in ("images", "audio") and f.grouped is None:
        out.append("grouped")
    if f.success_target is None:
        out.append("success_target")
    return out


def to_brief(f: Facts) -> dict[str, Any]:
    """Los hechos como ficha del caso del wizard (`UseCaseBrief`)."""
    brief: dict[str, Any] = {
        "problem": f.problem,
        "prediction": f.prediction,
        "labels_available": f.labels_available,
        "error_costs": f.error_costs if f.problem == "category" else None,
        "extrapolate": f.extrapolate if f.problem in ("value", "rule") else None,
        "has_entities": f.grouped,
    }
    if f.success_target is not None:
        unit = "% de aciertos" if f.problem == "category" else "% de error típico como máximo"
        brief["business_metric"] = f"{f.success_target:g} {unit}"
    return {k: v for k, v in brief.items() if v is not None}


def task_of(f: Facts) -> str:
    return "classification" if f.problem == "category" else "regression"


# ---------------------------------------------------------------------- guía de datos


class DataGuide(BaseModel):
    """Qué datos hacen falta y cómo prepararlos, con un ejemplo concreto."""

    layout: Literal["table", "folders"]
    accept: list[str]
    steps: list[str]
    example_columns: list[str] = Field(default_factory=list)
    example_rows: list[list[str]] = Field(default_factory=list)
    example_tree: list[str] = Field(default_factory=list)
    template_csv: str | None = None
    tips: list[str] = Field(default_factory=list)


def _csv(columns: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    w.writerows(rows)
    return buf.getvalue()


def data_guide(f: Facts) -> DataGuide:
    target = (f.target_name or "").strip() or (
        "categoria" if f.problem == "category" else "resultado"
    )
    classes = f.classes or []
    if f.data_kind in ("images", "audio"):
        ext = "jpg" if f.data_kind == "images" else "wav"
        noun = "imágenes" if f.data_kind == "images" else "grabaciones"
        names = classes or ["categoria_a", "categoria_b"]
        group = "objeto_01/" if f.grouped else ""
        stem = "muestra" if ext == "wav" else "foto"
        tree = [f"{c}/{group}{stem}_0{i}.{ext}" for c in names for i in (1, 2)]
        tips = [
            f"Al menos 20 {noun} por categoría; con 100 o más el resultado es mucho más confiable.",
            "Que se parezcan a las que vas a usar después: mismo equipo, luz o ruido, encuadre.",
            "Cada archivo en la carpeta de su categoría: el nombre de la carpeta es la respuesta.",
        ]
        if f.grouped:
            tips.append(
                f"Varias {noun} del mismo objeto o sesión van en la misma subcarpeta (objeto_01, "
                "…): así nunca se usan a la vez para aprender y para evaluar, y el resultado no "
                "se infla."
            )
        return DataGuide(
            layout="folders",
            accept=[".zip"],
            steps=[
                f"Creá una carpeta por categoría ({', '.join(names)}).",
                f"Poné cada {'imagen' if ext == 'jpg' else 'grabación'} en la carpeta que le "
                "corresponde.",
                "Comprimí todo en un único archivo ZIP y subilo acá.",
            ],
            example_tree=tree,
            tips=tips,
        )
    if f.data_kind == "text":
        cols = ["texto", target]
        rows = [
            ["El envío llegó tarde y la caja estaba rota", classes[0] if classes else "reclamo"],
            [
                "¿Puedo cambiar el talle de la remera?",
                classes[1] if len(classes) > 1 else "consulta",
            ],
        ]
        return DataGuide(
            layout="table",
            accept=[".csv", ".xlsx", ".parquet", ".jsonl"],
            steps=[
                "Armá una tabla con una fila por texto.",
                f"Una columna con el texto y otra («{target}») con la categoría correcta.",
                "Guardala como CSV o Excel con los nombres de columna en la primera fila.",
            ],
            example_columns=cols,
            example_rows=rows,
            template_csv=_csv(cols, rows),
            tips=[
                "Al menos 50 textos por categoría; con varios cientos es mucho mejor.",
                "Textos reales, como los que van a llegar después (no ejemplos inventados).",
            ],
        )
    # Tabla: una fila por caso, columnas de entrada y la respuesta.
    inputs = [c for c in (f.input_names or []) if c.strip() and c != target] or [
        "dato_1",
        "dato_2",
        "dato_3",
    ]
    cols = [*inputs, target]
    answers = (classes or ["sí", "no"])[:2] if f.problem == "category" else ["123.4", "98.7"]
    rows = [["…" for _ in inputs] + [a] for a in answers]
    tips = [
        "Una fila por caso: cada fila es un ejemplo con su resultado real.",
        f"Las columnas de entrada tienen que conocerse en el momento de predecir. No incluyas "
        f"datos que se conocen recién después de «{target}»: el modelo haría trampa y "
        "después fallaría.",
        f"Al menos {FEW_ROWS} filas; con miles el resultado es más confiable.",
        "Números con punto o coma decimal, sin símbolos de moneda ni unidades en la celda.",
    ]
    if f.extrapolate:
        tips.append(
            "Vas a predecir fuera de lo visto: cubrí el rango más amplio posible de cada entrada. "
            "Si el resultado sale de una regla, buscamos también la fórmula."
        )
    if f.problem == "category" and f.error_costs and f.error_costs != "symmetric":
        tips.append(
            f"Incluí suficientes casos de «{f.positive_class or 'la categoría importante'}»: "
            "si son pocos, el modelo los confunde."
        )
    return DataGuide(
        layout="table",
        accept=[".csv", ".xlsx", ".xls", ".parquet", ".jsonl"],
        steps=[
            "Armá una tabla con una fila por caso y una columna por dato.",
            f"Agregá la columna «{target}» con el resultado real de cada caso.",
            "Guardala como CSV o Excel con los nombres de columna en la primera fila y subila.",
        ],
        example_columns=cols,
        example_rows=rows,
        template_csv=_csv(cols, rows),
        tips=tips,
    )


# ---------------------------------------------------------------------- resultados


def summarize(
    report: EvaluationReport, f: Facts, typical_value: float | None = None
) -> dict[str, Any]:
    """Resultado en test explicado para alguien sin conocimientos de ML. `typical_value`: el
    valor típico del objetivo (mediana de train), para expresar el error en porcentaje."""
    target = f.success_target or 0.0
    if report.classification is not None:
        c = report.classification
        total = sum(sum(row) for row in c.confusion_matrix)
        correct = sum(c.confusion_matrix[i][i] for i in range(len(c.confusion_matrix)))
        pct = 100 * correct / total if total else 0.0
        out: dict[str, Any] = {
            "kind": "classification",
            "total": total,
            "correct": correct,
            "percent": round(pct, 1),
            "per_class": [
                {"label": p.label, "found": round(100 * p.recall, 1), "support": p.support}
                for p in c.per_class
            ],
        }
        explanation = [
            f"Sobre {total} casos que el modelo nunca vio, acertó {correct} ({pct:.1f} %)."
        ]
        positive = f.positive_class if f.positive_class in c.labels else None
        if positive is not None:
            k = c.labels.index(positive)
            misses = sum(c.confusion_matrix[k]) - c.confusion_matrix[k][k]
            alarms = sum(row[k] for row in c.confusion_matrix) - c.confusion_matrix[k][k]
            out |= {"positive": positive, "misses": misses, "false_alarms": alarms}
            explanation.append(
                f"De los casos de «{positive}», no detectó {misses}; y marcó {alarms} como "
                f"«{positive}» sin serlo (falsas alarmas)."
            )
        worst = min(c.per_class, key=lambda p: p.recall, default=None)
        if worst is not None and worst.recall < 0.8:
            explanation.append(
                f"La categoría que más le cuesta es «{worst.label}»: reconoce el "
                f"{100 * worst.recall:.0f} % de sus casos."
            )
        meets = pct >= target
    elif report.regression is not None:
        r = report.regression
        scale = abs(typical_value or 0.0)
        typical = 100 * r.mae / scale if scale else None
        out = {
            "kind": "regression",
            "total": report.num_samples,
            "typical_error": round(r.mae, 4),
            "typical_percent": round(typical, 2) if typical is not None else None,
            "r2": round(r.r2, 4),
        }
        explanation = [
            f"Sobre {report.num_samples} casos que el modelo nunca vio, se equivoca en promedio "
            f"por {r.mae:,.4g}".replace(",", ".")
            + (f" (un {typical:.1f} % del valor típico)." if typical is not None else "."),
            f"Explica el {100 * max(r.r2, 0):.1f} % de la variación del resultado.",
        ]
        meets = typical is not None and typical <= target
    else:
        out = {"kind": report.task.value, "total": report.num_samples}
        explanation = ["El modelo se evaluó con casos que nunca vio."]
        meets = False
    out["meets_target"] = bool(meets)
    out["explanation"] = explanation
    out["next_steps"] = next_steps(out, f)
    return out


def next_steps(result: dict[str, Any], f: Facts) -> list[str]:
    if result.get("meets_target"):
        return [
            "Validalo con casos nuevos que no estén en los datos que subiste (otro día, otros "
            "objetos u otras personas), cargando su resultado real para comparar.",
            f"Con al menos {VALIDATION_MIN} casos nuevos"
            + (" de todas las categorías" if f.problem == "category" else "")
            + " que cumplan el objetivo, queda listo para usar.",
        ]
    steps = ["Todavía no llega al objetivo que definiste."]
    if result.get("kind") == "classification":
        steps.append(
            "Lo que más ayuda: más ejemplos de las categorías que más le cuestan, y que se "
            "parezcan a los casos reales."
        )
    else:
        steps.append(
            "Lo que más ayuda: más filas, cubrir mejor el rango de cada dato y sumar columnas "
            "que influyan en el resultado."
        )
    steps.append(
        "Podés subir una nueva versión de los datos y volver a entrenar, o revisar el detalle "
        "en «Experimentos» (opciones avanzadas)."
    )
    return steps


# ---------------------------------------------------------------------- predicción


class InputField(BaseModel):
    name: str
    kind: Literal["number", "category", "boolean", "date", "text"]
    options: list[str] = Field(default_factory=list)
    example: str | None = None


class InputForm(BaseModel):
    """Lo que el modelo necesita para una predicción, y nada más."""

    kind: Literal["fields", "image", "audio", "text"]
    fields: list[InputField] = Field(default_factory=list)


def input_form(signature: dict[str, Any], card: ProfileCard, example: dict[str, Any]) -> InputForm:
    from perceptron.data.schema import SemanticType

    kind = str(signature.get("inputs", {}).get("kind"))
    if kind == "image":
        return InputForm(kind="image")
    if kind == "spectrogram":
        return InputForm(kind="audio")
    if kind == "tokens":
        return InputForm(kind="text")
    target = card.target.name if card.target else None
    by_name = {c.name: c for c in card.columns}
    inputs = signature.get("inputs", {})
    # Lo que el modelo usa de verdad (los identificadores que la preparación descarta, no).
    used = {*inputs.get("numeric_features", []), *inputs.get("categorical_features", [])}

    def needed(name: str) -> bool:
        return not used or name in used or any(u.startswith(f"{name}.") for u in used)

    fields: list[InputField] = []
    for name in inputs.get("columns", []):
        if name == target or name not in by_name or not needed(name):
            continue
        if by_name[name].semantic in (SemanticType.ID, SemanticType.FILEPATH):
            continue
        col = by_name[name]
        sem = col.semantic
        value = example.get(name)
        ex = None if value is None else str(value)
        if sem is SemanticType.NUMERIC:
            fields.append(InputField(name=name, kind="number", example=ex))
        elif sem is SemanticType.BOOLEAN:
            fields.append(InputField(name=name, kind="boolean", example=ex))
        elif sem is SemanticType.DATETIME:
            fields.append(InputField(name=name, kind="date", example=ex))
        elif sem is SemanticType.CATEGORICAL:
            options = [c.value for c in (col.categorical.top if col.categorical else [])]
            fields.append(InputField(name=name, kind="category", options=options, example=ex))
        else:
            fields.append(InputField(name=name, kind="text", example=ex))
    return InputForm(kind="fields", fields=fields)
