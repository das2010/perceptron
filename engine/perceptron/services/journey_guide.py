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
    from perceptron.data.series import SeriesProfile
    from perceptron.evaluation.evaluate import EvaluationReport

DataKind = Literal["table", "images", "text", "audio", "other"]
ProblemKind = Literal[
    "category", "value", "rule", "forecast", "anomaly", "locate", "segment", "read_text", "other"
]
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
    "horizon",
]
# Lo que el recorrido guiado sabe preparar, entrenar, evaluar y servir de punta a punta.
SUPPORTED: frozenset[tuple[str, str]] = frozenset(
    {
        ("table", "category"),
        ("table", "value"),
        ("table", "rule"),
        ("table", "forecast"),
        ("table", "anomaly"),
        ("images", "category"),
        ("images", "locate"),
        ("images", "segment"),
        ("images", "read_text"),
        ("text", "category"),
        ("audio", "category"),
    }
)
# La tarea del motor que resuelve cada problema (TaskType).
TASKS: dict[str, str] = {
    "category": "classification",
    "value": "regression",
    "rule": "regression",
    "forecast": "forecasting",
    "anomaly": "anomaly_detection",
    "locate": "object_detection",
    "segment": "segmentation",
    "read_text": "ocr",
}
SERIES = ("forecast", "anomaly")
VISION = ("locate", "segment", "read_text")
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
    classes: list[str] | None = Field(default=None, min_length=1, max_length=100)
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
    horizon: int | None = Field(
        default=None, ge=1, le=500, description="Pronóstico: cuántos valores hacia adelante"
    )


def unsupported_reason(f: Facts) -> str | None:
    """Por qué el recorrido guiado no puede resolver el caso, con un replanteo que sí puede (o
    el camino del modo avanzado). None si el asistente lo resuelve de punta a punta."""
    if f.data_kind == "other":
        return (
            "Por ahora el asistente trabaja con tablas (planillas, mediciones en el tiempo), "
            "fotos, textos o grabaciones. Si podés exportar tus datos a una planilla, casi "
            "siempre se puede plantear el caso; si no, el wizard de las opciones avanzadas "
            "tiene más formatos."
        )
    if f.problem == "other":
        return (
            "Eso no es algo que el asistente sepa resolver todavía. Casi todos los casos se "
            "pueden plantear como elegir entre respuestas posibles, estimar un número, "
            "pronosticar valores en el tiempo o detectar casos raros: contame qué decisión "
            "querés tomar con el resultado y lo vemos juntos."
        )
    if f.data_kind and f.problem and (f.data_kind, f.problem) not in SUPPORTED:
        return _reframe(f.data_kind, f.problem)
    if f.labels_available is False and f.problem not in ("anomaly", "forecast"):
        return (
            "Para aprender hacen falta ejemplos con la respuesta correcta. Podés marcarlos en "
            "«Etiquetado» (opciones avanzadas): con unos cientos alcanza para empezar, y "
            "después volvés a este recorrido."
        )
    return None


def _reframe(kind: str, problem: str) -> str:
    """Un planteo equivalente que el asistente sí resuelve."""
    if problem in ("value", "rule"):
        return (
            "Con "
            + {"images": "fotos", "text": "textos", "audio": "grabaciones"}.get(kind, "esos datos")
            + " por ahora puedo elegir entre respuestas posibles, no estimar un número exacto. "
            "Si te sirve, agrupá el resultado en rangos (por ejemplo «bajo», «medio», «alto») "
            "y lo planteamos así; o, si tenés esos datos en una planilla, estimo el número."
        )
    if problem in SERIES:
        return (
            "Los pronósticos y los casos raros en el tiempo se arman con una planilla de "
            "mediciones: una fila por fecha y una columna por valor. Si podés exportar tus "
            "datos así, lo planteamos con una tabla."
        )
    if problem in VISION:
        return (
            "Ubicar objetos, marcar zonas o leer texto se hace con fotos. Si tenés imágenes, "
            "lo planteamos con fotos."
        )
    return (
        "Esa combinación todavía no tiene recorrido guiado: usá el wizard de las opciones "
        "avanzadas."
    )


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
    # En un pronóstico la respuesta correcta es la propia historia: no se pregunta.
    if f.labels_available is None and f.problem != "forecast":
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
    if f.problem == "forecast" and f.horizon is None:
        out.append("horizon")
    if f.problem == "category" and f.data_kind in ("images", "audio") and f.grouped is None:
        out.append("grouped")
    if f.success_target is None:
        out.append("success_target")
    return out


def to_brief(f: Facts) -> dict[str, Any]:
    """Los hechos como ficha del caso del wizard (`UseCaseBrief`)."""
    other = {
        "locate": "ubicar objetos en fotos",
        "segment": "marcar zonas en fotos",
        "read_text": "leer texto en imágenes",
    }
    brief: dict[str, Any] = {
        "problem": "other" if f.problem in other else f.problem,
        "problem_other": other.get(f.problem or ""),
        "prediction": f.prediction,
        "labels_available": f.labels_available,
        "error_costs": f.error_costs if f.problem == "category" else None,
        "extrapolate": f.extrapolate if f.problem in ("value", "rule") else None,
        "has_entities": f.grouped,
    }
    if f.success_target is not None:
        brief["business_metric"] = f"{f.success_target:g} {target_unit(f)}"
    return {k: v for k, v in brief.items() if v is not None}


def task_of(f: Facts) -> str:
    return TASKS.get(f.problem or "", "regression")


def target_unit(f: Facts) -> str:
    """Qué significa `success_target` en cada problema."""
    return {
        "category": "% de aciertos",
        "anomaly": "% de casos raros detectados",
        "locate": "% de objetos bien ubicados",
        "segment": "% de coincidencia de la zona marcada",
        "read_text": "% de caracteres bien leídos",
    }.get(f.problem or "", "% de error típico como máximo")


def higher_is_better(f: Facts) -> bool:
    """`success_target` es un mínimo (aciertos) o un máximo (error)."""
    return f.problem not in ("value", "rule", "forecast")


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
    if f.problem in SERIES:
        return _series_guide(f, target)
    if f.problem in VISION:
        return _vision_guide(f)
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


def _series_guide(f: Facts, target: str) -> DataGuide:
    """Pronóstico o casos raros: mediciones en el tiempo, una fila por momento."""
    if f.problem == "forecast":
        value = (f.target_name or "").strip() or "ventas"
        cols = ["fecha", "producto", value]
        rows = [
            ["2026-01-05", "A", "120"],
            ["2026-01-12", "A", "135"],
            ["2026-01-05", "B", "40"],
            ["2026-01-12", "B", "38"],
        ]
        return DataGuide(
            layout="table",
            accept=[".csv", ".xlsx", ".xls", ".parquet"],
            steps=[
                "Armá una tabla con una fila por fecha (y por producto, máquina o lugar, si "
                "tenés varios).",
                "Una columna con la fecha, una con qué es (si hay varios) y una con el valor "
                f"(«{value}»).",
                "Guardala como CSV o Excel con los nombres de columna en la primera fila y subila.",
            ],
            example_columns=cols,
            example_rows=rows,
            template_csv=_csv(cols, rows),
            tips=[
                "Al menos 30 fechas por serie; con un par de años de historia el pronóstico "
                "capta mejor las temporadas.",
                "Fechas a intervalos regulares (todos los días, todas las semanas…), sin huecos "
                "grandes.",
                "Si hay datos que se conocen de antemano (precio, feriados, promociones), "
                "agregalos como columnas: ayudan.",
            ],
        )
    marked = f.labels_available is not False
    cols = ["fecha", "equipo", "temperatura", "vibracion", *(["anomalia"] if marked else [])]
    base = [["2026-03-01 10:00", "M1", "61.2", "0.31"], ["2026-03-01 10:05", "M1", "61.5", "0.30"]]
    rows = [r + (["no"] if marked else []) for r in base]
    if marked:
        rows.append(["2026-03-01 10:10", "M1", "79.8", "1.42", "sí"])
    steps = [
        "Armá una tabla con una fila por momento (y por equipo, si tenés varios).",
        "Una columna con la fecha y hora y una columna por cada medición.",
    ]
    if marked:
        steps.append(
            "Agregá la columna «anomalia» con «sí» en los momentos raros que ya conocés y «no» "
            "en el resto."
        )
    steps.append(
        "Guardala como CSV o Excel con los nombres de columna en la primera fila y subila."
    )
    tips = [
        "Al menos 30 momentos por equipo; cuanto más tiempo de funcionamiento normal, mejor "
        "aprende qué es lo habitual.",
        "Mediciones a intervalos regulares (por ejemplo, cada 5 minutos), sin huecos grandes.",
    ]
    if not marked:
        tips.append(
            "No hace falta marcar nada: aprendo cómo es lo normal y te aviso de lo que se "
            "aparta. Si después marcás algunos casos raros que conozcas, puedo medir cuántos "
            "detecta."
        )
    return DataGuide(
        layout="table",
        accept=[".csv", ".xlsx", ".xls", ".parquet"],
        steps=steps,
        example_columns=cols,
        example_rows=rows,
        template_csv=_csv(cols, rows),
        tips=tips,
    )


def _vision_guide(f: Facts) -> DataGuide:
    """Ubicar objetos, marcar zonas o leer texto: fotos con su respuesta en un ZIP."""
    if f.problem == "locate":
        return DataGuide(
            layout="folders",
            accept=[".zip"],
            steps=[
                "Marcá con un recuadro cada objeto en cada foto, con una herramienta de "
                "etiquetado (CVAT, Label Studio, Roboflow) o en «Etiquetado» de las opciones "
                "avanzadas.",
                "Exportá las marcas en formato COCO (un archivo .json), Pascal VOC (.xml) o "
                "YOLO (.txt).",
                "Poné las fotos y el archivo de marcas en un único ZIP y subilo acá.",
            ],
            example_tree=["fotos/pieza_001.jpg", "fotos/pieza_002.jpg", "anotaciones.json"],
            tips=[
                "Al menos 100 objetos marcados de cada tipo; con varios cientos es mucho más "
                "confiable.",
                "Marcá todos los objetos de cada foto: uno sin marcar le enseña que ahí no hay "
                "nada.",
                "Fotos parecidas a las que vas a usar después: misma cámara, luz y distancia.",
            ],
        )
    if f.problem == "segment":
        return DataGuide(
            layout="folders",
            accept=[".zip"],
            steps=[
                "Para cada foto, pintá la zona que importa en una imagen aparte del mismo "
                "tamaño (la máscara): blanco donde está, negro donde no.",
                "Guardá las fotos en «images/» y las máscaras en «masks/», con el mismo nombre "
                "de archivo.",
                "Comprimí las dos carpetas en un único ZIP y subilo acá.",
            ],
            example_tree=[
                "images/foto_001.png",
                "images/foto_002.png",
                "masks/foto_001.png",
                "masks/foto_002.png",
            ],
            tips=[
                "Al menos 50 fotos con su máscara; con varios cientos el contorno sale mucho "
                "más preciso.",
                "La máscara tiene que coincidir pixel a pixel con la foto (mismo tamaño y "
                "encuadre).",
                "Si hay varios tipos de zona, usá un tono de gris distinto para cada uno.",
            ],
        )
    cols = ["file_name", "texto"]
    rows = [["etiqueta_001.png", "LOTE 4471-B"], ["etiqueta_002.png", "VTO 03/2027"]]
    return DataGuide(
        layout="folders",
        accept=[".zip"],
        steps=[
            "Guardá las imágenes con el texto a leer en una carpeta «images/».",
            "Armá una planilla «labels.csv» con dos columnas: el nombre de cada imagen y el "
            "texto exacto que se ve.",
            "Comprimí la carpeta y la planilla en un único ZIP y subilo acá.",
        ],
        example_columns=cols,
        example_rows=rows,
        example_tree=["images/etiqueta_001.png", "images/etiqueta_002.png", "labels.csv"],
        template_csv=_csv(cols, rows),
        tips=[
            "Al menos 200 imágenes con su texto; con miles lee mucho mejor.",
            "Recortes que muestren solo el texto (una línea por imagen funciona mejor).",
            "Escribí el texto tal cual se ve: mayúsculas, guiones y espacios incluidos.",
        ],
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
        out, explanation, meets = _summarize_task(report, f, target)
    out["meets_target"] = bool(meets)
    out["explanation"] = explanation
    out["next_steps"] = next_steps(out, f)
    return out


def _summarize_task(
    report: EvaluationReport, f: Facts, target: float
) -> tuple[dict[str, Any], list[str], bool]:
    """Pronóstico, casos raros, ubicar objetos, marcar zonas y leer texto, en palabras."""
    m = report.metrics
    task = report.task.value
    n = report.num_samples
    if task == "forecasting":
        smape = float(m.get("smape", 0.0))
        skill = float(m.get("skill_vs_naive", 0.0))
        out: dict[str, Any] = {"kind": task, "total": n, "typical_percent": round(smape, 2)}
        lines = [
            f"Pronosticando períodos que el modelo nunca vio, se equivoca en promedio un "
            f"{smape:.1f} % del valor real."
        ]
        if skill > 0:
            lines.append(
                f"Es un {100 * skill:.0f} % más preciso que repetir el último valor conocido."
            )
        else:
            lines.append(
                "Todavía no le gana a repetir el último valor conocido: hace falta más historia "
                "o datos que expliquen los cambios."
            )
        return out, lines, smape <= target and skill > 0
    if task == "anomaly_detection":
        rate = 100 * float(m.get("anomaly_rate", 0.0))
        if "recall" in m:
            recall, precision = 100 * float(m["recall"]), 100 * float(m.get("precision", 0.0))
            out = {
                "kind": task,
                "total": n,
                "percent": round(recall, 1),
                "precision": round(precision, 1),
            }
            lines = [
                f"En un período que el modelo nunca vio, detectó el {recall:.0f} % de los "
                "casos raros marcados.",
                f"De lo que marcó como raro, el {precision:.0f} % lo era de verdad (el resto "
                "son falsas alarmas).",
            ]
            return out, lines, recall >= target
        out = {"kind": task, "total": n, "anomaly_rate": round(rate, 2), "measurable": False}
        lines = [
            "Aprendió cómo es el funcionamiento normal y marca lo que se aparta: en un "
            f"período que nunca vio, marcó como raro el {rate:.1f} % de los momentos.",
            "Como no había casos raros marcados, no puedo medir cuántos detecta: validalo con "
            "momentos que conozcas.",
        ]
        return out, lines, True
    if task == "object_detection":
        found = 100 * float(m.get("map_50", 0.0))
        out = {"kind": task, "total": n, "percent": round(found, 1)}
        lines = [
            f"En {n} fotos que nunca vio, ubica bien el {found:.0f} % de los objetos (con el "
            "recuadro en el lugar correcto)."
        ]
        per_class = report.details.get("ap_per_class") or {}
        if isinstance(per_class, dict) and len(per_class) > 1:
            worst = min(per_class.items(), key=lambda kv: float(kv[1]))
            lines.append(
                f"Lo que más le cuesta ubicar es «{worst[0]}» ({100 * float(worst[1]):.0f} %)."
            )
        return out, lines, found >= target
    if task == "segmentation":
        iou = 100 * float(m.get("iou_foreground", m.get("mean_iou", 0.0)))
        out = {"kind": task, "total": n, "percent": round(iou, 1)}
        lines = [
            f"En {n} fotos que nunca vio, la zona que marca coincide en promedio un {iou:.0f} % "
            "con la zona real."
        ]
        return out, lines, iou >= target
    if task == "ocr":
        chars = 100 * (1 - float(m.get("cer", 1.0)))
        exact = 100 * float(m.get("exact_match", 0.0))
        out = {"kind": task, "total": n, "percent": round(chars, 1), "exact": round(exact, 1)}
        lines = [
            f"En {n} imágenes que nunca vio, lee bien el {chars:.0f} % de los caracteres.",
            f"Lee completo y sin errores el {exact:.0f} % de los textos.",
        ]
        return out, lines, chars >= target
    return {"kind": task, "total": n}, ["El modelo se evaluó con casos que nunca vio."], False


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
    kind = result.get("kind")
    if kind == "classification":
        steps.append(
            "Lo que más ayuda: más ejemplos de las categorías que más le cuestan, y que se "
            "parezcan a los casos reales."
        )
    elif kind in ("forecasting", "anomaly_detection"):
        steps.append(
            "Lo que más ayuda: más tiempo de historia, mediciones sin huecos y columnas que "
            "expliquen los cambios (precios, turnos, clima)."
        )
    elif kind in ("object_detection", "segmentation", "ocr"):
        steps.append(
            "Lo que más ayuda: más fotos marcadas, parecidas a las reales (misma cámara, luz y "
            "distancia), con las marcas bien precisas."
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

    kind: Literal["fields", "image", "audio", "text", "series"]
    fields: list[InputField] = Field(default_factory=list)
    task: str | None = Field(default=None, description="Qué devuelve el modelo (TaskType)")
    history: int | None = Field(
        default=None, description="Series: cuántos valores recientes hacen falta por serie"
    )
    horizon: int | None = Field(default=None, description="Pronóstico: valores hacia adelante")


def input_form(
    signature: dict[str, Any],
    card: ProfileCard,
    example: dict[str, Any],
    series: SeriesProfile | None = None,
) -> InputForm:
    from perceptron.data.schema import SemanticType

    kind = str(signature.get("inputs", {}).get("kind"))
    task = str(signature.get("outputs", {}).get("task") or "") or None
    if kind == "image":
        return InputForm(kind="image", task=task)
    if kind == "spectrogram":
        return InputForm(kind="audio", task=task)
    if kind == "tokens":
        return InputForm(kind="text", task=task)
    if kind == "sequence" or series is not None:
        # La historia reciente, con las mismas columnas que los datos que subió la persona.
        target_name = card.target.name if card.target else ""
        columns = [
            InputField(name=c.name, kind=_field_kind(c.semantic))
            for c in card.columns
            if c.semantic is not SemanticType.ID
            or c.name == (card.target.name if card.target else "")
        ]
        if card.target and card.target.name not in {f.name for f in columns}:
            columns.append(InputField(name=card.target.name, kind="number"))
        return InputForm(
            kind="series",
            # En casos raros la marca (si la hay) no se pide: es lo que el modelo detecta.
            fields=[f for f in columns if task == "forecasting" or f.name != target_name],
            task=task,
            history=series.lookback if series else None,
            horizon=series.horizon if series and task == "forecasting" else None,
        )
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
    return InputForm(kind="fields", fields=fields, task=task)


def _field_kind(semantic: Any) -> Literal["number", "category", "boolean", "date", "text"]:
    from perceptron.data.schema import SemanticType

    return {
        SemanticType.NUMERIC: "number",
        SemanticType.BOOLEAN: "boolean",
        SemanticType.DATETIME: "date",
        SemanticType.CATEGORICAL: "category",
    }.get(semantic, "text")  # type: ignore[return-value]
