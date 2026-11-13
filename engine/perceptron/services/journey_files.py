"""Los archivos que sube la persona, leídos para proponerle qué puede aprender la red (RF-WIZ-01,
RF-ING-01/07).

Sin efectos y determinístico: se detecta qué hay (planillas, fotos por carpeta, máscaras, textos,
grabaciones), se resume sin valores individuales y se arman objetivos posibles, cada uno con los
hechos del caso que implica (`Facts`). Con varias planillas se reconoce si son partes de una
misma tabla (mismas columnas: se juntan) o si se relacionan por una columna en común que en una
de ellas es única (se unen, como clientes y sus compras). El LLM, cuando está, solo ordena y
explica estas propuestas: nunca inventa columnas ni objetivos.
"""

from __future__ import annotations

import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from perceptron.data.audio import AUDIO_EXTENSIONS
from perceptron.data.schema import IMAGE_EXTENSIONS, SemanticType, TableSchema, infer_schema
from perceptron.data.series import detect_series
from perceptron.data.sources.files import TABLE_SUFFIXES, _safe_extract, scan_table
from perceptron.domain.enums import TaskType
from perceptron.services.journey_guide import Facts

FileKind = Literal["table", "images", "audio", "text", "segment", "read_text", "locate", "unknown"]
Combine = Literal["stack", "join"]
SAMPLE_ROWS = 5000
MAX_GOALS = 8
MAX_CLASSES = 30
MIN_FILLED = 0.5
_ANOMALY = ("anomal", "falla", "fault", "outlier", "raro", "alarm")
_SKIP = (".", "__MACOSX")
_TABLES = TABLE_SUFFIXES - {".txt"}  # en carpetas, un .txt es un texto


class ColumnInfo(BaseModel):
    name: str
    type: str
    distinct: int
    filled: float = Field(ge=0, le=1, description="Fracción de filas con dato")


class FileInfo(BaseModel):
    """Lo que se leyó de un archivo o carpeta, sin valores individuales."""

    name: str
    kind: FileKind
    rows: int | None = None
    columns: list[ColumnInfo] = Field(default_factory=list)
    files: int | None = Field(default=None, description="Fotos, grabaciones o textos")
    classes: dict[str, int] = Field(default_factory=dict, description="Carpeta → cantidad")
    error: str | None = None


class GoalProposal(BaseModel):
    """Un objetivo posible para la red, con los hechos del caso que implica."""

    id: str
    title: str
    detail: str
    facts: Facts
    target: str | None = Field(default=None, description="Columna con el resultado (tablas)")
    parts: list[str] = Field(
        default_factory=list, description="Planillas usadas (rutas relativas); la primera manda"
    )
    combine: Combine | None = None
    keys: list[str] = Field(default_factory=list, description="Columna de unión de cada parte")


@dataclass
class _Table:
    rel: str
    path: Path
    rows: int
    sample: pl.DataFrame
    schema: TableSchema


@dataclass
class Reading:
    files: list[FileInfo]
    goals: list[GoalProposal]
    combined: str | None = None  # «une a y b por k», «junta los 3 archivos»
    note: str | None = None  # por qué no hay objetivos


# ---------------------------------------------------------------------- lectura


def _visible(rel: str) -> bool:
    return bool(rel) and not any(p.startswith(_SKIP) for p in rel.split("/"))


def _names(path: Path) -> list[str]:
    """Rutas relativas (posix) de lo subido: archivos de una carpeta o miembros de un zip."""
    if path.is_dir():
        names = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()]
    else:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
        tops = {n.split("/")[0] for n in names if _visible(n)}
        if len(tops) == 1 and all("/" in n for n in names if _visible(n)):
            prefix = next(iter(tops)) + "/"
            names = [n[len(prefix) :] for n in names if n.startswith(prefix)]
    return sorted(n for n in names if _visible(n))


def _suffix(name: str) -> str:
    return Path(name).suffix.lower()


def _table(root: Path, rel: str) -> _Table:
    path = root / rel
    lf = scan_table(path)
    sample = lf.head(SAMPLE_ROWS).collect()
    rows = int(lf.select(pl.len()).collect().item())
    return _Table(rel, path, rows, sample, infer_schema(sample))


def _info(t: _Table) -> FileInfo:
    n = max(t.sample.height, 1)
    return FileInfo(
        name=t.rel,
        kind="table",
        rows=t.rows,
        columns=[
            ColumnInfo(
                name=c.name,
                type=c.semantic.value,
                distinct=c.n_unique,
                filled=round(1 - t.sample[c.name].null_count() / n, 3),
            )
            for c in t.schema.columns
        ],
    )


def _folder(name: str, names: list[str]) -> FileInfo:
    """Fotos, grabaciones o textos: qué tarea sugiere la estructura de la carpeta."""
    tops = {n.split("/")[0] for n in names if "/" in n}
    images = [n for n in names if _suffix(n) in IMAGE_EXTENSIONS]
    audio = [n for n in names if _suffix(n) in AUDIO_EXTENSIONS]
    texts = [n for n in names if _suffix(n) == ".txt" and "/" in n]
    if images and {"images", "masks"} <= tops:
        return FileInfo(
            name=name, kind="segment", files=sum(n.startswith("images/") for n in names)
        )
    root_files = [n for n in names if "/" not in n]
    if images and "images" in tops and any(_suffix(n) == ".csv" for n in root_files):
        return FileInfo(name=name, kind="read_text", files=len(images))
    chosen = images or audio or texts
    if not chosen:
        return FileInfo(name=name, kind="unknown")
    classes: dict[str, int] = {}
    for n in chosen:
        if "/" in n:
            label = n.split("/")[-2]
            classes[label] = classes.get(label, 0) + 1
    boxes = images and (
        any(_suffix(n) == ".json" for n in root_files)
        or any(_suffix(n) == ".xml" for n in names)
        or "labels" in tops
    )
    kind: FileKind = "locate" if boxes else "images" if images else "audio" if audio else "text"
    # Con recuadros y además carpetas por respuesta, se puede ubicar o clasificar.
    return FileInfo(name=name, kind=kind, files=len(chosen), classes=dict(sorted(classes.items())))


def read(path: Path, language: str = "es") -> Reading:
    """Lee lo subido (archivo, zip o carpeta) y propone objetivos."""
    root = path if path.is_dir() else path.parent
    if path.is_file() and _suffix(path.name) in TABLE_SUFFIXES:
        rels = [path.name]
    elif path.is_file() and _suffix(path.name) != ".zip":
        info = FileInfo(name=path.name, kind="unknown", error=_unknown(path.name, language))
        return Reading(files=[info], goals=[], note=info.error)
    else:
        names = _names(path)
        tables = [n for n in names if _suffix(n) in _TABLES]
        if names and len(tables) == len(names):
            if path.is_file():
                # Un zip con planillas: se extrae al lado para leerlas una por una (las rutas
                # quedan relativas a la carpeta del zip, como las de un archivo suelto).
                dest = path.parent / path.stem
                with zipfile.ZipFile(path) as zf:
                    _safe_extract(zf, dest)
                rels = sorted(
                    p.relative_to(root).as_posix()
                    for p in dest.rglob("*")
                    if p.is_file() and _suffix(p.name) in _TABLES and _visible(p.name)
                )
            else:
                rels = tables
        else:
            info = _folder(path.name, names)
            if info.kind == "unknown":
                info.error = _unknown(path.name, language)
            goals = _folder_goals(info, language)
            for i, g in enumerate(goals, 1):
                g.id = f"g{i}"
            note = info.error or (None if goals else _no_labels(info, language))
            return Reading(files=[info], goals=goals, note=note)
    read_ok: list[_Table] = []
    infos: list[FileInfo] = []
    for rel in rels:
        try:
            t = _table(root, rel)
        except Exception:  # archivo dañado o formato raro: se informa y se sigue con el resto
            infos.append(FileInfo(name=rel, kind="unknown", error=_unreadable(rel, language)))
            continue
        read_ok.append(t)
        infos.append(_info(t))
    goals, combined = _table_goals(root, read_ok, language)
    for i, g in enumerate(goals, 1):
        g.id = f"g{i}"
    note = None if goals else _no_goals(read_ok, language)
    return Reading(files=infos, goals=goals, combined=combined, note=note)


# ---------------------------------------------------------------------- varias planillas


def combine(root: Path, parts: list[str], how: Combine | None, keys: list[str]) -> pl.LazyFrame:
    """Las planillas como una sola tabla: apiladas (mismas columnas) o unidas por su clave."""
    frames = [scan_table(root / p) for p in parts]
    if how == "stack":
        return pl.concat(frames, how="diagonal_relaxed")
    main = frames[0]
    for rel, frame, key in zip(parts[1:], frames[1:], keys, strict=True):
        taken = set(main.collect_schema().names())
        stem = Path(rel).stem
        # Las columnas con el mismo nombre en las dos planillas se distinguen por el archivo.
        dim = frame.rename(
            {c: f"{c}_{stem}" for c in frame.collect_schema().names() if c != key and c in taken}
        )
        main = main.with_columns(pl.col(key).cast(pl.String)).join(
            dim.with_columns(pl.col(key).cast(pl.String)), on=key, how="left"
        )
    return main


def _joins(tables: list[_Table]) -> tuple[_Table, list[tuple[_Table, str]]]:
    """La planilla principal (la más larga) y las que se le pueden unir por una columna que en
    ellas es única (una fila por cliente, por producto) y aparece en la principal."""
    main = max(tables, key=lambda t: t.rows)
    found: list[tuple[_Table, str]] = []
    for other in tables:
        if other is main:
            continue
        for col in main.sample.columns:
            if col not in other.sample.columns:
                continue
            values = (
                scan_table(other.path).select(pl.col(col).cast(pl.String)).collect().to_series()
            )
            if values.n_unique() != other.rows or values.null_count():
                continue
            seen = main.sample[col].cast(pl.String).drop_nulls()
            if seen.is_empty() or seen.is_in(values.implode()).mean() < 0.5:  # type: ignore[operator]
                continue
            found.append((other, col))
            break
    return main, found


def _table_goals(
    root: Path, tables: list[_Table], language: str
) -> tuple[list[GoalProposal], str | None]:
    if not tables:
        return [], None
    if len(tables) == 1:
        t = tables[0]
        return _goals(t.schema, t.sample, [t.rel], None, [], "", language)[:MAX_GOALS], None
    if len({tuple(sorted(t.sample.columns)) for t in tables}) == 1:
        parts = [t.rel for t in tables]
        sample = combine(root, parts, "stack", []).head(SAMPLE_ROWS).collect()
        note = _stacked(len(parts), language)
        stacked = _goals(infer_schema(sample), sample, parts, "stack", [], "", language)
        return stacked[:MAX_GOALS], note
    main, joins = _joins(tables)
    goals: list[GoalProposal] = []
    combined = None
    if joins:
        parts = [main.rel, *(o.rel for o, _ in joins)]
        keys = [k for _, k in joins]
        sample = combine(root, parts, "join", keys).head(SAMPLE_ROWS).collect()
        combined = _joined(Path(main.rel).name, [(Path(o.rel).name, k) for o, k in joins], language)
        goals = _goals(infer_schema(sample), sample, parts, "join", keys, "", language)
    used = {main.rel, *(o.rel for o, _ in joins)} if joins else set()
    for t in tables:
        if t.rel not in used:
            goals += _goals(
                t.schema, t.sample, [t.rel], None, [], _in(Path(t.rel).name, language), language
            )
    return goals[:MAX_GOALS], combined


# ---------------------------------------------------------------------- propuestas


def _filled(df: pl.DataFrame, col: str) -> float:
    return 1 - df[col].null_count() / max(df.height, 1)


def _goals(
    schema: TableSchema,
    df: pl.DataFrame,
    parts: list[str],
    how: Combine | None,
    keys: list[str],
    where: str,
    language: str,
) -> list[GoalProposal]:
    """Objetivos posibles de una tabla: cada columna que se puede predecir con las demás."""
    en = language.startswith("en")
    cols = schema.columns
    dates = [c.name for c in cols if c.semantic is SemanticType.DATETIME]
    texts = [c.name for c in cols if c.semantic is SemanticType.TEXT]
    # La columna que une planillas identifica filas: no es un dato para aprender.
    usable = [
        c
        for c in cols
        if c.semantic not in (SemanticType.ID, SemanticType.FILEPATH) and c.name not in keys
    ]
    hinted = list(schema.target_candidates)
    order = hinted + [c.name for c in reversed(cols) if c.name not in hinted]
    out: list[GoalProposal] = []
    marked = False

    def add(title: str, detail: str, facts: Facts, target: str | None) -> None:
        out.append(
            GoalProposal(
                id="",  # se numeran al final, entre todas las planillas
                title=title,
                detail=f"{detail} ({where})" if where else detail,
                facts=facts,
                target=target,
                parts=parts,
                combine=how,
                keys=keys,
            )
        )

    for name in order:
        c = schema.column(name)
        if c.semantic in (
            SemanticType.ID,
            SemanticType.FILEPATH,
            SemanticType.DATETIME,
            SemanticType.TEXT,
        ):
            continue
        if name in keys or c.n_unique <= 1 or _filled(df, name) < MIN_FILLED:
            continue
        inputs = [u.name for u in usable if u.name != name]
        if not inputs:
            continue
        human = name.replace("_", " ")
        anomaly_named = any(k in name.casefold() for k in _ANOMALY)
        if c.semantic is SemanticType.BOOLEAN and dates and anomaly_named:
            marked = True
            add(
                "Detect unusual moments" if en else "Detectar momentos raros",
                (
                    f"learns from the ones marked in “{name}”"
                    if en
                    else f"aprende de los marcados en «{name}»"
                ),
                Facts(
                    data_kind="table",
                    problem="anomaly",
                    prediction=human,
                    labels_available=True,
                    target_name=name,
                    input_names=inputs[:40],
                ),
                name,
            )
            continue
        categorical = c.semantic is SemanticType.BOOLEAN or (
            c.semantic is SemanticType.CATEGORICAL and c.n_unique <= MAX_CLASSES
        )
        if categorical:
            text_only = bool(texts) and len(inputs) <= 2
            add(
                f"Predict “{name}”" if en else f"Predecir «{name}»",
                (
                    f"choose among {c.n_unique} possible answers"
                    + (" from the text" if text_only else f" from the other {len(inputs)} columns")
                    if en
                    else f"elegir entre {c.n_unique} respuestas posibles"
                    + (
                        " a partir del texto"
                        if text_only
                        else f" con las otras {len(inputs)} columnas"
                    )
                ),
                Facts(
                    data_kind="text" if text_only else "table",
                    problem="category",
                    prediction=human,
                    labels_available=True,
                    target_name=name,
                    input_names=inputs[:40],
                ),
                name,
            )
            continue
        if c.semantic is not SemanticType.NUMERIC:
            continue
        cfg = detect_series(schema.model_copy(update={"target": name}), df) if dates else None
        if cfg is not None and cfg.task is TaskType.FORECASTING:
            by = (
                (f" for each “{cfg.series_id}”" if en else f" para cada «{cfg.series_id}»")
                if cfg.series_id
                else ""
            )
            add(
                f"Forecast “{name}”" if en else f"Pronosticar «{name}»",
                (
                    f"the next values from its history{by}"
                    if en
                    else f"los próximos valores según su historia{by}"
                ),
                Facts(
                    data_kind="table",
                    problem="forecast",
                    prediction=human,
                    target_name=name,
                    input_names=[cfg.time_column, *([cfg.series_id] if cfg.series_id else [])],
                ),
                name,
            )
        add(
            f"Estimate “{name}”" if en else f"Estimar «{name}»",
            (
                f"a number, from the other {len(inputs)} columns"
                if en
                else f"un número, con las otras {len(inputs)} columnas"
            ),
            Facts(
                data_kind="table",
                problem="value",
                prediction=human,
                labels_available=True,
                target_name=name,
                input_names=inputs[:40],
            ),
            name,
        )
    measures = [c.name for c in cols if c.semantic is SemanticType.NUMERIC]
    if dates and measures and not marked:
        shown = measures[:3]
        names_ = ", ".join(f"“{m}”" if en else f"«{m}»" for m in shown)
        add(
            "Detect unusual moments" if en else "Detectar momentos raros",
            (
                f"in {names_}, learning what normal looks like (nothing has to be marked)"
                if en
                else f"en {names_}, aprendiendo cómo es lo normal (sin marcar nada)"
            ),
            Facts(
                data_kind="table",
                problem="anomaly",
                prediction=", ".join(m.replace("_", " ") for m in shown),
                labels_available=False,
                input_names=measures[:40],
            ),
            None,
        )
    return out


def _folder_goals(info: FileInfo, language: str) -> list[GoalProposal]:
    """Fotos con marcas (recuadros, máscaras, textos) y carpetas por respuesta."""
    en = language.startswith("en")
    n = info.files or 0
    out: list[GoalProposal] = []
    if info.kind in ("segment", "read_text", "locate"):
        title, detail = {
            "segment": (
                ("Mark the area on each photo", f"{n} photos with their masks")
                if en
                else ("Marcar la zona en cada foto", f"{n} fotos con sus máscaras")
            ),
            "read_text": (
                ("Read the text in each image", f"{n} images with their text")
                if en
                else ("Leer el texto de cada imagen", f"{n} imágenes con su texto")
            ),
            "locate": (
                ("Locate the objects on each photo", f"{n} photos with their boxes")
                if en
                else ("Ubicar los objetos en cada foto", f"{n} fotos con sus recuadros")
            ),
        }[info.kind]
        problem: Literal["segment", "read_text", "locate"] = info.kind
        out.append(
            GoalProposal(
                id="",
                title=title,
                detail=detail,
                facts=Facts(
                    data_kind="images",
                    problem=problem,
                    prediction=title[:1].lower() + title[1:],
                    labels_available=True,
                ),
            )
        )
    classes = [c for c in info.classes if c]
    if info.kind in ("segment", "read_text") or not 2 <= len(classes) <= 100:
        return out
    kind: Literal["images", "audio", "text"] = (
        "audio" if info.kind == "audio" else "text" if info.kind == "text" else "images"
    )
    title, what = {
        "images": (("Classify the photos", "photos"), ("Clasificar las fotos", "fotos")),
        "audio": (
            ("Classify the recordings", "recordings"),
            ("Clasificar las grabaciones", "grabaciones"),
        ),
        "text": (("Classify the texts", "texts"), ("Clasificar los textos", "textos")),
    }[kind][0 if en else 1]
    shown = ", ".join(classes[:6]) + ("…" if len(classes) > 6 else "")
    out.append(
        GoalProposal(
            id="",
            title=title,
            detail=(
                f"into {len(classes)} answers ({shown}), {n} {what}"
                if en
                else f"en {len(classes)} respuestas ({shown}), {n} {what}"
            ),
            facts=Facts(
                data_kind=kind,
                problem="category",
                prediction=(
                    f"which of the {len(classes)} answers fits each one"
                    if en
                    else f"cuál de las {len(classes)} respuestas corresponde a cada una"
                ),
                labels_available=True,
                classes=classes,
            ),
        )
    )
    return out


def materialize(root: Path, goal: GoalProposal, dest: Path) -> Path:
    """La tabla del objetivo elegido como un solo archivo (si combina varias planillas)."""
    if len(goal.parts) <= 1:
        return root / goal.parts[0] if goal.parts else root
    dest.parent.mkdir(parents=True, exist_ok=True)
    combine(root, goal.parts, goal.combine, goal.keys).sink_parquet(dest)
    return dest


def mentioned(message: str, goals: list[GoalProposal]) -> GoalProposal | None:
    """El objetivo que la persona nombró con sus palabras: la columna (o el título) de uno solo.

    Si la misma columna se puede pronosticar o estimar, deciden las palabras («pronóstico»,
    «próximos» → pronóstico). Ante la duda, None: se vuelve a preguntar."""
    said = _plain(message)
    hits = [
        g
        for g in goals
        if _plain(g.title) in said
        or (g.target and len(g.target) >= 3 and _plain(g.target.replace("_", " ")) in said)
        or (g.target and len(g.target) >= 3 and _plain(g.target) in said)
    ]
    if len({g.id for g in hits}) <= 1:
        return hits[0] if hits else None
    future = any(w in said for w in ("pronost", "proxim", "futur", "forecast", "next", "venir"))
    narrowed = [g for g in hits if (g.facts.problem == "forecast") == future]
    return narrowed[0] if len(narrowed) == 1 else None


def consistent(facts: Facts, goals: list[GoalProposal]) -> list[GoalProposal]:
    """Las propuestas que no contradicen lo que ya contó la persona (tipo de datos, problema y
    columna del resultado)."""
    target = _plain(facts.target_name or "")
    return [
        g
        for g in goals
        if (facts.data_kind is None or g.facts.data_kind == facts.data_kind)
        and (facts.problem is None or g.facts.problem == facts.problem)
        and (
            not target
            or not g.target
            or _plain(g.target) == target
            or _plain(g.target.replace("_", " ")) == target
        )
    ]


def _plain(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(ch for ch in folded if not unicodedata.combining(ch)).split())


# ---------------------------------------------------------------------- textos


def _in(rel: str, language: str) -> str:
    return f"in “{rel}”" if language.startswith("en") else f"en «{rel}»"


def _stacked(n: int, language: str) -> str:
    return f"the {n} files together" if language.startswith("en") else f"los {n} archivos juntos"


def _joined(main: str, others: list[tuple[str, str]], language: str) -> str:
    en = language.startswith("en")
    parts = ", ".join(
        (f"“{rel}” on “{key}”" if en else f"«{rel}» por «{key}»") for rel, key in others
    )
    return f"“{main}” joined with {parts}" if en else f"«{main}» unido con {parts}"


def _unknown(name: str, language: str) -> str:
    if language.startswith("en"):
        return (
            f"I can't read “{name}”: I read spreadsheets (CSV, Excel, Parquet, JSON) and "
            "folders or zips of photos, recordings or texts."
        )
    return (
        f"No puedo leer «{name}»: leo planillas (CSV, Excel, Parquet, JSON) y carpetas o zip "
        "de fotos, grabaciones o textos."
    )


def _unreadable(name: str, language: str) -> str:
    if language.startswith("en"):
        return f"I couldn't open “{name}”: check that it isn't damaged or password protected."
    return f"No pude abrir «{name}»: fijate que no esté dañado ni protegido con contraseña."


def _no_labels(info: FileInfo, language: str) -> str:
    en = language.startswith("en")
    what = {"images": "photos", "audio": "recordings", "text": "texts"}.get(info.kind, "files")
    if en:
        return (
            f"I see {info.files or 0} {what}, but not sorted by answer. To learn, put them in "
            "one folder per answer (for example “ok/” and “defect/”) and upload them again, "
            "or tell me what you want to achieve."
        )
    que = {"images": "fotos", "audio": "grabaciones", "text": "textos"}.get(info.kind, "archivos")
    return (
        f"Veo {info.files or 0} {que}, pero no están separadas por respuesta. Para aprender, "
        "ponelas en una carpeta por respuesta (por ejemplo «sana/» y «con_falla/») y subilas "
        "de nuevo, o contame qué querés lograr."
    )


def _no_goals(tables: list[_Table], language: str) -> str:
    if language.startswith("en"):
        return (
            "I couldn't find a column to predict from the others (they look like identifiers, "
            "free text or almost empty). Tell me what you want to achieve and we'll see."
            if tables
            else "I couldn't open any of the files."
        )
    return (
        "No encontré una columna que se pueda predecir con las demás (parecen identificadores, "
        "texto libre o están casi vacías). Contame qué querés lograr y lo vemos."
        if tables
        else "No pude abrir ninguno de los archivos."
    )
