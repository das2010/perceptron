"""Lo que dice el asistente guiado cuando no habla el LLM (ADR-0042), sin efectos.

La conversación tiene que sentirse natural también sin proveedor de IA, en L0 o si el LLM
falla: cada pregunta está redactada para el caso de la persona, cada respuesta se acusa y cada
etapa del recorrido (datos, revisión, entrenamiento, validación) se cuenta en el mismo hilo. El
LLM, cuando está, reemplaza estas frases por otras más ajustadas; nunca decide transiciones.
"""

from __future__ import annotations

from typing import Any, Literal

from perceptron.services.journey_guide import VALIDATION_MIN, Facts

Language = Literal["es", "en"]


def _lang(language: str) -> Language:
    return "en" if language.startswith("en") else "es"


def _join(items: list[str], language: str, last: str | None = None) -> str:
    """«a», «b» y «c» (o “a”, “b” and “c”)."""
    en = _lang(language) == "en"
    quoted = [f"“{i}”" if en else f"«{i}»" for i in items]
    word = last or ("and" if en else "y")
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} {word} {quoted[-1]}"


def _about(f: Facts, language: str) -> str:
    """«qué se predice» con las palabras de la persona, para armar las preguntas."""
    if f.prediction:
        return f.prediction.rstrip(" .")
    return "the result" if _lang(language) == "en" else "el resultado"


# ---------------------------------------------------------------------- preguntas

# Ejemplos según el tipo de datos: (dos respuestas, qué se predice, respuestas posibles).
_EXAMPLES_ES = {
    "table": (
        "«se va» o «se queda»",
        "«si el cliente se va a dar de baja el mes que viene»",
        "«se va» y «se queda»",
    ),
    "images": (
        "«con fisura» o «sana»",
        "«si la pieza de la foto tiene una fisura»",
        "«con fisura» y «sin fisura»",
    ),
    "text": (
        "«reclamo» o «consulta»",
        "«de qué tema es cada mensaje»",
        "«facturación», «envíos» y «garantía»",
    ),
    "audio": (
        "«falla» o «normal»",
        "«si el motor de la grabación tiene una falla»",
        "«normal» y «falla»",
    ),
}
_EXAMPLES_EN = {
    "table": (
        "“leaves” or “stays”",
        "“whether the customer will cancel next month”",
        "“leaves” and “stays”",
    ),
    "images": (
        "“cracked” or “fine”",
        "“whether the part in the photo has a crack”",
        "“cracked” and “not cracked”",
    ),
    "text": (
        "“complaint” or “question”",
        "“what topic each message is about”",
        "“billing”, “shipping” and “warranty”",
    ),
    "audio": (
        "“fault” or “normal”",
        "“whether the engine in the recording has a fault”",
        "“normal” and “fault”",
    ),
}


def question(field: str, f: Facts, language: str = "es") -> str:
    """La próxima pregunta, redactada para el caso, con un ejemplo concreto."""
    about = _about(f, language)
    if _lang(language) == "en":
        two, what, answers = _EXAMPLES_EN.get(f.data_kind or "", _EXAMPLES_EN["table"])
        en = {
            "data_kind": "To start: how do you keep that information? For example in a "
            "spreadsheet (Excel or CSV), as photos, as texts or as recordings.",
            "problem": "What would you like the system to tell you for each case? It can pick "
            f"one of several answers (like {two}), estimate a number (like a "
            "price) or discover the formula behind a result.",
            "prediction": "Tell me in your own words what you want to know about each case. "
            f"For example: {what}.",
            "labels_available": {
                "images": "Do you already have the photos sorted by the right answer? For "
                "example, one folder per answer.",
                "audio": "Do you already have the recordings sorted by the right answer? For "
                "example, one folder per answer.",
                "text": "Do your texts already have the right answer? For example, each "
                "message with its topic.",
            }.get(
                f.data_kind or "",
                f"Does your spreadsheet already show {about} for past cases? For example, a "
                "column that says what actually happened.",
            ),
            "classes": f"What are the possible answers? For example: {answers}.",
            "error_costs": "No system is right every time. What worries you more: missing a "
            "case that did matter, or getting warnings about cases that turn out fine?",
            "positive_class": (
                f"Which one matters most to catch: {_join(f.classes, language, 'or')}?"
                if f.classes
                else "Which answer matters most to catch?"
            ),
            "extrapolate": "Will you use it with cases similar to the ones you have, or with "
            "values bigger or smaller than those in your data?",
            "grouped": (
                "Do you have several recordings of the same machine, person or session? "
                if f.data_kind == "audio"
                else "Do you have several photos of the same object, for example the same part "
                "from different angles? "
            )
            + "I need it to test the model fairly, on objects it never saw.",
            "success_target": (
                "How often would it need to be right to be useful? For example, 8 out of 10 (80 %)."
                if f.problem == "category"
                else "How far off can it be and still be useful? For example, within 10 % of "
                "the real value."
            ),
        }
        return en.get(field, "Tell me a bit more about your case.")
    two, what, answers = _EXAMPLES_ES.get(f.data_kind or "", _EXAMPLES_ES["table"])
    es = {
        "data_kind": "Para empezar: ¿cómo tenés guardada esa información? Por ejemplo, en una "
        "planilla (Excel o CSV), en fotos, en textos o en grabaciones.",
        "problem": "¿Qué te gustaría que te diga el sistema para cada caso? Puede elegir entre "
        f"varias respuestas (como {two}), estimar un número (como un precio) o "
        "descubrir la fórmula que hay detrás de un resultado.",
        "prediction": "Contame con tus palabras qué querés saber de cada caso. Por ejemplo: "
        f"{what}.",
        "labels_available": {
            "images": "¿Ya tenés las fotos separadas según la respuesta correcta? Por ejemplo, "
            "una carpeta para cada respuesta.",
            "audio": "¿Ya tenés las grabaciones separadas según la respuesta correcta? Por "
            "ejemplo, una carpeta para cada respuesta.",
            "text": "¿Tus textos ya tienen la respuesta correcta? Por ejemplo, cada mensaje con "
            "su tema.",
        }.get(
            f.data_kind or "",
            f"En tu planilla, ¿ya figura {about} en casos pasados? Por ejemplo, una columna que "
            "diga qué pasó en cada caso.",
        ),
        "classes": f"¿Cuáles son las respuestas posibles? Por ejemplo: {answers}.",
        "error_costs": "Ningún sistema acierta siempre. ¿Qué te preocupa más: que se le escape "
        "un caso que sí importaba, o que te avise de más con casos que estaban bien?",
        "positive_class": (
            f"¿Cuál es la que más te importa detectar: {_join(f.classes, language, 'o')}?"
            if f.classes
            else "¿Cuál de las respuestas es la que más te importa detectar?"
        ),
        "extrapolate": "¿Lo vas a usar con casos parecidos a los que ya tenés, o con valores más "
        "grandes o más chicos que los de tus datos?",
        "grouped": (
            "¿Tenés varias grabaciones de la misma máquina, persona o sesión? "
            if f.data_kind == "audio"
            else "¿Tenés varias fotos del mismo objeto, por ejemplo la misma pieza desde distintos "
            "ángulos? "
        )
        + "Lo necesito para probar el modelo de forma justa, con objetos que nunca vio.",
        "success_target": (
            "¿Qué tan seguido tendría que acertar para que te sirva? Por ejemplo, 8 de cada 10 "
            "(80 %)."
            if f.problem == "category"
            else "¿Cuánto se puede equivocar y que igual te sirva? Por ejemplo, un 10 % del valor "
            "real."
        ),
    }
    return es.get(field, "Contame un poco más de tu caso.")


def greeting(language: str = "es") -> str:
    if _lang(language) == "en":
        return "Hi! I'll help you build it step by step."
    return "¡Hola! Te ayudo a armarlo paso a paso."


def acknowledge(turn: int, language: str = "es") -> str:
    """Un acuse breve y variado, para que la respuesta no suene a formulario."""
    words = (
        ["Perfect.", "Got it.", "Great.", "Understood."]
        if _lang(language) == "en"
        else ["Perfecto.", "Entendido.", "Bien.", "Anotado."]
    )
    return words[turn % len(words)]


def summary(f: Facts, language: str = "es") -> str:
    """Lo que entendió el asistente, en una frase, para que la persona lo confirme."""
    en = _lang(language) == "en"
    kind = (
        {"table": "a spreadsheet", "images": "photos", "text": "texts", "audio": "recordings"}
        if en
        else {"table": "una planilla", "images": "fotos", "text": "textos", "audio": "grabaciones"}
    ).get(f.data_kind or "", "your data" if en else "tus datos")
    about = _about(f, language)
    if en:
        goal = {
            "category": "you want to know this about each case",
            "value": "you want to estimate this",
            "rule": "you want to discover the rule behind this",
        }.get(f.problem or "", "you want to predict this")
        parts = [f"Let me check I got it right: you have {kind} and {goal}: {about}."]
        if f.classes:
            parts.append(f"The possible answers are {_join(f.classes, language)}.")
        if f.error_costs == "false_negative_worse":
            parts.append("Missing an important case is worse than a false alarm.")
        elif f.error_costs == "false_positive_worse":
            parts.append("A false alarm is worse than missing a case.")
        if f.success_target is not None:
            parts.append(
                f"It is useful if it is right at least {f.success_target:g} % of the time."
                if f.problem == "category"
                else f"It is useful if it is off by at most {f.success_target:g} %."
            )
        parts.append("Is that right? If something is off, just tell me.")
        return " ".join(parts)
    goal = {
        "category": "querés saber esto de cada caso",
        "value": "querés estimar esto",
        "rule": "querés descubrir la regla detrás de esto",
    }.get(f.problem or "", "querés predecir esto")
    parts = [f"A ver si entendí bien: tenés {kind} y {goal}: {about}."]
    if f.classes:
        parts.append(f"Las respuestas posibles son {_join(f.classes, language)}.")
    if f.error_costs == "false_negative_worse":
        parts.append("Que se escape un caso importante es peor que una falsa alarma.")
    elif f.error_costs == "false_positive_worse":
        parts.append("Una falsa alarma es peor que dejar pasar un caso.")
    if f.success_target is not None:
        parts.append(
            f"Te sirve si acierta al menos el {f.success_target:g} % de las veces."
            if f.problem == "category"
            else f"Te sirve si se equivoca como mucho un {f.success_target:g} %."
        )
    parts.append("¿Es así? Si algo no está bien, decímelo.")
    return " ".join(parts)


# ---------------------------------------------------------------------- etapas


def after_confirm(language: str = "es") -> str:
    if _lang(language) == "en":
        return (
            "Great, the case is clear. Now I need your data: below you can see how it should "
            "look, with an example and a template. Upload it here when you have it."
        )
    return (
        "¡Bien! Ya tengo claro el caso. Ahora necesito tus datos: abajo te muestro cómo tienen "
        "que quedar, con un ejemplo y una plantilla. Cuando los tengas, subilos acá."
    )


def uploaded(name: str, language: str = "es") -> str:
    return f"Here is «{name}»." if _lang(language) == "en" else f"Te paso «{name}»."


def reviewed(summary_: dict[str, Any], warnings: list[str], language: str = "es") -> str:
    rows = int(summary_.get("rows") or 0)
    target = summary_.get("target")
    if _lang(language) == "en":
        text = f"I checked your data: {rows} examples" + (
            f", and the result to predict is in «{target}»." if target else "."
        )
        if warnings:
            text += " There are a few things to keep in mind (listed below)."
        return text + (
            " Everything is ready: I will design the model for your case, train it and test "
            "it on examples it never saw. Shall I start?"
        )
    text = f"Revisé tus datos: {rows} ejemplos" + (
        f", y el resultado a predecir está en «{target}»." if target else "."
    )
    if warnings:
        text += " Hay algunas cosas para tener en cuenta (te las dejo abajo)."
    return text + (
        " Está todo listo: diseño el modelo para tu caso, lo entreno y lo pruebo con ejemplos "
        "que nunca vio. ¿Arranco?"
    )


def blocked(blockers: list[str], has_columns: bool, language: str = "es") -> str:
    items = " ".join(f"• {b}" for b in blockers)
    if _lang(language) == "en":
        tail = " Fix the file and upload it again" + (
            ", or tell me below which column holds the result." if has_columns else "."
        )
        return f"I checked your data, but before going on we need to fix this: {items}{tail}"
    tail = " Corregí el archivo y volvé a subirlo" + (
        ", o elegí abajo cuál columna tiene el resultado." if has_columns else "."
    )
    return f"Revisé tus datos, pero antes de seguir hay que resolver esto: {items}{tail}"


def training_started(language: str = "es") -> str:
    if _lang(language) == "en":
        return (
            "Starting! I'll keep you posted on each step. You can close this screen: I'll let "
            "you know when it's done."
        )
    return (
        "¡Arranco! Te voy contando cada paso. Podés cerrar esta pantalla: te aviso cuando termine."
    )


def trained(result: dict[str, Any], language: str = "es") -> str:
    lines = list(result.get("explanation") or [])
    en = _lang(language) == "en"
    if result.get("meets_target"):
        lines.append(
            "It meets the goal you set. Now let's test it on new cases, ones that are not in "
            "the data you uploaded, to make sure it works in practice."
            if en
            else "Cumple el objetivo que definiste. Ahora probémoslo con casos nuevos, que no "
            "estén en los datos que subiste, para confirmar que funciona en la práctica."
        )
    else:
        lines.extend(result.get("next_steps") or [])
    return " ".join(lines)


def failed(error: str | None, language: str = "es") -> str:
    if _lang(language) == "en":
        return (
            f"Something went wrong: {error or 'unknown error'}. Your data is safe: shall I "
            "try again?"
        )
    return (
        f"Algo falló: {error or 'error desconocido'}. Tus datos quedan guardados: ¿lo "
        "intento de nuevo?"
    )


def interrupted(language: str = "es") -> str:
    if _lang(language) == "en":
        return "I stopped the work. Your data is safe; whenever you want, I'll pick it up again."
    return "Detuve el trabajo. Tus datos quedan guardados; cuando quieras, lo retomo."


def validated(validations: list[dict[str, Any]], ready: bool, language: str = "es") -> str:
    total = len(validations)
    ok = sum(1 for v in validations if v.get("correct"))
    if _lang(language) == "en":
        if ready:
            return (
                f"I tested it on {total} new cases and it got {ok} right. It's ready to use: "
                "fill in the data below and I'll tell you the result."
            )
        return (
            f"On new cases it got {ok} out of {total} right, below the goal you set. More "
            "examples like the ones it misses usually help; you can upload new data and retrain."
        )
    if ready:
        return (
            f"Lo probé con {total} casos nuevos y acertó {ok}. Ya podés usarlo: completá los "
            "datos abajo y te digo el resultado."
        )
    return (
        f"Con casos nuevos acertó {ok} de {total}, por debajo del objetivo que definiste. Suele "
        "ayudar sumar ejemplos parecidos a los que falla; podés subir datos nuevos y volver a "
        "entrenar."
    )


def step_help(state: str, language: str = "es") -> str:
    """Respuesta cuando la persona escribe y no hay LLM: qué se puede hacer ahora."""
    en = _lang(language) == "en"
    es_help = {
        "awaiting_data": "Ahora necesito tus datos: seguí el ejemplo que te mostré y subilos ahí.",
        "reviewing_data": "Estoy revisando tus datos; en un momento te cuento qué encontré.",
        "ready_to_train": "Tus datos están listos. Cuando quieras, tocá «Preparar y entrenar».",
        "training": "Estoy entrenando el modelo; te aviso cuando termine.",
        "ready_to_validate": (
            f"Probalo con casos nuevos: completá un caso con su resultado real y lo comparo. "
            f"Con {VALIDATION_MIN} casos queda listo para usar."
        ),
        "needs_improvement": "Podés subir datos nuevos o volver a entrenar desde acá.",
        "ready_to_use": "Completá los datos de un caso en el formulario y te digo el resultado.",
        "failed": "Algo falló; podés reintentar o subir otros datos.",
        "interrupted": "El trabajo quedó detenido; podés retomarlo cuando quieras.",
    }
    en_help = {
        "awaiting_data": "Now I need your data: follow the example I showed you and upload it.",
        "reviewing_data": "I'm checking your data; I'll tell you what I found in a moment.",
        "ready_to_train": "Your data is ready. Whenever you want, click “Prepare and train”.",
        "training": "I'm training the model; I'll let you know when it's done.",
        "ready_to_validate": (
            f"Test it on new cases: enter a case with its real result and I'll compare. With "
            f"{VALIDATION_MIN} cases it's ready to use."
        ),
        "needs_improvement": "You can upload new data or retrain from here.",
        "ready_to_use": "Fill in a case in the form and I'll tell you the result.",
        "failed": "Something went wrong; you can retry or upload other data.",
        "interrupted": "The work was stopped; you can resume it whenever you want.",
    }
    prefix = (
        "I couldn't reach the AI to answer that. "
        if en
        else "No pude consultar a la IA para responderte eso. "
    )
    return prefix + (en_help if en else es_help).get(state, "")


# ---------------------------------------------------------------------- sin LLM: la respuesta

MAX_ANSWER = 500  # una respuesta a la pregunta del asistente; lo demás no se interpreta
_YES = {"si", "sí", "s", "yes", "y", "claro", "dale", "correcto", "exacto", "obvio", "seguro"}
_NO = {"no", "n", "nop", "nope", "para nada"}


def understand(field: str | None, text: str) -> dict[str, Any] | None:
    """Sin LLM: lo que respondió la persona a la pregunta que se le acaba de hacer, cuando se
    puede leer sin ambigüedad (texto libre, lista, porcentaje, sí o no). None si no se entiende:
    entonces se vuelve a preguntar con las respuestas rápidas."""
    # Sin expresiones regulares sobre lo que escribe la persona (nada de ReDoS): texto acotado,
    # espacios normalizados y palabras.
    answer = " ".join(text[:MAX_ANSWER].split()).strip(" .!¡¿?")
    if not field or not answer:
        return None
    if field in ("prediction", "target_name"):
        return {field: answer[:300]}
    if field in ("classes", "input_names"):
        joined = f" {answer} ".replace(";", ",")
        for word in (" y ", " o ", " and ", " or "):
            joined = joined.replace(word, ",")
        items = list(dict.fromkeys(p.strip(" .«»\"'") for p in joined.split(",") if p.strip()))
        return {field: items} if len(items) >= (2 if field == "classes" else 1) else None
    if field == "positive_class":
        return {field: answer.strip("«»\"'")[:100]}
    if field == "success_target":
        value = _target(answer.casefold())
        return {field: round(value, 1)} if value is not None and 0 < value <= 100 else None
    if field in ("labels_available", "extrapolate", "grouped"):
        words = answer.casefold().replace(",", " ").replace(".", " ").split()
        first = words[0].strip(";:!") if words else ""
        if first in _YES:
            return {field: True}
        if first in _NO or answer.casefold() in _NO:
            return {field: False}
    return None


def _number(word: str) -> float | None:
    try:
        return float(word.strip("%()").replace(",", "."))
    except ValueError:
        return None


def _target(answer: str) -> float | None:
    """«8 de cada 10», «8/10» o «80 %» → 80."""
    for sep in (" de cada ", " out of ", " of every ", "/"):
        if sep in answer:
            left, right = answer.split(sep, 1)
            num = _number(left.split()[-1]) if left.split() else None
            den = _number(right.split()[0]) if right.split() else None
            if num is not None and den:
                return 100 * num / den
    for word in answer.replace("%", " ").split():
        value = _number(word)
        if value is not None:
            return value
    return None
