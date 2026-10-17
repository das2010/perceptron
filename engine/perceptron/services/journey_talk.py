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


def _about(f: Facts, language: str) -> str:
    """«qué se predice» con las palabras de la persona, para armar las preguntas."""
    if f.prediction:
        return f.prediction.rstrip(" .")
    return "the result" if _lang(language) == "en" else "el resultado"


# ---------------------------------------------------------------------- preguntas


def question(field: str, f: Facts, language: str = "es") -> str:
    """La próxima pregunta, redactada para el caso, con un ejemplo concreto."""
    about = _about(f, language)
    if _lang(language) == "en":
        en = {
            "data_kind": "To start: how do you keep that information? For example in a "
            "spreadsheet (Excel or CSV), as photos, as texts or as recordings.",
            "problem": "What would you like the system to tell you for each case? It can pick "
            "one of several answers (like “leaves” or “stays”), estimate a number (like a "
            "price) or discover the formula behind a result.",
            "prediction": "Tell me in your own words what you want to know about each case. "
            "For example: “whether the customer will cancel next month”.",
            "labels_available": f"Do your records already show {about} for past cases? For "
            "example, a column that says what actually happened.",
            "classes": "What are the possible answers? For example: “cracked” and “not cracked”.",
            "error_costs": "No system is right every time. What worries you more: missing a "
            "case that did matter, or getting warnings about cases that turn out fine?",
            "positive_class": "Which answer matters most to catch"
            + (f": {', '.join(f.classes)}?" if f.classes else "?"),
            "extrapolate": "Will you use it with cases similar to the ones you have, or with "
            "values bigger or smaller than those in your data?",
            "grouped": "Do you have several photos or recordings of the same object, person or "
            "session? I need it to test the model fairly, on objects it never saw.",
            "success_target": (
                "How often would it need to be right to be useful? For example, 8 out of 10 (80 %)."
                if f.problem == "category"
                else "How far off can it be and still be useful? For example, within 10 % of "
                "the real value."
            ),
        }
        return en.get(field, "Tell me a bit more about your case.")
    es = {
        "data_kind": "Para empezar: ¿cómo tenés guardada esa información? Por ejemplo, en una "
        "planilla (Excel o CSV), en fotos, en textos o en grabaciones.",
        "problem": "¿Qué te gustaría que te diga el sistema para cada caso? Puede elegir entre "
        "varias respuestas (como «se va» o «se queda»), estimar un número (como un precio) o "
        "descubrir la fórmula que hay detrás de un resultado.",
        "prediction": "Contame con tus palabras qué querés saber de cada caso. Por ejemplo: «si "
        "el cliente se va a dar de baja el mes que viene».",
        "labels_available": f"En tus registros, ¿ya figura {about} de casos pasados? Por "
        "ejemplo, una columna que diga qué pasó realmente.",
        "classes": "¿Cuáles son las respuestas posibles? Por ejemplo: «con fisura» y «sin fisura».",
        "error_costs": "Ningún sistema acierta siempre. ¿Qué te preocupa más: que se le escape "
        "un caso que sí importaba, o que te avise de más con casos que estaban bien?",
        "positive_class": "¿Cuál de las respuestas es la que más te importa detectar"
        + (f": {', '.join(f.classes)}?" if f.classes else "?"),
        "extrapolate": "¿Lo vas a usar con casos parecidos a los que ya tenés, o con valores más "
        "grandes o más chicos que los de tus datos?",
        "grouped": "¿Tenés varias fotos o grabaciones del mismo objeto, persona o sesión? Lo "
        "necesito para probar el modelo de forma justa, con objetos que nunca vio.",
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
            parts.append(f"The possible answers are {', '.join(f.classes)}.")
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
        parts.append(f"Las respuestas posibles son {', '.join(f.classes)}.")
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
        "awaiting_data": "Ahora necesito tus datos: seguí el ejemplo de abajo y subilos acá.",
        "reviewing_data": "Estoy revisando tus datos; en un momento te cuento qué encontré.",
        "ready_to_train": "Tus datos están listos. Cuando quieras, tocá «Preparar y entrenar».",
        "training": "Estoy entrenando el modelo; te aviso cuando termine.",
        "ready_to_validate": (
            f"Probalo con casos nuevos: completá un caso con su resultado real y lo comparo. "
            f"Con {VALIDATION_MIN} casos queda listo para usar."
        ),
        "needs_improvement": "Podés subir datos nuevos o volver a entrenar desde acá.",
        "ready_to_use": "Completá los datos de un caso abajo y te digo el resultado.",
        "failed": "Algo falló; podés reintentar o subir otros datos.",
        "interrupted": "El trabajo quedó detenido; podés retomarlo cuando quieras.",
    }
    en_help = {
        "awaiting_data": "Now I need your data: follow the example below and upload it here.",
        "reviewing_data": "I'm checking your data; I'll tell you what I found in a moment.",
        "ready_to_train": "Your data is ready. Whenever you want, click “Prepare and train”.",
        "training": "I'm training the model; I'll let you know when it's done.",
        "ready_to_validate": (
            f"Test it on new cases: enter a case with its real result and I'll compare. With "
            f"{VALIDATION_MIN} cases it's ready to use."
        ),
        "needs_improvement": "You can upload new data or retrain from here.",
        "ready_to_use": "Fill in a case below and I'll tell you the result.",
        "failed": "Something went wrong; you can retry or upload other data.",
        "interrupted": "The work was stopped; you can resume it whenever you want.",
    }
    prefix = (
        "I couldn't reach the AI to answer that. "
        if en
        else "No pude consultar a la IA para responderte eso. "
    )
    return prefix + (en_help if en else es_help).get(state, "")
