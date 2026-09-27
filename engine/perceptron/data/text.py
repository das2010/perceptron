"""Utilidades de texto (RF-PRF-03, RF-PIP-03): normalización, tokenización,
vocabulario ajustado solo con train y heurística de idioma.

El tokenizador propio es a nivel palabra con índice reservado para padding (0)
y desconocidas (1). Para encoders preentrenados se usa el tokenizador de Hugging
Face del modelo (ver `data.pipeline.pipeline`).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

PAD, UNK = 0, 1
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"

_URL = re.compile(r"https?://\S+|www\.\S+")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_STOPWORDS = {
    "es": frozenset(
        "de la que el en y a los se del las un por con no una su para es al lo como más mi me "
        "pero sus".split()
    ),
    "en": frozenset(
        "the of and to a in is it you that was for on are with as i be at this have my not "
        "but can".split()
    ),
}


def strip_accents(text: str) -> str:
    """Quita tildes y diéresis pero conserva la ñ."""
    text = text.replace("ñ", "\0").replace("Ñ", "\1")
    base = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return base.replace("\0", "ñ").replace("\1", "Ñ")


def normalize(
    text: str,
    *,
    lowercase: bool = True,
    accents: bool = True,
    urls: bool = True,
    numbers: bool = False,
) -> str:
    """`accents=False` elimina tildes; `urls`/`numbers` los reemplazan por marcadores."""
    t = unicodedata.normalize("NFC", text or "")
    if urls:
        t = _URL.sub(" <url> ", t)
    if numbers:
        t = _NUM.sub(" <num> ", t)
    if lowercase:
        t = t.lower()
    if not accents:
        t = strip_accents(t)
    return " ".join(t.split())


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for part in text.split():
        if part in ("<url>", "<num>"):
            tokens.append(part)
        else:
            tokens.extend(_TOKEN.findall(part))
    return tokens


def build_vocab(texts: Iterable[str], *, max_size: int = 20_000, min_freq: int = 2) -> list[str]:
    """Vocabulario ordenado por frecuencia (desempate alfabético → determinístico)."""
    counts = Counter(tok for t in texts for tok in tokenize(t))
    items = sorted(
        ((w, c) for w, c in counts.items() if c >= min_freq), key=lambda x: (-x[1], x[0])
    )
    return [PAD_TOKEN, UNK_TOKEN, *[w for w, _ in items[: max(max_size - 2, 0)]]]


def encode(text: str, index: dict[str, int], max_length: int) -> list[int]:
    ids = [index.get(tok, UNK) for tok in tokenize(text)][:max_length]
    return ids + [PAD] * (max_length - len(ids))


def guess_language(texts: Iterable[str], sample: int = 500) -> tuple[str | None, float]:
    """Idioma más probable ('es'/'en') por proporción de stopwords, y su confianza (0–1)."""
    hits = dict.fromkeys(_STOPWORDS, 0)
    total = 0
    for i, t in enumerate(texts):
        if i >= sample:
            break
        for tok in tokenize(t.lower()):
            total += 1
            for lang, words in _STOPWORDS.items():
                if tok in words:
                    hits[lang] += 1
    if not total or not any(hits.values()):
        return None, 0.0
    lang = max(hits, key=lambda k: hits[k])
    return lang, round(hits[lang] / max(sum(hits.values()), 1), 3)
