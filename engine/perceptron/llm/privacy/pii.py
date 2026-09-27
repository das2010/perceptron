"""Enmascarado de PII para el nivel L2 (SPEC §7.7.3; ADR-0022).

- `PatternPiiEngine`: reconocedores por patrón (email, URL, IP, IBAN, tarjeta con Luhn,
  teléfono, DNI/CUIT/CUIL) + reglas regex del usuario. Sin dependencias.
- `PresidioPiiEngine`: agrega NER (nombres, lugares, organizaciones) con Presidio y un
  motor NLP configurado. Los modelos spaCy en español son GPL-3.0 y no se usan: el
  modelo NER se configura explícitamente con uno de licencia comercial compatible.

Sin un motor con NER, los campos de **texto libre** no se envían en L2 (falla cerrada):
los patrones no detectan nombres de personas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class PiiEngine(Protocol):
    name: str
    handles_free_text: bool

    def mask(self, text: str) -> tuple[str, list[str]]:
        """Texto enmascarado y tipos de entidad encontrados."""
        ...


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("URL", re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b")),
    ("CUIT", re.compile(r"\b(?:20|23|24|27|30|33|34)-?\d{8}-?\d\b")),
    ("TARJETA", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("DNI", re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}\b|\b\d{7,8}\b")),
    ("TELEFONO", re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d\b")),
]


@dataclass
class PatternPiiEngine:
    rules: list[tuple[str, str]] = field(default_factory=list)
    name: str = "patterns"
    handles_free_text: bool = False

    def __post_init__(self) -> None:
        self._user = [(label, re.compile(rx)) for label, rx in self.rules]

    def mask(self, text: str) -> tuple[str, list[str]]:
        found: list[str] = []

        def repl(label: str) -> Any:
            def inner(m: re.Match[str]) -> str:
                raw = m.group(0)
                digits = re.sub(r"\D", "", raw)
                if label == "TARJETA" and not _luhn_ok(digits):
                    return raw
                if label == "TELEFONO" and (len(digits) < 8 or _DATE.match(raw.strip())):
                    return raw
                found.append(label)
                return f"<{label}>"

            return inner

        for label, rx in [*self._user, *_PATTERNS]:
            text = rx.sub(repl(label), text)
        return text, found


class PresidioPiiEngine:
    """Presidio con NER. Requiere el extra `privacy` y un motor NLP configurado."""

    name = "presidio"
    handles_free_text = True

    def __init__(self, nlp_configuration: dict[str, Any], rules: list[tuple[str, str]]) -> None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        nlp = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
        languages = [m["lang_code"] for m in nlp_configuration.get("models", [])] or ["es"]
        self._languages = languages
        self._analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=languages)
        self._anonymizer = AnonymizerEngine()
        self._patterns = PatternPiiEngine(rules)

    def mask(self, text: str) -> tuple[str, list[str]]:
        text, found = self._patterns.mask(text)
        results = self._analyzer.analyze(text=text, language=self._languages[0])
        if results:
            text = str(self._anonymizer.anonymize(text=text, analyzer_results=results).text)
            found += [str(r.entity_type) for r in results]
        return text, found


def pii_engine(
    rules: list[tuple[str, str]] | None = None, ner: dict[str, Any] | None = None
) -> PiiEngine:
    if ner:
        return PresidioPiiEngine(ner, rules or [])
    return PatternPiiEngine(rules or [])
