"""Identificadores: ULID ordenables en el tiempo con prefijo por tipo de entidad.

Ej.: `prj_01J8Z3K4V6Q9X2B7M5N1C0D8EF`. El prefijo hace los ids legibles en logs.
"""

from __future__ import annotations

from enum import StrEnum

from ulid import ULID


class IdPrefix(StrEnum):
    WORKSPACE = "wsp"
    USER = "usr"
    MEMBERSHIP = "mbr"
    PROJECT = "prj"
    DATASOURCE = "src"
    DATASET_VERSION = "dsv"
    PROFILE = "prf"
    LABEL_SET = "lbl"
    PIPELINE = "pip"
    ARCHSPEC = "arc"
    STUDY = "std"
    RUN = "run"
    EVALUATION = "evl"
    MODEL_VERSION = "mdl"
    EXPORT = "exp"
    DEPLOYMENT = "dep"
    DRIFT_REPORT = "drf"
    RETRAIN_POLICY = "rtp"
    LLM_SESSION = "lls"
    LLM_CALL = "llc"
    AGENT_RUN = "agr"
    PROJECT_DRAFT = "dft"
    JOB = "job"
    SESSION = "ses"
    REFRESH_TOKEN = "rtk"  # noqa: S105 - prefijo de ID, no un secreto
    ALERT = "alr"
    PREDICTION = "prd"


_SEP = "_"


def new_id(prefix: IdPrefix) -> str:
    return f"{prefix.value}{_SEP}{ULID()}"


def parse_id(value: str) -> tuple[IdPrefix, ULID]:
    """Valida y descompone un id. Lanza `ValueError` si es inválido."""
    prefix, sep, raw = value.partition(_SEP)
    if not sep:
        raise ValueError(f"id sin prefijo: {value!r}")
    return IdPrefix(prefix), ULID.from_str(raw)


def is_valid_id(value: str, prefix: IdPrefix | None = None) -> bool:
    try:
        p, _ = parse_id(value)
    except ValueError:
        return False
    return prefix is None or p is prefix
