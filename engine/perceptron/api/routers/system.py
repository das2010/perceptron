"""`/system/*` (SPEC §10). Hardware y variantes de torch llegan en Capa 1."""

from __future__ import annotations

import platform
import sys
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from perceptron import __version__
from perceptron.api.context import EngineContext, get_context
from perceptron.core.config import RuntimeMode
from perceptron.training.hardware import HardwareReport, detect_hardware

router = APIRouter(prefix="/system", tags=["system"])


class Health(BaseModel):
    status: Literal["ok"]
    version: str


class VersionInfo(BaseModel):
    version: str
    python: str
    platform: str
    mode: RuntimeMode


@router.get("/health", operation_id="getHealth")
def health() -> Health:
    return Health(status="ok", version=__version__)


@router.get("/hardware", operation_id="getHardware")
def hardware(ctx: Annotated[EngineContext, Depends(get_context)]) -> HardwareReport:
    """Hardware disponible y dispositivo recomendado (RF-TRN-01)."""
    return detect_hardware(ctx.settings.workspace_dir)


@router.get("/version", operation_id="getVersion")
def version(ctx: Annotated[EngineContext, Depends(get_context)]) -> VersionInfo:
    return VersionInfo(
        version=__version__,
        python=sys.version.split()[0],
        platform=platform.platform(),
        mode=ctx.settings.mode,
    )
