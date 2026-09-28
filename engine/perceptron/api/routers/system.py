"""`/system/*` (SPEC §10). Hardware y variantes de torch llegan en Capa 1."""

from __future__ import annotations

import platform
import sys
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from perceptron import __version__
from perceptron.api.context import EngineContext, get_context
from perceptron.api.jobs import Job
from perceptron.catalog.weights_cache import (
    CacheReport,
    VerifyReport,
    delete_cached,
    list_cached,
    prefetch,
    verify,
)
from perceptron.core.config import RuntimeMode
from perceptron.core.errors import ValidationError
from perceptron.training.hardware import HardwareReport, detect_hardware

router = APIRouter(prefix="/system", tags=["system"])
Ctx = Annotated[EngineContext, Depends(get_context)]


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


# ------------------------------------------------------------------ licencia (RF-LIC, ADR-0035)


class LicenseView(BaseModel):
    status: str
    message: str = ""
    enforce: bool
    licensee: str | None = None
    edition: str | None = None
    seats: int | None = None
    servers: int | None = None
    gpus: int | None = None
    features: list[str] = []
    expires_at: str | None = None


class LicenseUpload(BaseModel):
    content: str


def _view(ctx: EngineContext) -> LicenseView:
    from perceptron.licensing.signed import license_provider

    provider = license_provider(ctx.settings)
    check = provider.check()
    t = check.terms
    return LicenseView(
        status=check.status.value,
        message=check.message,
        enforce=provider.enforce,
        licensee=t.licensee if t else None,
        edition=t.edition if t else None,
        seats=t.seats if t else None,
        servers=t.servers if t else None,
        gpus=t.gpus if t else None,
        features=t.features if t else [],
        expires_at=t.expires_at.isoformat() if t and t.expires_at else None,
    )


@router.get("/license", operation_id="getLicense")
def get_license(ctx: Annotated[EngineContext, Depends(get_context)]) -> LicenseView:
    return _view(ctx)


@router.put("/license", operation_id="putLicense")
def put_license(
    body: LicenseUpload, ctx: Annotated[EngineContext, Depends(get_context)]
) -> LicenseView:
    """Instala una licencia: solo si la firma y la vigencia son válidas."""
    from perceptron.core.errors import ValidationError
    from perceptron.licensing.signed import license_provider, verify

    provider = license_provider(ctx.settings)
    check = verify(body.content, provider.trusted)
    if not check.valid:
        raise ValidationError(
            f"licencia rechazada: {check.message}", details={"status": check.status.value}
        )
    provider.path.parent.mkdir(parents=True, exist_ok=True)
    provider.path.write_text(body.content, encoding="utf-8")
    return _view(ctx)


# ------------------------------------------------------------------ telemetría opt-in (D7)


class TelemetryStatus(BaseModel):
    asked: bool
    opt_in: bool
    endpoint_configured: bool
    preview: dict[str, Any]


class TelemetryConsent(BaseModel):
    opt_in: bool


@router.get("/telemetry", operation_id="getTelemetry")
def telemetry_status(ctx: Annotated[EngineContext, Depends(get_context)]) -> TelemetryStatus:
    """Estado del consentimiento y exactamente lo que se enviaría."""
    return TelemetryStatus.model_validate(ctx.telemetry.status())


@router.put("/telemetry", operation_id="putTelemetry")
def telemetry_consent(
    body: TelemetryConsent, ctx: Annotated[EngineContext, Depends(get_context)]
) -> TelemetryStatus:
    ctx.telemetry.consent(body.opt_in)
    return TelemetryStatus.model_validate(ctx.telemetry.status())


# ------------------------------------------------------------------ modelos preentrenados


class PrefetchBody(BaseModel):
    model: str


def _cache_dir(ctx: EngineContext) -> Any:
    return ctx.settings.models_cache or ctx.settings.paths.models_cache_dir


def _report(ctx: EngineContext) -> CacheReport:
    from perceptron.catalog.registry import HF_TEXT_MODELS, TIMM_WEIGHTS
    from perceptron.data.labeling.zero_shot import ZERO_SHOT_MODELS

    report = list_cached(_cache_dir(ctx))
    cached = {m.id for m in report.models}
    curated = sorted({*HF_TEXT_MODELS, *TIMM_WEIGHTS, *ZERO_SHOT_MODELS})
    return report.model_copy(update={"available": [m for m in curated if m not in cached]})


@router.get("/models-cache", operation_id="getModelsCache")
def models_cache(ctx: Ctx) -> CacheReport:
    """Pesos preentrenados descargados: tamaño y último uso (RF-TRN-11)."""
    return _report(ctx)


@router.delete("/models-cache/{model_id:path}", operation_id="deleteCachedModel")
def delete_cached_model(model_id: str, ctx: Ctx) -> CacheReport:
    delete_cached(model_id)
    return _report(ctx)


@router.post("/models-cache/verify", operation_id="verifyModelsCache")
def verify_models_cache() -> VerifyReport:
    """Recalcula los checksums de los archivos descargados."""
    return verify()


@router.post("/models-cache/prefetch", status_code=202, operation_id="prefetchModel")
def prefetch_model(body: PrefetchBody, ctx: Ctx) -> Job:
    """Descarga un modelo curado ahora, para entrenar después sin conexión."""
    from perceptron.catalog.registry import HF_TEXT_MODELS, TIMM_WEIGHTS
    from perceptron.data.labeling.zero_shot import ZERO_SHOT_MODELS

    if body.model not in {*HF_TEXT_MODELS, *TIMM_WEIGHTS, *ZERO_SHOT_MODELS}:
        raise ValidationError(f"{body.model} no está en el catálogo de pesos curados")
    return ctx.jobs.submit("prefetch", lambda _: {"model": prefetch(body.model)})
