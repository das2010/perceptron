"""LLM Gateway: única puerta hacia cualquier proveedor (RF-LLM-01..07, RF-PRV-01..03).

Orden de cada llamada estructurada:
1. política: LLM activo, nivel efectivo > L0, proveedor permitido, credenciales;
2. PrivacyFilter sobre el `LLMContext` → payload;
3. render del prompt versionado;
4. caché (hash de proveedor + request);
5. presupuesto (estimación previa, corte antes de pasarse);
6. llamada al proveedor;
7. parseo + validación Pydantic + validador de dominio; reintento con el error (RF-LLM-04);
8. auditoría: cada intento queda en un `LLMCall` con el payload ya filtrado.

Quien llama captura `LLMUnavailableError`, `LLMBudgetExceededError`, `LLMProviderError` y
`LLMOutputInvalidError` y cae a reglas (RF-WIZ-03, RF-ARC-04).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from perceptron.core.config import RuntimeMode, Settings
from perceptron.core.logging import log_context
from perceptron.domain.enums import LLMPurpose, PrivacyLevel
from perceptron.domain.models import LLMCall, Project, Workspace
from perceptron.llm.budget import OUTPUT_FRACTION, BudgetLedger, estimate_tokens
from perceptron.llm.cache import LLMCache, cache_key
from perceptron.llm.config import LLMConfig, ProviderInfo, Resolved
from perceptron.llm.errors import (
    LLMOutputInvalidError,
    LLMProviderError,
    LLMUnavailableError,
)
from perceptron.llm.jsonparse import extract_json
from perceptron.llm.privacy import FilteredPayload, LLMContext, PrivacyFilter, PrivacyPolicy
from perceptron.llm.prompts.registry import PromptRegistry, PromptTemplate
from perceptron.llm.providers import NEEDS_KEY, FakeLLMProvider, LLMProvider, build_provider
from perceptron.llm.secrets import SecretStore, default_secret_store
from perceptron.llm.types import LLMRequest, LLMResponse, Message
from perceptron.storage.db import Database
from perceptron.storage.repositories import SqlRepository

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Validator = Callable[[T], str | None]
ProviderFactory = Callable[[str, ProviderInfo, str | None], LLMProvider]

MAX_AUDIT_TEXT = 20_000
JSON_INSTRUCTIONS = (
    "Respondé únicamente con un objeto JSON válido (sin texto alrededor) que cumpla este "
    "JSON Schema:\n{schema}"
)
RETRY_MESSAGE = (
    "La respuesta anterior no es válida:\n{error}\n\n"
    "Corregí esos problemas y devolvé la respuesta completa otra vez."
)


@dataclass
class LLMResult[V: BaseModel]:
    value: V
    call_id: str
    provider: str
    model: str
    cost_usd: float
    cache_hit: bool
    attempts: int
    level: PrivacyLevel
    redactions: list[str] = field(default_factory=list)


def _level_min(a: PrivacyLevel, b: PrivacyLevel) -> PrivacyLevel:
    return a if a.rank <= b.rank else b


class Gateway:
    def __init__(
        self,
        db: Database,
        config: LLMConfig,
        secrets: SecretStore,
        *,
        provider_factory: ProviderFactory | None = None,
        prompts: PromptRegistry | None = None,
        policy: PrivacyPolicy | None = None,
    ) -> None:
        self.db = db
        self.config = config
        self.settings = config.settings
        self.secrets = secrets
        self.prompts = prompts or PromptRegistry()
        self.policy = policy or PrivacyPolicy(l2_samples=self.settings.l2_samples)
        self.calls: SqlRepository[LLMCall] = SqlRepository(db, LLMCall)
        self.workspaces: SqlRepository[Workspace] = SqlRepository(db, Workspace)
        self.cache = LLMCache(db)
        self.ledger = BudgetLedger(self.calls)
        self._factory = provider_factory or self._default_factory
        self._custom_factory = provider_factory is not None
        self._providers: dict[str, LLMProvider] = {}

    @classmethod
    def from_settings(cls, settings: Settings, db: Database) -> Gateway:
        config = LLMConfig(settings.llm, settings.workspace_dir)
        secrets = default_secret_store(
            server=settings.mode is RuntimeMode.SERVER,
            secrets_file=settings.workspace_dir / "secrets.enc.json",
        )
        factory: ProviderFactory | None = None
        if settings.llm.fake_cassette is not None:
            fake = FakeLLMProvider.from_cassette(settings.llm.fake_cassette)

            def fake_factory(_name: str, _info: ProviderInfo, _key: str | None) -> LLMProvider:
                return fake

            factory = fake_factory

        return cls(db, config, secrets, provider_factory=factory)

    @staticmethod
    def _default_factory(_name: str, info: ProviderInfo, api_key: str | None) -> LLMProvider:
        return build_provider(info, api_key)

    @property
    def fake_mode(self) -> bool:
        return self.settings.fake_cassette is not None or self._custom_factory

    # ------------------------------------------------------------------ política

    def workspace(self) -> Workspace:
        found = list(self.workspaces.list(limit=1))
        return found[0] if found else Workspace(name="local")

    def effective_level(self, project: Project, *, local: bool) -> PrivacyLevel:
        """Nivel del proyecto acotado por la política del workspace (RF-PRV-02, RF-PRV-04)."""
        ws = self.workspace()
        cap = ws.max_privacy_level
        if local and ws.local_llm_max_privacy is not None:
            cap = ws.local_llm_max_privacy if ws.local_llm_max_privacy.rank > cap.rank else cap
        return _level_min(project.privacy_level, cap)

    def resolve(self, purpose: LLMPurpose, project: Project) -> Resolved:
        if not self.settings.enabled:
            raise LLMUnavailableError(
                "La capa LLM está desactivada", details={"reason": "disabled"}
            )
        if project.privacy_level is PrivacyLevel.L0:
            raise LLMUnavailableError("Privacidad L0: sin LLM", details={"reason": "privacy_l0"})
        res = self.config.resolve(purpose, project.llm_profile_id)
        if res is None:
            raise LLMUnavailableError(
                f"El perfil no cubre el propósito {purpose.value}",
                details={"reason": "no_profile", "profile": project.llm_profile_id},
            )
        allowed = self.workspace().allowed_llm_providers
        if allowed is not None and res.provider_name not in allowed:
            raise LLMUnavailableError(
                f"El Admin no permite el proveedor {res.provider_name}",
                details={"reason": "provider_not_allowed", "provider": res.provider_name},
            )
        if self.effective_level(project, local=res.provider.is_local) is PrivacyLevel.L0:
            raise LLMUnavailableError("Política L0: sin LLM", details={"reason": "privacy_l0"})
        if (
            not self.fake_mode
            and res.provider.kind in NEEDS_KEY
            and not self._api_key(res.provider)
        ):
            raise LLMUnavailableError(
                f"Falta la clave de {res.provider_name}",
                details={"reason": "missing_key", "secret": res.provider.api_key_ref},
            )
        return res

    def available(self, purpose: LLMPurpose, project: Project) -> bool:
        try:
            self.resolve(purpose, project)
        except LLMUnavailableError:
            return False
        return True

    def _api_key(self, info: ProviderInfo) -> str | None:
        return self.secrets.get(info.api_key_ref) if info.api_key_ref else None

    def provider(self, res: Resolved) -> LLMProvider:
        if res.provider_name not in self._providers:
            self._providers[res.provider_name] = self._factory(
                res.provider_name, res.provider, self._api_key(res.provider)
            )
        return self._providers[res.provider_name]

    # ------------------------------------------------------------------ llamadas

    def _prepare(
        self,
        purpose: LLMPurpose,
        ctx: LLMContext,
        project: Project,
        prompt_name: str,
        prompt_vars: dict[str, Any] | None,
    ) -> tuple[Resolved, PromptTemplate, FilteredPayload, str, str]:
        res = self.resolve(purpose, project)
        level = self.effective_level(project, local=res.provider.is_local)
        filtered = PrivacyFilter(level, self.policy).apply(ctx, vision=res.model.vision)
        prompt = self.prompts.get(purpose, prompt_name, res.ref.prompt_version)
        system, user = prompt.render(filtered.data, **(prompt_vars or {}))
        return res, prompt, filtered, system, user

    def structured(
        self,
        purpose: LLMPurpose,
        output_model: type[T],
        ctx: LLMContext,
        *,
        project: Project,
        validator: Validator[T] | None = None,
        prompt: str = "main",
        prompt_vars: dict[str, Any] | None = None,
        scope: str | None = None,
        scope_budget_usd: float | None = None,
    ) -> LLMResult[T]:
        """Salida estructurada validada; reintenta con el error hasta `max_attempts`."""
        res, template, filtered, system, user = self._prepare(
            purpose, ctx, project, prompt, prompt_vars
        )
        schema = output_model.model_json_schema()
        if not res.model.structured_output:
            system += "\n\n" + JSON_INSTRUCTIONS.format(
                schema=json.dumps(schema, ensure_ascii=False)
            )
        messages = [Message(role="user", content=user, images=filtered.images)]
        provider = self.provider(res)
        last_value: T | None = None
        last_call: str | None = None
        error = ""
        total_cost = 0.0
        for attempt in range(1, self.settings.max_attempts + 1):
            request = LLMRequest(
                model=res.model_id,
                system=system,
                messages=messages,
                output_schema=schema,
                schema_name=purpose.value,
                temperature=res.ref.temperature,
                max_tokens=res.ref.max_tokens,
                purpose=purpose.value,
            )
            key = cache_key(res.provider_name, request)
            cached = self.cache.get(key) if self.settings.cache else None
            started = time.perf_counter()
            if cached is not None:
                response, cost = LLMResponse.model_validate(cached), 0.0
            else:
                estimate = res.model.cost(
                    estimate_tokens(system + user),
                    int(request.max_tokens * OUTPUT_FRACTION),
                )
                self.ledger.check(
                    project.id,
                    estimate,
                    project_limit=self.settings.project_budget_usd,
                    scope=scope,
                    scope_limit=scope_budget_usd,
                )
                try:
                    response = provider.complete(request, res.model)
                except LLMProviderError as e:
                    self._audit(
                        purpose,
                        project,
                        res,
                        template,
                        filtered,
                        request,
                        None,
                        attempt,
                        status="error",
                        error=e.message,
                        scope=scope,
                        latency=time.perf_counter() - started,
                    )
                    raise
                cost = res.model.cost(response.usage.input_tokens, response.usage.output_tokens)
            total_cost += cost
            value, error, raw = self._parse(response, output_model, filtered, validator)
            call = self._audit(
                purpose,
                project,
                res,
                template,
                filtered,
                request,
                response,
                attempt,
                status="ok" if value is not None and not error else "invalid",
                error=error or None,
                scope=scope,
                cost=cost,
                cache_hit=cached is not None,
                latency=time.perf_counter() - started,
            )
            last_call = call.id
            if value is not None and not error:
                if cached is None and self.settings.cache:
                    self.cache.put(key, response.model_dump(mode="json"))
                return LLMResult(
                    value=value,
                    call_id=call.id,
                    provider=res.provider_name,
                    model=res.model_id,
                    cost_usd=total_cost,
                    cache_hit=cached is not None,
                    attempts=attempt,
                    level=filtered.level,
                    redactions=filtered.redactions,
                )
            if value is not None:
                last_value = value
            messages = [
                *messages,
                Message(role="assistant", content=raw or "(vacío)"),
                Message(role="user", content=RETRY_MESSAGE.format(error=error)),
            ]
            logger.info(
                "salida LLM inválida, reintento",
                extra={"purpose": purpose.value, "attempt": attempt, "error": error[:500]},
            )
        raise LLMOutputInvalidError(
            f"La salida del LLM no validó tras {self.settings.max_attempts} intentos",
            last_value=last_value,
            details={"purpose": purpose.value, "error": error[:2000], "call_id": last_call},
        )

    @staticmethod
    def _parse(
        response: LLMResponse,
        output_model: type[T],
        filtered: FilteredPayload,
        validator: Validator[T] | None,
    ) -> tuple[T | None, str, str]:
        raw = json.dumps(response.data, ensure_ascii=False) if response.data else response.text
        try:
            data = response.data if response.data is not None else extract_json(response.text)
        except ValueError as e:
            return None, str(e), raw
        try:
            value = output_model.model_validate(filtered.restore(data))
        except PydanticValidationError as e:
            msgs = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
            return None, "Errores de schema:\n" + "\n".join(msgs[:30]), raw
        error = validator(value) if validator else None
        return value, error or "", raw

    def _audit(
        self,
        purpose: LLMPurpose,
        project: Project,
        res: Resolved,
        template: PromptTemplate,
        filtered: FilteredPayload,
        request: LLMRequest,
        response: LLMResponse | None,
        attempt: int,
        *,
        status: Literal["ok", "invalid", "error"],
        error: str | None = None,
        scope: str | None = None,
        cost: float = 0.0,
        cache_hit: bool = False,
        latency: float = 0.0,
    ) -> LLMCall:
        """RF-PRV-03: exactamente lo que salió (post-filtro), a qué proveedor y modelo."""
        payload = {
            "system": request.system,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "images": sum(len(m.images) for m in request.messages),
            "data": filtered.data,
        }
        body: dict[str, Any] | None = None
        if response is not None:
            body = {
                "data": response.data,
                "text": response.text[:MAX_AUDIT_TEXT],
                "stop_reason": response.stop_reason,
            }
        call = LLMCall(
            project_id=project.id,
            purpose=purpose,
            provider=res.provider_name,
            model=res.model_id,
            prompt_version=template.ref,
            privacy_level=filtered.level,
            payload=payload,
            response=body,
            input_tokens=response.usage.input_tokens if response and not cache_hit else 0,
            output_tokens=response.usage.output_tokens if response and not cache_hit else 0,
            cost_usd=cost,
            status=status,
            error=error,
            attempt=attempt,
            cache_hit=cache_hit,
            redactions=filtered.redactions,
            scope=scope,
            latency_s=round(latency, 3),
        )
        with log_context(llm_call_id=call.id):
            logger.info(
                "llm call",
                extra={
                    "purpose": purpose.value,
                    "provider": res.provider_name,
                    "model": res.model_id,
                    "status": status,
                    "attempt": attempt,
                    "cache_hit": cache_hit,
                    "cost_usd": round(cost, 6),
                },
            )
        return self.calls.add(call)

    def stream_chat(
        self,
        purpose: LLMPurpose,
        ctx: LLMContext,
        *,
        project: Project,
        history: list[Message] | None = None,
        prompt: str = "main",
        prompt_vars: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Respuesta en texto por fragmentos para el copiloto (RF-LLM-05). Se audita al final."""
        res, template, filtered, system, user = self._prepare(
            purpose, ctx, project, prompt, prompt_vars
        )
        request = LLMRequest(
            model=res.model_id,
            system=system,
            messages=[*(history or []), Message(role="user", content=user, images=filtered.images)],
            temperature=res.ref.temperature,
            max_tokens=res.ref.max_tokens,
            purpose=purpose.value,
        )
        self.ledger.check(
            project.id,
            res.model.cost(
                estimate_tokens(system + user), int(request.max_tokens * OUTPUT_FRACTION)
            ),
            project_limit=self.settings.project_budget_usd,
        )
        started = time.perf_counter()
        pieces: list[str] = []
        for piece in self.provider(res).stream(request, res.model):
            pieces.append(piece)
            yield piece
        text = "".join(pieces)
        usage_in, usage_out = estimate_tokens(system + user), estimate_tokens(text)
        response = LLMResponse(text=text, model=res.model_id)
        response.usage.input_tokens, response.usage.output_tokens = usage_in, usage_out
        self._audit(
            purpose,
            project,
            res,
            template,
            filtered,
            request,
            response,
            1,
            status="ok",
            cost=res.model.cost(usage_in, usage_out),
            latency=time.perf_counter() - started,
        )
