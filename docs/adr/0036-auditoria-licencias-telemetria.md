# ADR-0036: Auditoría de licencias, telemetría opt-in y actualización N → N+1 (Capa 7)
- Estado: propuesto
- Fecha: 2026-09-28
- Contexto: la aceptación de la Capa 7 pide un informe de licencias sin dependencias
  incompatibles con la distribución comercial y actualizar de N a N+1 sin perder proyectos;
  D7 (telemetría) queda para definir con Preteco.
- Decisión:
  - **Auditoría en CI** (`scripts/license_audit.py`, job `licenses`): runtime Python (`uv export
    --no-dev`), JS de producción (`pnpm licenses`), crates del desktop (`cargo metadata`) y
    pesos del catálogo (licencia + `commercial_ok`). Política en `docs/licenses/policy.yaml`:
    permitidas (MIT, BSD, Apache-2.0, ISC, PSF…), a revisar (MPL, LGPL, EPL), prohibidas
    (GPL, AGPL, SSPL, BUSL, NC) y excepciones justificadas. Falla si hay algo prohibido; el
    informe (Markdown + JSON) queda como artefacto.
  - **Telemetría opt-in**: apagada por defecto y **sin endpoint de fábrica** (no se envía nada
    hasta que Preteco defina el destino, D7). Contenido fijo y visible antes de aceptar:
    versión, SO, CPU/RAM agregados, contadores de uso por operación y tipos de error; nunca
    datos, columnas, rutas, textos ni métricas. Identificador de instalación aleatorio.
  - **Actualización N → N+1** (job `upgrade`): se crea un workspace con `main` (quickstart
    UC-01) y se abre con la versión nueva; `scripts/upgrade_check.py` exige que todas las
    entidades se lean con los modelos actuales, que cada versión de datos se pueda leer y
    que runs y modelos conserven sus artefactos. El Team Server ya migra con Alembic.
  - **Seguridad en CI** (job `security`): `pip-audit` del runtime y `pnpm audit` de
    producción (altas y críticas); las reglas S de Ruff (bandit) ya corren en cada PR.
- Consecuencias: pendiente de Preteco la firma de código Authenticode, las claves del
  updater de Tauri y el destino de la telemetría.
