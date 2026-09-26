# ADR-0001: Registro de decisiones y monorepo
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: El SPEC (§2) fija decisiones vinculantes y pide registrar todo desvío como ADR (§18.3). El sistema tiene engine Python, servidor, UI React y desktop Tauri que deben evolucionar juntos.
- Decisión: Un único monorepo con la estructura de SPEC §12 (`engine/`, `server/`, `ui/`, `desktop/`, `fixtures/`, `docs/`). Las decisiones de §2 se registran como ADRs 0002–0012 en estado aceptado; toda decisión nueva o desvío se registra antes de implementarse. Conventional Commits.
- Consecuencias: Cambios transversales (API + tipos TS + UI) en un solo commit; CI único. Hay que mantener dos gestores de paquetes (uv y pnpm).
- Alternativas consideradas: Repos separados por componente (descartado: sincronizar versiones del contrato API sería costoso).
