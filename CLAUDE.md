# Perceptron — instrucciones para Claude Code

- La fuente de verdad es SPEC.md. Antes de implementar algo, leé la sección correspondiente y citá los IDs de RF.
- Las decisiones de SPEC.md §2 son vinculantes; si hace falta cambiarlas, proponé un ADR en docs/adr/ y esperá aprobación.
- Construcción por capas (SPEC.md §14). No adelantes capas; dejá interfaces y feature flags para lo futuro.
- Python: uv, Python 3.12+, Ruff, mypy strict en core/archspec/llm/domain, pytest. Nada de lógica de ML en la UI.
- Frontend: React + TS strict, TanStack Query, shadcn/ui con tokens de ui/src/styles/tokens.css (marca Preteco, Titillium Web). Todos los strings por i18n (es/en).
- Tipos del API: siempre regenerar desde OpenAPI (`pnpm gen:api`) tras cambiar endpoints.
- LLM: nunca llamar a un proveedor directo; siempre vía llm.gateway con propósito, schema de salida y PrivacyFilter. En tests usar FakeLLMProvider.
- No hardcodear IDs de modelos LLM; van en configuración.
- Toda salida del LLM que modifica el sistema debe validarse contra schema y ser aceptable/rechazable por el usuario.
- Windows y Linux: usar pathlib, probar rutas con espacios y acentos; nada de comandos solo-bash en runtime.
- Cada RF implementada lleva tests; actualizá docs/progress.md.
- Comandos: `uv run pytest`, `uv run ruff check .`, `uv run mypy engine`, `pnpm -C ui test`, `pnpm -C ui e2e`, `pnpm -C desktop tauri dev`.
