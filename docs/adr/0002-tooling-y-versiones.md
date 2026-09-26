# ADR-0002: Tooling y versiones
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: SPEC §5 pide fijar versiones estables al iniciar la Capa 0 y revisar licencias (producto comercial).
- Decisión: Python 3.12 gestionado por **uv** (workspace `engine` + `server`, `requires-python >=3.12,<3.14`); Ruff, mypy (strict en core/domain/archspec/llm), pytest + hypothesis + pytest-cov. Frontend con **pnpm** 12 (workspaces `ui`, `desktop`), Node ≥ 22. Versiones resueltas el 2026-09-26 — Python: FastAPI 0.141, Pydantic 2.13, SQLAlchemy 2.1, Typer 0.27, Uvicorn 0.54, pytest 9.1, hypothesis 6.168, Ruff 0.16, mypy 2.3; frontend: React 19.3, Vite 8.3, Vitest 5.0, TanStack Query 5.103, i18next 26.4, openapi-typescript 7.13, openapi-fetch 0.17, ESLint 10.11, typescript-eslint 8.70. **TypeScript fijado en ~6.0** (6.0.3): TS 7.0 todavía no es compatible con typescript-eslint (<6.1) ni con openapi-typescript. Las versiones exactas quedan congeladas en `uv.lock` y `pnpm-lock.yaml`.
- Consecuencias: Revisar el pin de TypeScript cuando typescript-eslint soporte TS 7. Python 3.14 del sistema no se usa (compatibilidad del ecosistema PyTorch).
- Alternativas consideradas: Poetry / pip-tools (descartados: uv también gestiona el runtime embebido del instalador, SPEC §5.1).
