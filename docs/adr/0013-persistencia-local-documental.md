# ADR-0013: Persistencia de metadata: SQLAlchemy 2 + tabla documental, ids ULID
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: Decisión propia de Capa 0 (no contradice §2). El modelo de dominio va a cambiar mucho en las capas 1–6 y el servidor usará PostgreSQL.
- Decisión: SQLAlchemy 2 también en local (SQLite con WAL), para compartir la capa con PostgreSQL. En Capa 0 una tabla `entities(kind, id, project_id, version, created_at, updated_at, data JSON)` con repositorio genérico, bloqueo optimista por `version` y filtros por columnas indexadas o campos JSON. Ids ULID con prefijo por entidad (`prj_…`, `run_…`). Configuración con pydantic-settings (`PERCEPTRON_*`).
- Consecuencias: Evolución del esquema sin migraciones en Capa 0–1; cuando una entidad necesite consultas relacionales intensivas (runs, auditoría LLM) se promueve a tabla propia con Alembic (obligatorio en Capa 5).
- Alternativas consideradas: Tablas normalizadas por entidad desde el inicio (descartado por ahora: churn de migraciones); TinyDB/JSON en disco (sin transacciones).
