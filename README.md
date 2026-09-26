# Perceptron

Plataforma de escritorio + servidor de equipo (Preteco) para **entrenar redes neuronales localmente**, guiada por un LLM copiloto: desde los datos crudos hasta un modelo evaluado, explicado, exportado y monitoreado.

La especificación completa está en [SPEC.md](SPEC.md). El avance por capas se registra en [docs/progress.md](docs/progress.md) y las decisiones en [docs/adr/](docs/adr/).

## Requisitos de desarrollo

- [uv](https://docs.astral.sh/uv/) (gestiona Python 3.12)
- Node.js ≥ 22 y pnpm
- Rust estable (solo para el desktop Tauri, Capa 3)

## Primeros pasos

```bash
uv sync --all-packages          # entorno Python del workspace (engine + server)
uv run perceptron --version
uv run pytest
pnpm install
pnpm -C ui test
pnpm gen:api                    # regenera docs/api/openapi.json y los tipos TS
```

## Estructura

| Carpeta | Contenido |
|---------|-----------|
| `engine/` | Paquete Python `perceptron-engine` (API FastAPI, CLI, dominio, datos, ML) |
| `server/` | Extensiones del Team Server (Capa 5) |
| `ui/` | Frontend React + TypeScript (desktop y web) |
| `desktop/` | Shell Tauri 2 (Capa 3) |
| `fixtures/` | Datasets sintéticos diminutos, uno por caso de uso (UC-01…UC-10) |
| `docs/` | ADRs, OpenAPI exportada, manual de usuario |
