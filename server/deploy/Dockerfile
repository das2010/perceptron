# syntax=docker/dockerfile:1
# Team Server (Capa 5a): Engine + auth/RBAC + UI web en una imagen. Variante CPU; la CUDA
# para los workers GPU llega con la cola de jobs (Capa 5b).

FROM node:22-bookworm-slim AS ui
WORKDIR /src
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY ui/package.json ui/
COPY desktop/package.json desktop/
RUN pnpm install --frozen-lockfile --filter ./ui...
COPY ui ui
RUN pnpm -C ui build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY engine engine
COPY server server
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --package perceptron-server
COPY --from=ui /src/ui/dist /app/ui

RUN useradd --system --uid 10001 --home-dir /data perceptron \
    && mkdir -p /data /sources && chown perceptron /data
USER perceptron
ENV PATH=/app/.venv/bin:$PATH \
    PERCEPTRON_MODE=server \
    PERCEPTRON_WORKSPACE_DIR=/data \
    PERCEPTRON_SERVER__SPA_DIR=/app/ui \
    PERCEPTRON_LOGGING__TO_FILE=false
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/v1/system/health', timeout=4)"]
CMD ["perceptron-server", "serve", "--host", "0.0.0.0", "--port", "8080"]
