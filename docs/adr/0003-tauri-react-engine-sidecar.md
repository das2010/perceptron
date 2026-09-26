# ADR-0003: Desktop con Tauri 2 + React/TS y Engine FastAPI como sidecar
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: SPEC §2: app de escritorio para Windows y Linux, la misma UI servida como web por el Team Server.
- Decisión: Shell **Tauri 2** (Rust); UI **React + TypeScript**; el Engine Python (FastAPI) se lanza como sidecar en `127.0.0.1`, puerto aleatorio y token efímero. Contrato de arranque: `perceptron serve --new-token` escribe por stdout `{"event":"ready","host","port","token"}`; la UI envía `X-Perceptron-Token`.
- Consecuencias: Instalador liviano; el runtime Python embebido y la variante de PyTorch se gestionan aparte (Capa 3).
- Alternativas consideradas: Electron (más pesado); Python embebido vía PyO3 (acopla el ciclo de vida del Engine al proceso de UI).
