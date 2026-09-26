# ADR-0009: Topología híbrida: un solo Engine, una sola UI
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: SPEC §4.2–4.3: standalone, conectado y estación en servidor.
- Decisión: El paquete `perceptron-engine` es idéntico en desktop y servidor (`RuntimeMode`). La UI solo habla HTTP/WS con un Engine y aísla lo nativo en `PlatformBridge` (implementación web en Capa 0, Tauri en Capa 3). Headless primero: todo por API y CLI.
- Consecuencias: El servidor (Capa 5) agrega auth, RBAC, sync y cola sin bifurcar el Engine.
- Alternativas consideradas: Engines distintos para desktop y servidor (descartado: duplicación).
