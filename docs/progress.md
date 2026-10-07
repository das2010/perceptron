# Progreso de implementación

Estado de cada requisito funcional de [SPEC.md](../SPEC.md). Leyenda: ✅ hecho · 🟡 parcial · ⬜ pendiente.
La columna **Capa** es la capa de §14 donde se implementa.

## Capa 0 — Fundaciones

| Entregable (§14) | Estado |
|---|---|
| Monorepo §12, `CLAUDE.md`, README | ✅ |
| Tooling Python (uv, Ruff, mypy, pytest) | ✅ en CI: ruff + mypy (46 archivos) + 57 tests en Windows y Linux |
| Tooling frontend (pnpm, Vite, ESLint, Prettier, Vitest) | ✅ lint, typecheck, 17 tests y build en verde |
| CI Windows + Linux (`.github/workflows/ci.yml`) | ✅ verde en `das2010/perceptron` |
| Config y logging estructurado | ✅ `perceptron.core` |
| Modelos de dominio base (§6) | ✅ `perceptron.domain` (20 entidades) |
| Almacenamiento local (SQLite + filesystem) | ✅ `perceptron.storage` |
| OpenAPI → TypeScript | ✅ `pnpm gen:api`; el job de contrato verifica que no haya drift |
| Fixtures por caso de uso | ✅ `fixtures/` UC-01…UC-10, determinísticos |
| ADRs de §2 | ✅ `docs/adr/` 0001–0013 |

**Criterios de aceptación: ✅ cumplidos en CI** — `uv run pytest` (57 tests, cobertura ≥ 80 % en core/domain), `pnpm test` y
`perceptron --version` pasan en `ubuntu-latest` y `windows-latest`. En la máquina de desarrollo local una política de seguridad
bloquea `.venv\Scripts\python.exe`, por lo que el engine se valida en CI hasta que IT lo habilite.

## Capa 1a — Engine núcleo (tabular + imagen)

| Sub-hito | Estado |
|---|---|
| 1. Dependencias ML + hardware + ADR 0014–0016 | ✅ |
| 2. Ingesta, esquema, splits, DatasetVersion | ✅ |
| 3. Profiling + alertas + ProfileCard | ✅ |
| 4. Pipeline fit/transform + propuesta | ✅ |
| 5. Catálogo + ArchSpec + reglas | ✅ |
| 6. Entrenamiento (worker + supervisor) | ✅ |
| 7. Tracking MLflow | ✅ |
| 8. HPO Optuna | ✅ |
| 9. Evaluación + registro | ✅ |
| 10. Jobs, API/WS, CLI, quickstart, E2E UC-01/UC-04 | ✅ |
| Caso «Tabla 3»: la única entrada no se descarta por parecer id (aviso `id_like_feature`); sin entradas el pipeline se rechaza (422) y el perfil alerta `no_features`; tipos corregibles desde Datos (RF-ING-06); métricas de validación de regresión en unidades reales | ✅ |
| Con una sola entrada, asociación ≈ 1 se informa como relación determinística (`deterministic_relation`), no como fuga; Experimentos marca el mejor run de cada estudio | ✅ |
| Regresión lineal pura sin weight decay (sesgo en la pendiente); épocas por trial sugeridas por la arquitectura según el tiempo estimado | ✅ |
| Archivos con coma decimal (`0,25` con separador `;` o `|`) se leen como números; corregir texto a numérica convierte los valores o responde 422 (antes, perfil con HTTP 500) | ✅ |
| Fórmula sugerida: regresión simbólica (PyOperon) como modelo de referencia en Experimentos, con prueba de valores y copia a Python/Excel (ADR-0039) | ✅ |
| Wizard adaptativo (ADR-0040, fase 1): ficha del caso con entrevista aceptable, hechos medidos de los datos y plan compilado (pasos, defaults con «por qué», chequeos) | ✅ |
| Wizard adaptativo (ADR-0040, fase 2): reconciliación de la ficha con los datos (reglas y LLM), pasos condicionales Fórmula sugerida y Umbral de decisión, umbral por costo en la evaluación, aviso de cambios del plan, golden tests con LLM real | ✅ |
| Caso «Sensores»: alertas de fuga ajustadas a la ficha (regla/extrapolar → relación determinística), costos asimétricos solo en clasificación (aviso en regresión, entrevista v3), agente con base lineal y la ficha en su contexto, paso Fórmula también al extrapolar | ✅ |
| Wizard que diseña (ADR-0041, iteración 1): requisitos de diseño por escenario (preentrenada con pocas imágenes, modelo chico en tablas chicas, opción lineal si hay que extrapolar, topes para edge, desbalance, tiempo por época), el arquitecto los recibe y reintenta si falta un obligatorio, plantilla que lo cubre si no, propuesta recomendada primera con su evaluación | ✅ |
| Wizard que diseña (ADR-0041, iteración 2): diseño guiado de punta a punta en un job (`POST /projects/{id}/draft/design`): preparación, propuestas evaluadas, mini-torneo de las mejores que cumplen los obligatorios (métrica común, presupuesto según el tamaño de los datos), elegida con evidencia, épocas por trial y estrategia de HPO; queda en el borrador para aceptar | ✅ |
| Wizard que diseña (ADR-0041, iteración 3): «Próximo paso» en cada run — las acciones del diagnóstico como cambios concretos de la arquitectura desde el mejor punto (lr, regularización, épocas, desbalance), aplicar y entrenar con un clic; lo que no es de la arquitectura indica dónde hacerlo | ✅ |
| Wizard que diseña (ADR-0041, iteración 5): memo «Por qué esta arquitectura» en la revisión (motivo, requisitos que cumple y que no, aviso si se eligió otra) y botón para usar la búsqueda del diseño guiado en el paso HPO | ✅ |
| Wizard que diseña (ADR-0041, iteración 6): «Aceptar y revisar» — un clic fija arquitectura, búsqueda y épocas del diseño guiado y pasa a la revisión con el memo | ✅ |
| La `val_loss` se mide sin suavizado de etiquetas (el entrenamiento sí lo usa): comparable entre trials del HPO con distinto `label_smoothing` y sin sesgar su importancia en el análisis del estudio | ✅ |
| Progreso del trial en curso en Entrenamientos y Estudios: barra con época actual, total y tiempo restante (del evento por época del job); la tabla se refresca con un estudio activo aunque se entre desde el menú | ✅ |
| Caso «Tabla X»: ajuste exacto por mínimos cuadrados al final de una regresión lineal (y un solo intento), constantes con forma cerrada en la fórmula sugerida (3^(1/5), π, e) y sin términos despreciables, estado «Podado», sin poda con menos de 15 trials y con el mejor valor de cada curva, sugerencias repetidas del HPO sin reentrenar | ✅ |
| Plan de intentos y épocas del HPO (ADR-0041): el sistema los propone según los hiperparámetros a buscar, el tiempo medido por época y el tiempo que se quiere esperar (`POST /projects/{id}/hpo/plan`), con sus motivos; el estratega LLM parte del plan y puede bajarlo con justificación; Entrenar, el paso HPO del wizard y el diseño guiado lo usan | ✅ |
| Caso «Tabla X» (curva): forma de los datos medida (R² de una recta vs. una curva) → requisito «capas ocultas» o «lineal obligatoria»; diagnóstico con errores con patrón y la fórmula sugerida como referencia; fórmula con raíces de fracciones (√(2/7)) y casi-enteros; aviso al registrar un modelo mucho peor que la fórmula | ✅ |

**Aceptación:** job `e2e` de CI — `perceptron quickstart` con 10 trials sobre UC-01 (ROC-AUC > 0,75) y UC-04 (accuracy > 0,8) → modelo evaluado en test sellado y registrado.

## Capa 1b — texto, series, audio, visión avanzada

| Sub-hito | Estado |
|---|---|
| 0. Refactor TaskAdapter (ADR-0017) | ✅ |
| 1. Texto — UC-03 | ✅ |
| 2. Series — UC-07 forecasting, UC-08 anomalías | ✅ |
| 3. Audio — UC-09 (ADR-0018) | ✅ |
| 4. Visión avanzada — UC-04 detección, UC-05, UC-06 (ADR-0019) | ✅ |
| 5. Baseline LightGBM, oversampling | ✅ |

**Aceptación (matriz e2e, `quickstart` con 10 trials en CPU, sin descargar pesos):**

| Caso | Arquitectura (reglas) | Test sellado | Umbral |
|---|---|---|---|
| UC-01 churn | MLP | ROC-AUC 0,906 (LightGBM de referencia: 0,910) | > 0,75 |
| UC-03 tickets (texto) | TextCNN | accuracy 1,0 | > 0,9 |
| UC-04 defectos — clasificación | CNN compacta | accuracy 1,0 | > 0,8 |
| UC-04 defectos — detección | CenterNet compacto | mAP@.5 1,0 | > 0,5 |
| UC-05 daños (segmentación) | U-Net compacta | IoU daño 1,0 | > 0,5 |
| UC-06 remitos (OCR) | CRNN + CTC | CER 0,0 | < 0,1 |
| UC-07 demanda (forecasting) | N-BEATS + RevIN | sMAPE 4,5 %, MASE 0,79 (naive 0,97) | < 15 % y < 1 |
| UC-08 telemetría (anomalías) | autoencoder conv | F1 0,71 (recall 1,0, ROC-AUC 0,99) | > 0,7 |
| UC-09 motores (audio) | CRNN sobre log-mel | accuracy 1,0 | > 0,8 |

Los fixtures son sintéticos y chicos: métricas perfectas indican que el flujo funciona, no rendimiento en datos reales (eso lo mide el benchmark O2, §15.4).

**Estabilidad del quickstart:** el HPO arranca siempre por la configuración de la plantilla (trial 0) y, con presupuestos chicos, el early stopping y la poda por mediana esperan un tercio de las épocas antes de cortar. Sin esto, UC-09 quedaba en 0,72 (una meseta de validación cerca de la época 10).

**Pendiente, documentado:** detectores de torchvision y U-Net con encoder timm (ADR-0019), detección de eventos sonoros (SED), LoRA/PEFT, fusión tabular + texto (UC-02).

## Capa 2a — LLM Gateway, privacidad, prompts y roles

| Sub-hito | Estado |
|---|---|
| 0. Configuración, catálogo de modelos, secretos, caché (ADR-0020) | ✅ |
| 1. Gateway + 6 adaptadores + `FakeLLMProvider` (ADR-0021) | ✅ |
| 2. PrivacyFilter L0–L3, PII, auditoría y test de propiedad (ADR-0022) | ✅ |
| 3. Prompts versionados (`llm/prompts/<propósito>/<nombre>-vN.md`) | ✅ |
| 4. Roles: arquitecto, estratega de HPO, diagnosticador, informante, etiquetador | ✅ |
| Presupuesto atómico: cada llamada reserva su costo máximo (salida completa) hasta quedar auditada; llamadas concurrentes no superan el límite (RF-LLM-06) | ✅ |

**Aceptación 2a (CI de cada PR, con `FakeLLMProvider`):** salida válida por rol, reintento con feedback, fallback a reglas (L0, sin clave, presupuesto, 3 fallos), caché gratis en la segunda llamada, auditoría de cada intento y ningún valor individual en payloads L1 (hypothesis). La aceptación con Claude y Ollama es de la Capa 2b.

## Capa 2b — mini-torneo, agente autónomo, aceptación real y benchmark

| Sub-hito | Estado |
|---|---|
| 5. Mini-torneo de arquitecturas (RF-ARC-03) | ✅ `services.tournament`, `POST /projects/{id}/arch/tournament`, en `quickstart --llm` |
| 6. Agente autónomo (RF-AGT-01..05, ADR-0021) | ✅ `agent/` (acciones con schema, límites, aprobaciones, bitácora, fallback); API `/agent/runs`, CLI `perceptron agent` |
| 7. Aceptación con OpenAI (ADR-0023) y Ollama + golden tests | ✅ OpenAI y Ollama completan UC-01/04/09 sin fallback y sin fugas; golden OpenAI 5/5, Ollama 4/5 (estratega: fix de parámetros incompletos) |
| 8. Benchmark O2 (3 datasets públicos) | ✅ O2 cumplido con OpenAI: brecha −0,8 % / 0 % / 0 % ([reporte](benchmarks/2026-09-27.md)) |

**Aceptación real con OpenAI (`llm.yml`, 2026-09-27, perfil `openai`, L1):**

| Caso | Test sellado | Umbral | Iteraciones / trials | Llamadas LLM | Costo | Fugas L1 |
|---|---|---|---|---|---|---|
| UC-01 churn | ROC-AUC 0,918 | > 0,75 | 2 / 12 | 10 | $0,18 | 0 |
| UC-04 defectos | accuracy 1,0 | > 0,8 | 2 / 7 | 8 | $0,11 | 0 |
| UC-09 motores | accuracy 1,0 | > 0,8 | 1 / 4 | 6 | $0,06 | 0 |

Golden con OpenAI: estratega, diagnosticador, informante y etiquetador ✅; arquitecto con timeout del SDK (120 s) → subido a 600 s. Con Ollama (qwen3:4b en la CPU del runner, modo compacto): el agente completa los tres UC sin fallback, dentro del presupuesto y sin fugas — UC-01 ROC-AUC 0,929, UC-04 accuracy 1,0, UC-09 accuracy 1,0 (cierra solo con `finish`). Golden con Ollama: arquitecto, diagnosticador (detecta el sobreajuste), informe y guía ✅; el estratega omitía límites de rango, ahora completados con el catálogo antes de validar.

**Aceptación 2b en cada PR (FakeLLMProvider):** guiones del agente sobre UC-01 que cubren camino feliz, feedback del validador, límites, aprobación, fallback a reglas, detención, que el agente nunca ve el test y la auditoría sin fugas en L1. **Aceptación real (§14):** `gh workflow run llm.yml` → matriz Claude/Ollama × UC-01/04/09, golden tests y (opcional) benchmark.

## Capa 3a — shell de la UI, sistema de diseño y recorrido mínimo

| Entregable | Estado |
|---|---|
| Stack §5.2 (Router, Query, Zustand, Tailwind + Radix, ECharts; ADR-0024) | ✅ |
| Layout §11.2 (sidebar, área principal, panel de copiloto plegable), tema claro/oscuro, i18n es/en | ✅ |
| Inicio (Engine, hardware, proyectos), Proyecto (Resumen, Datos, Entrenar, Experimentos, Run, Modelos, Auditoría LLM), Configuración LLM | ✅ |
| Subida de archivos/carpetas desde el navegador (`POST /projects/{id}/uploads`), historia por época (`GET /runs/{rid}/history`) | ✅ |
| E2E Playwright UC-01 guiado contra el Engine real (job `e2e-ui`) | ✅ 32 s de punta a punta (O1: < 15 min) |

## Capa 3b — wizard, copiloto, agente y editores visuales

| Entregable | Estado |
|---|---|
| `ProjectDraft` versionado (`GET/PATCH /projects/{id}/draft`, bloqueo optimista, historial con origen) | ✅ |
| Wizard de 9 pasos (§7.6) para tabular e imagen, sin LLM por reglas | ✅ |
| Copiloto en streaming (`WS /projects/{id}/copilot`) con `DraftPatch` validado, aceptable o rechazable | ✅ |
| Vista del agente: lanzamiento con límites y aprobaciones, bitácora en vivo, aprobar/rechazar/detener | ✅ |
| Editor visual de ArchSpec (React Flow) con validación en vivo y "ver como código" (Monaco empaquetado, sin CDN) | ✅ |
| Editor visual de pipeline (React Flow): agregar, quitar, reordenar, parametrizar, vista previa | ✅ |
| Sub-wizard de definición de arquitectura (familia → backbone → cabeza → regularización, `POST /projects/{id}/arch/define`), con el copiloto como guía | ✅ tabular e imagen |
| Comparación de runs en Experimentos (curvas superpuestas, métricas e hiperparámetros que difieren) | ✅ |
| Modo experto con Monaco (RF-ARC-06): lint en vivo, confirmación explícita y prueba en el sandbox | ✅ |
| E2E Playwright UC-04 guiado por el wizard (incluye editor visual y código) | ✅ 40 s de punta a punta (O1: < 15 min) |

## Capa 3c — desktop (Tauri 2) y runtime embebido

| Entregable | Estado |
|---|---|
| App Tauri 2 con el Engine como sidecar (`serve --new-token`, puerto aleatorio, token efímero) y `TauriPlatformBridge` | ✅ |
| Runtime Python embebido con uv y variante de PyTorch por hardware; cambio de variante sin reinstalar (RF-TRN-02, ADR-0026) | ✅ en CI se verifica la variante `cpu`; CUDA/ROCm/XPU salen del mapa de índices y no hay runners con GPU para probarlas |
| Keychain del SO (Credential Manager / Secret Service) y selector de carpetas nativo | 🟡 implementado, sin prueba automatizada (necesita sesión de escritorio) |
| Instaladores NSIS/MSI y .deb/AppImage con smoke test de instalación limpia (`desktop.yml`) | ✅ Windows (NSIS silencioso) y Ubuntu 24.04 (.deb): aprovisionan desde cero, detectan `cpu` y el Engine responde |
| Updater firmado y firma de código de los instaladores | 🟡 updater firmado (ADR-0037); Authenticode postergado |


## Capa 4a — export, serving y playground

| Entregable | Estado |
|---|---|
| Export verificado ONNX/fp16/INT8, torch.export y TorchScript con firma (RF-EXP-01/05, ADR-0027) | ✅ |
| Servidor de inferencia FastAPI + ONNX Runtime con Dockerfiles CPU/CUDA (RF-EXP-03) | ✅ |
| Proyecto de código exportable autónomo con uv (RF-EXP-04) | ✅ |
| Playground en la página del run (RF-EXP-02) | ✅ tabular, imagen, texto y audio |
| Aceptación O5: proyecto y servidor en contenedores limpios (job `export-o5`) | ✅ uv sync → train → infer → pytest en `python:3.12-slim`; servidor Docker con API key |


## Capa 4b — evaluación avanzada e informe

| Entregable | Estado |
|---|---|
| Análisis de errores (RF-EVL-03) y fairness (RF-EVL-04) sobre el test sellado | ✅ |
| Explicabilidad con Captum (RF-EVL-02) | ✅ tabular, imagen, texto, audio y series |
| Robustez ante perturbaciones (RF-EVL-05) | ✅ tabular, imagen, texto y audio |
| Informe HTML/PDF/Markdown con marca y model card (RF-EVL-06, ADR-0028) | ✅ |
| Explicación local en el playground | ✅ |


## Capa 4c — etiquetado asistido y fuentes

| Entregable | Estado |
|---|---|
| Conjuntos de etiquetas con origen/confianza, cola de active learning y aplicar como versión nueva (ADR-0029) | ✅ |
| Herramienta de etiquetado en la UI (clase, multi-etiqueta, cajas; atajos y aceptación en lote) | ✅ |
| Aceptación UC-04: las pre-etiquetas reducen ≥ 50 % las acciones manuales (test de API) | ✅ |
| Fuentes SQL y datasets públicos (HF, Kaggle) | ✅ |


## Capa 5a — Team Server: auth local, RBAC, auditoría y UI web

| Entregable | Estado |
|---|---|
| `perceptron-server`: el mismo Engine con política de acceso enchufable, PostgreSQL 16 y Alembic (ADR-0030) | ✅ |
| Login local (Argon2id, JWT + refresco rotativo, cookies HttpOnly/SameSite y CSRF; bearer para CLI) | ✅ |
| RBAC Admin/Editor/Viewer por workspace y proyecto en todas las operaciones del Engine (HTTP y WS) | ✅ |
| Auditoría append-only y consola de administración (usuarios, roles, auditoría) | ✅ |
| Modo estación de trabajo: UI web servida por el servidor, fuentes del servidor | ✅ |
| Docker Compose (servidor + PostgreSQL) con smoke en CI | ✅ |
| Aceptación: dos usuarios con roles distintos colaboran; UC-07 solo desde el navegador | ✅ (CI: server, e2e web) |

## Capa 5b — cola de jobs, workers y MLflow server

| Entregable | Estado |
|---|---|
| Estudios autodescriptivos y `StudyLauncher` enchufable (local en desktop, cola en el servidor) (ADR-0031) | ✅ |
| Celery sobre Valkey con colas `gpu`/`cpu`, workers `perceptron-server worker`, cancelación | ✅ |
| Progreso en vivo de jobs remotos por relay pub/sub (mismos WS que los jobs locales) | ✅ |
| Cuotas de estudios en curso por usuario y workspace; vista de la cola con hardware de los workers | ✅ |
| MLflow server (PostgreSQL + artefactos) e imagen GPU (`TORCH_VARIANT`) en Compose | ✅ |
| Robustez: cancelación persistida (un estudio cancelado en la cola no entrena); agente y mini-torneo por la cola y las cuotas; jobs que esperan estudios en hilo propio (sin deadlock del pool); WS con reconexión | ✅ |
| Jobs de la cola persistidos: tras reiniciar el servidor se retoman (visibles, cancelables y contando para las cuotas); el worker anota inicio y fin en la base | ✅ |
| Control de estudios: estado (en curso, detenido, interrumpido, terminado), Detener/Reanudar desde Experimentos (reanudar conserva los trials), detección de jobs huérfanos cuando el worker se reinicia | ✅ |

## Capa 5c — SSO, Helm, backups y sync

| Entregable | Estado |
|---|---|
| SSO OIDC (Entra ID, Google Workspace, genérico) con PKCE, mapeo de grupos a roles y botones en el login (ADR-0032) | ✅ (CI con IdP simulado; falta Entra ID real) |
| Helm chart v1 (servidor, workers CPU/GPU, Valkey, MLflow) validado con kubeconform | ✅ |
| Backups y restauración (PostgreSQL + volúmenes) probados de punta a punta en CI | ✅ |
| Sync desktop ↔ servidor: proyectos de equipo con mismos IDs, bloqueo optimista, subida resumible por chunks con SHA-256 | ✅ |
| Aceptación: un desktop lanza un run en un worker del servidor y ve el progreso en vivo (HTTP + WS reales en CI) | ✅ (worker CPU en CI; GPU real pendiente de recurso) |

## Capa 6a — serving monitoreado, drift, alertas y champion/challenger

| Entregable | Estado |
|---|---|
| Deployments sobre el ONNX del champion con registro muestreado de predicciones y feedback (ADR-0033) | ✅ |
| Drift de datos (PSI, KS, χ², JS), de salida del modelo (MMD, dominio) y de performance | ✅ |
| Alertas en la app, email y webhook con cooldown | ✅ |
| Champion/challenger sobre el mismo holdout, promoción solo si mejora, rollback | ✅ |
| Diff entre versiones (filas, esquema, distribución, archivos) y linaje | ✅ |

## Capa 6b — reentrenamiento automático y fuentes streaming

| Entregable | Estado |
|---|---|
| Fuentes REST (paginación, auth), WebSocket y archivo que crece, con buffer y sondeo (ADR-0034) | ✅ |
| RetrainPolicy: drift, cron, volumen y degradación; aprobación opcional; bitácora | ✅ |
| Versión de datos con split predefinido (test del champion fijo) y linaje append | ✅ |
| Aceptación UC-10: drift → alerta → reentrenamiento → challenger → promoción si mejora → rollback | ✅ (test de API) |
| Kafka/MQTT (extras opcionales), embeddings internos para no estructurado | ✅ extra `streaming`; drift de embeddings en tabla, texto, imagen y audio |

## Capa 7a — licencia, auditoría, telemetría y actualización

| Entregable | Estado |
|---|---|
| Licencia firmada Ed25519 offline con topes y uso real (ADR-0035) | ✅ (clave pública preteco-2026 empaquetada) |
| Auditoría de licencias del runtime Python/JS/Rust y de los pesos en CI (ADR-0036) | ✅ |
| Telemetría opt-in con vista previa, sin endpoint de fábrica | ✅ (D7: destino a definir) |
| Actualización N → N+1 sin pérdida de proyectos (CI) | ✅ |
| Updater firmado: aviso, instalación desde Configuración, release con latest.json (ADR-0037) | ✅ (CI: N → N+1 real con el updater) |
| Firma Authenticode de los instaladores | ⬜ (certificado de Preteco, postergado) |
| Documentación de usuario es/en (MkDocs, compilada en CI) | ✅ |

## Capa 7c — hardening de seguridad (OWASP ASVS nivel 2, ADR-0038)

| Entregable | Estado |
|---|---|
| Rutas de cliente seguras en Windows y Linux (`safe_parts`, `ensure_within`, ids en sync) | ✅ |
| SSRF: destinos internos bloqueados en el servidor (fuentes API/WS, webhooks, DB), sin redirects | ✅ |
| Servidor sin rutas propias como fuente si no se configuran `SOURCE_ROOTS` | ✅ |
| Errores sin tracebacks ni eco de datos; logs sin tokens | ✅ |
| Cabeceras de seguridad y `no-store` en la API; Swagger apagado en el servidor | ✅ |
| Límites de uploads y ZIPs (zip bomb) | ✅ |
| Sesiones con vida absoluta, `Origin` en WebSockets, SSO con email verificado (nOAuth) | ✅ |
| Desktop: token fuera de los logs, keychain acotado a `ui.*`, CSP más estricta | ✅ |
| CI: `contents: read`, cargo audit, CodeQL, Dependabot; `torch>=2.6` | ✅ |
| Verificación ASVS L2 control por control (`docs/security/asvs-l2.md`) | ✅ (riesgos aceptados documentados) |

## Requisitos funcionales

| RF | MVP | Capa | Descripción | Estado |
|---|---|---|---|---|
| RF-PRJ-01 | sí | 1/3 | Crear, abrir, duplicar, archivar y eliminar proyectos. | ✅ crear, abrir, duplicar (configuración), archivar/restaurar y eliminar con datos y carpeta (API, CLI y UI) |
| RF-PRJ-02 | sí | 1/3 | Plantillas de proyecto por caso de uso (UC-01…UC-09) que preconfiguran modalidad,… | ✅ plantillas UC-01…UC-09 (modalidad, tarea y métrica `val_*`), selector en Nuevo proyecto y `--template`; el HPO optimiza la métrica del proyecto |
| RF-PRJ-03 |  | 3 | Exportar/importar proyecto como paquete .perceptron (zip con manifiesto; datos… | ✅ paquete `.perceptron` (manifiesto con proyecto y entidades, runs y modelos; datos opcionales), import con los mismos ids y rutas reescritas; API, CLI y UI |
| RF-PRJ-04 |  | 5 | Promover un proyecto local a proyecto de equipo (sube metadata, datasets y runs… | ✅ promoción explícita desde el desktop: metadata, datasets elegidos (y los de los runs), pipelines, arquitecturas y runs elegidos con evaluaciones y artefactos; el proyecto local queda como de equipo |
| RF-PRJ-05 |  | 3 | Historial de actividad del proyecto (quién hizo qué, cuándo). | ✅ cada escritura exitosa queda como ActivityEntry (operación, usuario del Team Server o local, cuándo); `GET /projects/{id}/activity` para cualquier miembro y Actividad reciente en el resumen |
| RF-ING-01 | sí | 1 | Archivos locales: CSV, TSV, XLSX, Parquet, JSON/JSONL; carpetas de imágenes… | ✅ tabular, imágenes, texto (tabla o clase/*.txt), audio (wav/flac/mp3/ogg), ZIP |
| RF-ING-02 |  | 1/4 | Formatos de anotación: COCO, Pascal VOC, YOLO (txt), máscaras PNG, CSV de eventos de… | ✅ COCO, Pascal VOC, YOLO, máscaras PNG, CSV de OCR y CSV de eventos de audio (import/export en conjuntos de segmentos) |
| RF-ING-03 |  | 4 | Bases de datos: SQL Server, PostgreSQL, MySQL/MariaDB, SQLite, vía query SQL con vista… | ✅ SQL Server, PostgreSQL, MySQL/MariaDB y SQLite con consulta, vista previa, límite de filas, lectura por lotes y refresco |
| RF-ING-04 |  | 4 | Datasets públicos: Hugging Face Datasets y Kaggle (con credenciales del usuario),… | ✅ Hugging Face Datasets (split, token) y Kaggle (API oficial), con credenciales en el llavero |
| RF-ING-05 |  | 6 | APIs REST (paginación, auth por header/token, mapeo JSON → tabla) y streaming (Kafka,… | ✅ REST paginado con auth, WebSocket, archivo que crece, Kafka (grupo con commit manual, SASL/SSL) y MQTT (sesión persistente, QoS 1) sobre `StreamSource`, con buffer, sondeo y política de red; UI en Monitoreo: alta por tipo, estado del buffer, lectura manual y selección en el reentrenamiento |
| RF-ING-06 | sí | 1 | Inferencia de esquema y tipos (numérico, categórico, fecha, texto, id, ruta de… | ✅ `data.schema`: tipos semánticos + candidatos a target, override manual |
| RF-ING-07 | sí | 1 | Cada ingesta crea un DatasetVersion inmutable con hash de contenido. | ✅ `DatasetVersion` inmutable content-addressed (ADR-0016) |
| RF-ING-08 | sí | 1 | Particionado: aleatorio estratificado, por grupo (evitar leakage entre entidades),… | ✅ aleatorio, estratificado, grupo, temporal (por serie), k-fold; test sellado |
| RF-ING-09 |  | 1 | Soporte hasta ~10 GB: los datos no tabulares se leen en streaming desde disco; los… | ✅ tabulares materializados en Parquet; si el split supera ~2 GB en memoria se entrena por lotes (pyarrow, shuffle con buffer, pesos de clase por conteo); imágenes y audio se leen desde disco por muestra |
| RF-PRF-01 | sí | 1 | Estadísticas por columna (tabular): tipo, nulos, cardinalidad, distribución, outliers,… | ✅ `profiling.tabular` |
| RF-PRF-02 | sí | 1 | Imágenes: resolución, canales, formatos, corruptas, duplicados/casi-duplicados (hash… | ✅ `profiling.images` (dHash + LSH para casi duplicados) |
| RF-PRF-03 |  | 1 | Texto: idioma, longitud en tokens, vocabulario, duplicados, balance. | ✅ `profiling.text` (idioma, largo en tokens, vocabulario, duplicados) |
| RF-PRF-04 |  | 1 | Series: frecuencia, gaps, estacionalidad, tendencia, número de series, horizonte factible. | ✅ `data.series` (frecuencia, gaps, estacionalidad, tendencia, horizonte factible) |
| RF-PRF-05 |  | 1 | Audio: duración, sample rate, canales, silencio, clipping, SNR estimado. | ✅ `profiling.audio` (duración, sample rate, canales, silencio, clipping, SNR) |
| RF-PRF-06 | sí | 1 | Alertas: desbalance, target con fuga (feature casi idéntica al target, ids, fechas… | ✅ `profiling.alerts` |
| RF-PRF-07 | sí | 1 | Genera un Dataset Profile Card (JSON + vista) que es la entrada principal del LLM en… | ✅ `ProfileCard` sin valores individuales (test de propiedad) |
| RF-PRF-08 |  | 1 | Estimación de complejidad y costo: tamaño efectivo, memoria estimada por batch, tiempo… | ✅ tamaño efectivo (train, total, disco y tensores de entrada), parámetros, memoria por batch y tiempo por época medido en cada dispositivo disponible, con aviso si no entra en memoria |
| RF-PIP-01 | sí | 1/3 | El sistema propone automáticamente un pipeline según profiling (imputación, encoding,… | ✅ `pipeline.propose` con justificación por paso |
| RF-PIP-02 |  | 1/3 | Editor visual (React Flow) de un DAG de pasos: agregar, quitar, reordenar,… | ✅ editor React Flow (agregar, quitar, reordenar, parametrizar, guardar con versión) con vista previa por paso del grafo sin guardar (`POST /projects/{id}/pipelines/preview-steps`) |
| RF-PIP-03 |  | 1/3 | Catálogo de pasos por modalidad (extensible por plugins): | ✅ tabular, imagen, texto, series, audio (augmentations incluidas) |
| RF-PIP-04 | sí | 1/3 | El pipeline se serializa (JSON) y se ajusta solo con train (fit/transform separado)… | ✅ fit solo con train, estado JSON empaquetable |
| RF-PIP-05 |  | 1/3 | El LLM puede sugerir cambios al pipeline con justificación; el usuario acepta/rechaza… | ✅ el LLM (propósito architect, prompt `pipeline`) sugiere cambios tipados con justificación, validados contra el catálogo de pasos y las columnas; fallback por reglas; diff antes/después y se aplican solo las aceptadas (bloqueo optimista) |
| RF-LBL-01 |  | 4 | Herramientas de etiquetado: clase por muestra (imagen/texto/audio), multi-etiqueta,… | ✅ clase por muestra (imagen/texto/tabla/audio), multi-etiqueta, cajas, polígonos (→ máscaras PNG de segmentación con nombres de clase) y segmentos temporales de audio (→ un clip por evento + `eventos.csv`); JSONL con formas; tests API y UI |
| RF-LBL-02 |  | 4 | Pre-etiquetado automático con: | ✅ modelo del proyecto, LLM (texto, L2/L3) y zero-shot local con nombres de clase: SigLIP (imagen, Apache-2.0), mDeBERTa-NLI (texto, MIT) y CLAP (audio, Apache-2.0), descargables desde la caché de modelos; backend reemplazable y tests API/UI |
| RF-LBL-03 |  | 4 | Active learning: priorizar para revisión humana las muestras de mayor… | ✅ cola por incertidumbre/diversidad, aceptación en lote por confianza y ciclo etiquetar → aplicar (nueva versión) → reentrenar → pre-etiquetar |
| RF-LBL-04 |  | 4 | Guía de etiquetado: el usuario describe las clases en lenguaje natural; el LLM genera… | ✅ guía de etiquetado por LLM (`POST /projects/{id}/labels/guide`); sin LLM, las definiciones del usuario |
| RF-LBL-05 |  | 4 | Métricas de calidad del etiquetado: acuerdo humano-modelo, clases confusas, posibles… | ✅ acuerdo humano-modelo, pares confusos y posibles errores de etiqueta |
| RF-LBL-06 |  | 4 | Importar/exportar etiquetas en COCO, YOLO, VOC, CSV, JSONL. | ✅ CSV, JSONL, COCO, YOLO y VOC (import y export) |
| RF-WIZ-01 | sí | 3 | Pasos 1, 2, 3, 5, 6, 7, 8, 9 para tabular e imagen. | ✅ los 9 pasos para tabular e imagen (UI + `ProjectDraft`) |
| RF-WIZ-02 |  | 3 | Cada paso muestra "¿Por qué?" con la explicación del LLM y permite rechazar/editar. | ✅ "¿Por qué?" en cada paso (pregunta al copiloto con el contexto del borrador); propuestas del LLM con justificación, aceptables o rechazables |
| RF-WIZ-03 |  | 3 | Sin LLM configurado (o privacidad L0) el wizard funciona con recomendaciones por… | ✅ sin LLM (o L0) cada paso usa las recomendaciones por reglas |
| RF-WIZ-04 |  | 3 | El estado del wizard es un documento ProjectDraft versionado. | ✅ `ProjectDraft` versionado con bloqueo optimista e historial de cambios (usuario/copiloto) |
| RF-LLM-01 | sí | 2 | Interfaz única LLMProvider con adaptadores: Anthropic, OpenAI, Google Gemini, Kimi /… | ✅ `llm.providers`: Anthropic, OpenAI, Gemini, Kimi/Moonshot, OpenAI-compatible, Ollama |
| RF-LLM-02 | sí | 2 | Capacidades declaradas por modelo: structured_output, tool_use, vision,… | ✅ capacidades en `llm/catalog.yaml`; sin `structured_output` → JSON en texto; sin `vision` no salen imágenes |
| RF-LLM-03 | sí | 2 | Perfiles LLM configurables: qué modelo se usa para cada *propósito* (copilot,… | ✅ perfiles por propósito (catálogo + `PUT /llm/profiles`), `Project.llm_profile_id` |
| RF-LLM-04 | sí | 2 | Salida estructurada: toda respuesta que alimenta al sistema se pide como JSON contra… | ✅ JSON Schema por proveedor + Pydantic + validador de dominio, 3 intentos con feedback |
| RF-LLM-05 |  | 2 | Streaming de respuestas al panel de copiloto. | ✅ `Gateway.stream_chat` + `WS /projects/{id}/copilot` al panel del copiloto |
| RF-LLM-06 |  | 2 | Control de costos: presupuesto por proyecto/run/usuario, conteo de tokens, estimación… | ✅ costo por llamada (catálogo), presupuesto por proyecto y por ámbito (run/agente) con corte previo, y cuota mensual por workspace en el Team Server (Admin → Políticas) |
| RF-LLM-07 |  | 2 | Caché de respuestas por hash de (prompt, modelo, parámetros) para reproducibilidad y… | ✅ tabla `llm_cache` por hash de proveedor + request |
| RF-LLM-08 |  | 2 | Claves API: en desktop, keychain del SO; en servidor, cifradas (AES-GCM, clave maestra… | ✅ keychain (`keyring`), archivo AES-GCM en servidor, entorno como último recurso; `allowed_llm_providers` |
| RF-PRV-01 | sí | 2 | El PrivacyFilter se aplica en el Gateway (no en cada llamador). | ✅ `llm.privacy.PrivacyFilter` aplicado en el Gateway sobre un `LLMContext` tipado |
| RF-PRV-02 |  | 2 | Con un LLM local el Admin puede permitir L3 aunque la política general sea L1. | ✅ `Workspace.local_llm_max_privacy` para LLM locales |
| RF-PRV-03 | sí | 2 | Log de auditoría: el usuario puede ver exactamente qué payload se envió en cada… | ✅ `LLMCall` con payload post-filtro, redacciones, intento y costo; `GET /llm/audit`, `perceptron llm audit`; `find_leaks` |
| RF-PRV-04 |  | 2 | Política de workspace: nivel máximo permitido y proveedores permitidos, definidos por… | ✅ nivel máximo, tope con LLM local y proveedores permitidos por workspace, editables por el Admin; se aplica la política del workspace de cada proyecto |
| RF-ARC-01 | sí | 2 | El LLM genera 2–4 propuestas usando exclusivamente bloques del catálogo (§8) en… | ✅ arquitecto LLM: 2–4 propuestas con justificación, pros/contras/riesgos y estimaciones del sistema (parámetros, memoria, tiempo por época medido) |
| RF-ARC-02 | sí | 1/2 | Validación de cada propuesta: schema, compatibilidad de shapes (construcción en meta… | ✅ `archspec.validate` (6 etapas de §9.3) sobre cada propuesta del LLM, con feedback al reintento |
| RF-ARC-03 |  | 2 | Mini-torneo opcional: entrenar cada propuesta con un presupuesto corto (p. ej. 10 %… | ✅ `services.tournament` (fracción de épocas + subconjunto de train por época); gana la mejor en validación |
| RF-ARC-04 | sí | 1/2 | Fallback por reglas si el LLM no está disponible o falla la validación 3 veces. | ✅ `catalog.rules` como fallback (L0, sin LLM, presupuesto, 3 fallos) con motivo informado |
| RF-ARC-05 |  | 3 | Editor visual de ArchSpec (React Flow): bloques del catálogo como nodos, parámetros en… | ✅ editor React Flow: bloques del catálogo, parámetros e HP ajustables, validación en vivo, resumen de parámetros y memoria; guarda como ArchSpec nueva (origen manual) |
| RF-ARC-06 |  | 3 | Modo experto — código libre: el LLM (o el usuario) escribe un… | ✅ modo experto: `build_model(config)` con validación estática (AST) y dinámica en sandbox (sin red ni procesos, E/S solo en el run, rlimits/Job Object); entrenamiento y evaluación en el sandbox; runs marcados no declarativos (ADR-0025) |
| RF-ARC-07 |  | 1/2 | Conversión ArchSpec → código PyTorch legible ("ver como código") para aprendizaje y… | ✅ `archspec.to_code` (equivalencia verificada con pesos) y "ver como código" en la UI (Monaco) |
| RF-HPO-01 | sí | 1 | Estrategias soportadas: | ✅ single/random/grid/TPE/CMA-ES/NSGA-II + median/ASHA/Hyperband |
| RF-HPO-02 | sí | 1 | El LLM estratega recibe el escenario (tamaño de datos, costo por trial, presupuesto,… | ✅ estratega LLM validado contra la ArchSpec, rangos del catálogo y presupuesto; fallback por reglas |
| RF-HPO-03 | sí | 1 | Presupuesto configurable en el wizard: tiempo total, n.º de trials, preset, métrica… | ✅ corte por trials, tiempo y métrica objetivo |
| RF-HPO-04 |  | 1 | Paralelismo de trials según recursos: varias GPUs → un trial por GPU; en servidor,… | 🟡 trials en paralelo en un nodo: uno por GPU (CUDA_VISIBLE_DEVICES por trial) o `parallelism` a mano, con cancelación de todos; repartir trials entre workers del Team Server (storage de Optuna compartido) pendiente |
| RF-HPO-05 |  | 1 | Reanudación de estudios interrumpidos (almacenamiento Optuna en SQLite/PostgreSQL). | ✅ reanudación desde SQLite |
| RF-HPO-06 |  | 1 | Visualizaciones: historia de optimización, importancia de hiperparámetros, coordenadas… | ✅ historia con mejor acumulado, importancia de hiperparámetros (PED-ANOVA de Optuna), coordenadas paralelas y frente de Pareto en Experimentos |
| RF-TRN-01 | sí | 1 | Detección de hardware al inicio y bajo demanda: GPUs (modelo, VRAM, capacidad de… | ✅ `training.hardware` + `GET /system/hardware` |
| RF-TRN-02 | sí | 1/3 | Instalación de PyTorch acorde al hardware: en el primer arranque (y desde… | 🟡 detección y recomendación en el Engine; instalación de la variante en el runtime embebido del desktop y cambio desde Configuración (ADR-0026, validación en CI) |
| RF-TRN-03 | sí | 1 | Construcción del LightningModule desde ArchSpec + configuración de optimizador,… | ✅ `training.module` |
| RF-TRN-04 | sí | 1 | Buenas prácticas por defecto: mixed precision (bf16/fp16 según hardware), gradient… | ✅ AMP, clipping, early stopping, checkpoints, semillas, workers automáticos, batch automático con búsqueda binaria contra OOM en GPU y LR finder opcional |
| RF-TRN-05 | sí | 1 | Cada run corre en un proceso separado; el Engine supervisa, captura OOM/crashes y los… | ✅ subproceso por run con diagnóstico de OOM/crash (ADR-0015) |
| RF-TRN-06 | sí | 1 | Progreso en vivo por WebSocket: época, batch, loss, métricas, LR, throughput, uso de… | ✅ eventos JSONL → EventBus → `WS /runs/{rid}/live` |
| RF-TRN-07 |  | 1 | Pausar, reanudar (desde checkpoint), cancelar. | ✅ cancelar, pausar y reanudar desde checkpoint |
| RF-TRN-08 |  | 1 | Multi-GPU en un nodo (DDP vía Lightning) cuando hay >1 GPU. | 🟡 DDP de Lightning en un nodo cuando un estudio de un solo trial tiene varias GPUs (eventos y resultado solo desde rank 0); lógica probada en CI, falta validarlo en un worker con varias GPUs reales |
| RF-TRN-09 |  | 1 | Técnicas de fine-tuning: congelar backbone, descongelado progresivo, LR… | ✅ congelado del backbone, descongelado progresivo (un grupo de capas por época desde la salida), LR discriminativo (`backbone_lr_mult`) y LoRA con peft para encoders de texto de HF |
| RF-TRN-10 |  | 1 | Manejo de desbalance: pesos de clase, focal loss, sobremuestreo, umbral óptimo… | ✅ pesos de clase, focal, oversampling, umbral óptimo en el reporte |
| RF-TRN-11 |  | 1 | Caché de modelos preentrenados: descarga única, verificación de checksum, uso offline,… | ✅ caché única en el workspace (HF Hub y torch hub; `PERCEPTRON_MODELS_CACHE` para una compartida; en el Team Server, el volumen compartido con los workers), modo offline, verificación de checksums, borrado y predescarga del catálogo curado desde Configuración |
| RF-AGT-01 |  | 2 | Herramientas del agente (tool use): get_profile, get_project_goal,… | ✅ las 14 herramientas como acciones con schema (unión discriminada), ejecutadas por `agent.loop` |
| RF-AGT-02 |  | 2 | Límites duros aplicados por el sistema (no por el LLM): tiempo, n.º de iteraciones,… | ✅ tiempo, decisiones, estudios, trials, costo de LLM (ámbito `agent:<id>`) y disco; el test solo lo abre `finish` |
| RF-AGT-03 |  | 2 | Puntos de aprobación configurables: nunca / antes de cada iteración / solo si cambia… | ✅ nunca / cada iteración / cambio de familia / % de presupuesto; `POST /agent/runs/{id}/approve|reject` |
| RF-AGT-04 |  | 2 | Bitácora legible del agente ("Iteración 3: el modelo sobreajusta desde la época 12 →… | ✅ bitácora en el AgentRun + `WS /agent/runs/{id}/log` |
| RF-AGT-05 |  | 2 | Si el LLM falla o no responde, el loop cae a una política por reglas o se detiene de… | ✅ fallo del LLM → una iteración por reglas o cierre con el mejor modelo |
| RF-TRK-01 | sí | 1 | Todo run se registra en MLflow: parámetros, métricas por paso, artefactos… | ✅ MLflow embebido (SQLite + artefactos) |
| RF-TRK-02 | sí | 1 | La UI de Perceptron muestra runs y comparaciones de forma nativa (no depende de la UI… | ✅ runs, curvas y comparación en la UI; enlace opcional a la UI de MLflow (el desktop la levanta a pedido) |
| RF-TRK-03 |  | 1 | Model Registry de MLflow para ModelVersion y stages. | ✅ registro nativo (ModelVersion con stages, promover, rollback, challenger) reflejado en el Model Registry de MLflow: versión por ModelVersion, stage como tag y alias `champion` |
| RF-TRK-04 |  | 1 | Comparación de runs: tabla, curvas superpuestas, diff de configuración (ArchSpec,… | ✅ tabla de métricas, curvas superpuestas, diff de hiperparámetros y diff de ArchSpec y pipeline (listas por id) |
| RF-EVL-01 | sí | 1 | Métricas por tarea: | ✅ clasificación, regresión, forecasting (MASE, backtesting, naive), anomalías, detección (mAP), segmentación (IoU/Dice), OCR (CER/WER); SED pendiente |
| RF-EVL-02 |  | 1/4 | Explicabilidad: SHAP (tabular, importancia global y local), Integrated Gradients /… | ✅ Captum: Shapley global/local (tabular), Integrated Gradients local (imagen) y global/local sobre el espectrograma (audio), oclusión por token global/local (texto) e Integrated Gradients global sobre la ventana de series (importancia por variable y por rezago; forecasting y anomalías) (ADR-0028) |
| RF-EVL-03 |  | 1/4 | Análisis de errores: explorador de muestras mal predichas con filtros, slices… | ✅ slices de bajo rendimiento, confusiones, posibles errores de etiqueta (confident learning) y explorador de mal predichos |
| RF-EVL-04 |  | 1/4 | Fairness: el usuario marca atributos sensibles; métricas por subgrupo (Fairlearn:… | ✅ métricas por grupo, paridad demográfica e igualdad de oportunidades (definiciones de Fairlearn), alertas por umbral |
| RF-EVL-05 |  | 1/4 | Robustez: sensibilidad a ruido/perturbaciones por modalidad (ruido gaussiano, blur,… | ✅ tabular (ruido, categorías cambiadas, faltantes), imagen (ruido, desenfoque, JPEG), texto (typos, palabras eliminadas) y audio (ruido de fondo por SNR, volumen bajo), a tres severidades |
| RF-EVL-06 | sí | 2/4 | Informe final generado por el LLM (o plantilla sin LLM): resumen ejecutivo, qué se… | ✅ informe LLM o plantilla exportable a HTML (Titillium Web embebida), PDF (reportlab) y Markdown con model card |
| RF-EXP-01 | sí | 4 | Exportar a ONNX (con verificación numérica vs. PyTorch), torch.export… | ✅ ONNX (opset 18, batch dinámico) verificado en ONNX Runtime (1e-4; fp16 1e-2; INT8 informado), torch.export y TorchScript legacy (ADR-0027) |
| RF-EXP-02 | sí | 4 | Playground en la app: cargar un archivo/fila/imagen/audio/texto, ver predicción,… | ✅ playground tabular, imagen, texto y audio con predicción, confianza, probabilidades y explicación local (Shapley, Integrated Gradients y oclusión por token) |
| RF-EXP-03 |  | 4 | API REST de inferencia: generar y levantar un servidor FastAPI (ONNX Runtime o… | ✅ servidor FastAPI + ONNX Runtime con pipeline embebido, /predict, /predict/batch, /metrics, API key, Dockerfile CPU/CUDA (tabular e imagen) |
| RF-EXP-04 |  | 4 | Proyecto de código exportable: repositorio Python standalone generado desde plantillas… | ✅ repo autónomo (uv): modelo generado, pipeline vendorizado, train/infer/serve, config, datos train/val, pesos y prueba de humo; O5 en CI (tabular e imagen) |
| RF-EXP-05 |  | 4 | Firma del modelo: schema de entrada/salida, versión, hash; incluido en todos los formatos. | ✅ firma (entradas, salidas, hash de ArchSpec, run, versión) en el reporte, `signature.json` y en el servidor |
| RF-MON-01 |  | 6 | Registro de predicciones del serving (muestreado, configurable, respetando privacidad)… | ✅ registro muestreado de predicciones (features de la firma + clave) y feedback por id o clave |
| RF-MON-02 |  | 6 | Drift de datos: tabular con Evidently (PSI, KS, Jensen-Shannon, chi²) por feature; no… | ✅ tabular (PSI, KS, χ², JS), espacio de salida y embeddings internos (entrada de la cabeza, expuesta del ONNX al cargar; PCA + MMD, centroides y dominio) para tabla, texto, imagen y audio; deployments de texto (filas) e imagen/audio (`/predict/file`, se guarda solo el embedding) |
| RF-MON-03 |  | 6 | Drift de concepto / performance: métricas sobre datos etiquetados recientes vs. baseline. | ✅ métricas sobre feedback reciente contra el test del champion |
| RF-MON-04 |  | 6 | Alertas: en la app, email (SMTP) y webhook (Teams/Slack genérico). | ✅ en la app, email SMTP y webhook (Teams/Slack) con cooldown |
| RF-MON-05 |  | 6 | Políticas de reentrenamiento (RetrainPolicy): disparadores por drift, calendario… | ✅ disparadores drift, cron, volumen y degradación; HPO reducido; aprobación opcional |
| RF-MON-06 |  | 6 | Champion/challenger: el nuevo modelo se evalúa contra el productivo en un holdout… | ✅ challenger contra champion en el mismo holdout, promoción solo si mejora, rollback |
| RF-MON-07 |  | 6 | Versionado de datasets: snapshots inmutables por manifiesto de hashes… | ✅ versiones inmutables, linaje y diff; retención (últimas N, protegiendo las en uso y sus padres, con vista previa) y export compatible con DVC 3 (.dvc con hash .dir md5) |
| RF-SRV-01 |  | 5 | Autenticación: usuarios locales (hash Argon2) y SSO OIDC (Entra ID, Google Workspace,… | ✅ locales (Argon2id, JWT con refresco rotativo, CSRF, bloqueo) y SSO OIDC con grupos → roles; validación con Entra ID real pendiente de tenant |
| RF-SRV-02 |  | 5 | RBAC por workspace y proyecto (§3.2). | ✅ Admin/Editor/Viewer por workspace y proyecto sobre todas las operaciones (HTTP y WS) |
| RF-SRV-03 |  | 5 | Sincronización desktop ↔ servidor para proyectos de equipo: metadata (PostgreSQL),… | 🟡 push de proyecto, datos, pipeline y arquitectura (bloqueo optimista, chunks reanudables), pull de estudios y runs y edición concurrente: estado por entidad (al día, para bajar, para subir, conflicto), pull con resolución explícita (la del servidor o la mía) y push de lo local, en la API y en el resumen del proyecto; S3 pendiente (D2) |
| RF-SRV-04 |  | 5 | Cola de jobs con prioridades y cuotas por usuario/workspace; los desktops pueden… | ✅ cola Celery/Valkey con colas gpu/cpu, cuotas, vista de la cola y envío desde el desktop con progreso en vivo; prioridades entre usuarios pendientes |
| RF-SRV-05 |  | 5 | Modo estación de trabajo: la UI web del servidor ofrece la misma funcionalidad que el… | ✅ UI web en el mismo origen con login; subida de archivos y fuentes del servidor confinadas |
| RF-SRV-06 |  | 5 | Consola de administración: usuarios, grupos, SSO, proveedores LLM y claves, políticas… | ✅ usuarios, roles, SSO, proveedores y perfiles LLM, políticas de privacidad y cuota LLM por workspace, auditoría global y pestaña Sistema (almacenamiento por proyecto, cuotas de estudios y estado de workers) |
| RF-SRV-07 |  | 5 | Auditoría: login, acceso a datasets, exportaciones, llamadas LLM, cambios de permisos. | ✅ login, escrituras, denegaciones, lecturas de datos, descargas y cambios de roles; LLM en LLMCall |
| RF-SRV-08 |  | 5 | Backups: guía y scripts para PostgreSQL y object storage. | ✅ backup.sh/restore.sh (PostgreSQL + volúmenes) verificados en CI; guía para Kubernetes |
| RF-LIC-01 |  | 0 | Módulo licensing con interfaz LicenseProvider y una implementación DevLicenseProvider… | ✅ LicenseProvider con proveedor de desarrollo y proveedor firmado Ed25519 |
| RF-LIC-02 |  | 0 | Todas las funciones premium consultan features.is_enabled("<feature_key>"); claves de… | ✅ features.is_enabled con claves en features.yaml |
| RF-LIC-03 |  | 7 | La arquitectura debe permitir más adelante: licencia por asiento, por servidor, por… | ✅ archivo firmado offline con topes; sin enforcement en v1 |
