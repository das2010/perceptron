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

## Requisitos funcionales

| RF | MVP | Capa | Descripción | Estado |
|---|---|---|---|---|
| RF-PRJ-01 | sí | 1/3 | Crear, abrir, duplicar, archivar y eliminar proyectos. | 🟡 CRUD por API/CLI; duplicar/archivar desde la UI en Capa 3 |
| RF-PRJ-02 | sí | 1/3 | Plantillas de proyecto por caso de uso (UC-01…UC-09) que preconfiguran modalidad,… | ⬜ pendiente |
| RF-PRJ-03 |  | 3 | Exportar/importar proyecto como paquete .perceptron (zip con manifiesto; datos… | ⬜ pendiente |
| RF-PRJ-04 |  | 5 | Promover un proyecto local a proyecto de equipo (sube metadata, datasets y runs… | ⬜ pendiente |
| RF-PRJ-05 |  | 3 | Historial de actividad del proyecto (quién hizo qué, cuándo). | ⬜ pendiente |
| RF-ING-01 | sí | 1 | Archivos locales: CSV, TSV, XLSX, Parquet, JSON/JSONL; carpetas de imágenes… | ✅ tabular, imágenes, texto (tabla o clase/*.txt), audio (wav/flac/mp3/ogg), ZIP |
| RF-ING-02 |  | 1/4 | Formatos de anotación: COCO, Pascal VOC, YOLO (txt), máscaras PNG, CSV de eventos de… | ✅ COCO, Pascal VOC, YOLO, máscaras PNG, CSV de OCR; eventos de audio pendiente |
| RF-ING-03 |  | 4 | Bases de datos: SQL Server, PostgreSQL, MySQL/MariaDB, SQLite, vía query SQL con vista… | ⬜ pendiente |
| RF-ING-04 |  | 4 | Datasets públicos: Hugging Face Datasets y Kaggle (con credenciales del usuario),… | ⬜ pendiente |
| RF-ING-05 |  | 6 | APIs REST (paginación, auth por header/token, mapeo JSON → tabla) y streaming (Kafka,… | ⬜ pendiente |
| RF-ING-06 | sí | 1 | Inferencia de esquema y tipos (numérico, categórico, fecha, texto, id, ruta de… | ✅ `data.schema`: tipos semánticos + candidatos a target, override manual |
| RF-ING-07 | sí | 1 | Cada ingesta crea un DatasetVersion inmutable con hash de contenido. | ✅ `DatasetVersion` inmutable content-addressed (ADR-0016) |
| RF-ING-08 | sí | 1 | Particionado: aleatorio estratificado, por grupo (evitar leakage entre entidades),… | ✅ aleatorio, estratificado, grupo, temporal (por serie), k-fold; test sellado |
| RF-ING-09 |  | 1 | Soporte hasta ~10 GB: los datos no tabulares se leen en streaming desde disco; los… | 🟡 streaming para tabulares (Polars `sink_parquet`) e imágenes desde disco; shards en 1b |
| RF-PRF-01 | sí | 1 | Estadísticas por columna (tabular): tipo, nulos, cardinalidad, distribución, outliers,… | ✅ `profiling.tabular` |
| RF-PRF-02 | sí | 1 | Imágenes: resolución, canales, formatos, corruptas, duplicados/casi-duplicados (hash… | ✅ `profiling.images` (dHash + LSH para casi duplicados) |
| RF-PRF-03 |  | 1 | Texto: idioma, longitud en tokens, vocabulario, duplicados, balance. | ✅ `profiling.text` (idioma, largo en tokens, vocabulario, duplicados) |
| RF-PRF-04 |  | 1 | Series: frecuencia, gaps, estacionalidad, tendencia, número de series, horizonte factible. | ✅ `data.series` (frecuencia, gaps, estacionalidad, tendencia, horizonte factible) |
| RF-PRF-05 |  | 1 | Audio: duración, sample rate, canales, silencio, clipping, SNR estimado. | ✅ `profiling.audio` (duración, sample rate, canales, silencio, clipping, SNR) |
| RF-PRF-06 | sí | 1 | Alertas: desbalance, target con fuga (feature casi idéntica al target, ids, fechas… | ✅ `profiling.alerts` |
| RF-PRF-07 | sí | 1 | Genera un Dataset Profile Card (JSON + vista) que es la entrada principal del LLM en… | ✅ `ProfileCard` sin valores individuales (test de propiedad) |
| RF-PRF-08 |  | 1 | Estimación de complejidad y costo: tamaño efectivo, memoria estimada por batch, tiempo… | ⬜ pendiente |
| RF-PIP-01 | sí | 1/3 | El sistema propone automáticamente un pipeline según profiling (imputación, encoding,… | ✅ `pipeline.propose` con justificación por paso |
| RF-PIP-02 |  | 1/3 | Editor visual (React Flow) de un DAG de pasos: agregar, quitar, reordenar,… | ⬜ pendiente |
| RF-PIP-03 |  | 1/3 | Catálogo de pasos por modalidad (extensible por plugins): | ✅ tabular, imagen, texto, series, audio (augmentations incluidas) |
| RF-PIP-04 | sí | 1/3 | El pipeline se serializa (JSON) y se ajusta solo con train (fit/transform separado)… | ✅ fit solo con train, estado JSON empaquetable |
| RF-PIP-05 |  | 1/3 | El LLM puede sugerir cambios al pipeline con justificación; el usuario acepta/rechaza… | ⬜ pendiente |
| RF-LBL-01 |  | 4 | Herramientas de etiquetado: clase por muestra (imagen/texto/audio), multi-etiqueta,… | ⬜ pendiente |
| RF-LBL-02 |  | 4 | Pre-etiquetado automático con: | ⬜ pendiente |
| RF-LBL-03 |  | 4 | Active learning: priorizar para revisión humana las muestras de mayor… | ⬜ pendiente |
| RF-LBL-04 |  | 4 | Guía de etiquetado: el usuario describe las clases en lenguaje natural; el LLM genera… | ⬜ pendiente |
| RF-LBL-05 |  | 4 | Métricas de calidad del etiquetado: acuerdo humano-modelo, clases confusas, posibles… | ⬜ pendiente |
| RF-LBL-06 |  | 4 | Importar/exportar etiquetas en COCO, YOLO, VOC, CSV, JSONL. | ⬜ pendiente |
| RF-WIZ-01 | sí | 3 | Pasos 1, 2, 3, 5, 6, 7, 8, 9 para tabular e imagen. | ⬜ pendiente |
| RF-WIZ-02 |  | 3 | Cada paso muestra "¿Por qué?" con la explicación del LLM y permite rechazar/editar. | ⬜ pendiente |
| RF-WIZ-03 |  | 3 | Sin LLM configurado (o privacidad L0) el wizard funciona con recomendaciones por… | ⬜ pendiente |
| RF-WIZ-04 |  | 3 | El estado del wizard es un documento ProjectDraft versionado. | ⬜ pendiente |
| RF-LLM-01 | sí | 2 | Interfaz única LLMProvider con adaptadores: Anthropic, OpenAI, Google Gemini, Kimi /… | ⬜ pendiente |
| RF-LLM-02 | sí | 2 | Capacidades declaradas por modelo: structured_output, tool_use, vision,… | ⬜ pendiente |
| RF-LLM-03 | sí | 2 | Perfiles LLM configurables: qué modelo se usa para cada *propósito* (copilot,… | ⬜ pendiente |
| RF-LLM-04 | sí | 2 | Salida estructurada: toda respuesta que alimenta al sistema se pide como JSON contra… | ⬜ pendiente |
| RF-LLM-05 |  | 2 | Streaming de respuestas al panel de copiloto. | ⬜ pendiente |
| RF-LLM-06 |  | 2 | Control de costos: presupuesto por proyecto/run/usuario, conteo de tokens, estimación… | ⬜ pendiente |
| RF-LLM-07 |  | 2 | Caché de respuestas por hash de (prompt, modelo, parámetros) para reproducibilidad y… | ⬜ pendiente |
| RF-LLM-08 |  | 2 | Claves API: en desktop, keychain del SO; en servidor, cifradas (AES-GCM, clave maestra… | ⬜ pendiente |
| RF-PRV-01 | sí | 2 | El PrivacyFilter se aplica en el Gateway (no en cada llamador). | ⬜ pendiente |
| RF-PRV-02 |  | 2 | Con un LLM local el Admin puede permitir L3 aunque la política general sea L1. | ⬜ pendiente |
| RF-PRV-03 | sí | 2 | Log de auditoría: el usuario puede ver exactamente qué payload se envió en cada… | ⬜ pendiente |
| RF-PRV-04 |  | 2 | Política de workspace: nivel máximo permitido y proveedores permitidos, definidos por… | ⬜ pendiente |
| RF-ARC-01 | sí | 2 | El LLM genera 2–4 propuestas usando exclusivamente bloques del catálogo (§8) en… | ⬜ pendiente |
| RF-ARC-02 | sí | 1/2 | Validación de cada propuesta: schema, compatibilidad de shapes (construcción en meta… | ✅ `archspec.validate` (6 etapas de §9.3) |
| RF-ARC-03 |  | 2 | Mini-torneo opcional: entrenar cada propuesta con un presupuesto corto (p. ej. 10 %… | ⬜ pendiente |
| RF-ARC-04 | sí | 1/2 | Fallback por reglas si el LLM no está disponible o falla la validación 3 veces. | ✅ `catalog.rules` |
| RF-ARC-05 |  | 3 | Editor visual de ArchSpec (React Flow): bloques del catálogo como nodos, parámetros en… | ⬜ pendiente |
| RF-ARC-06 |  | 3 | Modo experto — código libre: el LLM (o el usuario) escribe un… | ⬜ pendiente |
| RF-ARC-07 |  | 1/2 | Conversión ArchSpec → código PyTorch legible ("ver como código") para aprendizaje y… | ✅ `archspec.to_code` (equivalencia verificada con pesos) |
| RF-HPO-01 | sí | 1 | Estrategias soportadas: | ✅ single/random/grid/TPE/CMA-ES/NSGA-II + median/ASHA/Hyperband |
| RF-HPO-02 | sí | 1 | El LLM estratega recibe el escenario (tamaño de datos, costo por trial, presupuesto,… | 🟡 recomendación por reglas (ASHA ≥ 30 trials, si no mediana con calentamiento); estratega LLM en Capa 2 |
| RF-HPO-03 | sí | 1 | Presupuesto configurable en el wizard: tiempo total, n.º de trials, preset, métrica… | ✅ corte por trials, tiempo y métrica objetivo |
| RF-HPO-04 |  | 1 | Paralelismo de trials según recursos: varias GPUs → un trial por GPU; en servidor,… | ⬜ pendiente |
| RF-HPO-05 |  | 1 | Reanudación de estudios interrumpidos (almacenamiento Optuna en SQLite/PostgreSQL). | ✅ reanudación desde SQLite |
| RF-HPO-06 |  | 1 | Visualizaciones: historia de optimización, importancia de hiperparámetros, coordenadas… | ⬜ pendiente |
| RF-TRN-01 | sí | 1 | Detección de hardware al inicio y bajo demanda: GPUs (modelo, VRAM, capacidad de… | ✅ `training.hardware` + `GET /system/hardware` |
| RF-TRN-02 | sí | 1/3 | Instalación de PyTorch acorde al hardware: en el primer arranque (y desde… | 🟡 detección y recomendación de variante; instalación en Capa 3 |
| RF-TRN-03 | sí | 1 | Construcción del LightningModule desde ArchSpec + configuración de optimizador,… | ✅ `training.module` |
| RF-TRN-04 | sí | 1 | Buenas prácticas por defecto: mixed precision (bf16/fp16 según hardware), gradient… | 🟡 AMP, clipping, early stopping, checkpoints, semillas, workers y batch size automáticos; LR finder pendiente |
| RF-TRN-05 | sí | 1 | Cada run corre en un proceso separado; el Engine supervisa, captura OOM/crashes y los… | ✅ subproceso por run con diagnóstico de OOM/crash (ADR-0015) |
| RF-TRN-06 | sí | 1 | Progreso en vivo por WebSocket: época, batch, loss, métricas, LR, throughput, uso de… | ✅ eventos JSONL → EventBus → `WS /runs/{rid}/live` |
| RF-TRN-07 |  | 1 | Pausar, reanudar (desde checkpoint), cancelar. | ✅ cancelar, pausar y reanudar desde checkpoint |
| RF-TRN-08 |  | 1 | Multi-GPU en un nodo (DDP vía Lightning) cuando hay >1 GPU. | ⬜ pendiente |
| RF-TRN-09 |  | 1 | Técnicas de fine-tuning: congelar backbone, descongelado progresivo, LR… | 🟡 congelado del backbone (timm y encoders HF); LoRA/PEFT pendiente |
| RF-TRN-10 |  | 1 | Manejo de desbalance: pesos de clase, focal loss, sobremuestreo, umbral óptimo… | ✅ pesos de clase, focal, oversampling, umbral óptimo en el reporte |
| RF-TRN-11 |  | 1 | Caché de modelos preentrenados: descarga única, verificación de checksum, uso offline,… | ⬜ pendiente |
| RF-AGT-01 |  | 2 | Herramientas del agente (tool use): get_profile, get_project_goal,… | ⬜ pendiente |
| RF-AGT-02 |  | 2 | Límites duros aplicados por el sistema (no por el LLM): tiempo, n.º de iteraciones,… | ⬜ pendiente |
| RF-AGT-03 |  | 2 | Puntos de aprobación configurables: nunca / antes de cada iteración / solo si cambia… | ⬜ pendiente |
| RF-AGT-04 |  | 2 | Bitácora legible del agente ("Iteración 3: el modelo sobreajusta desde la época 12 →… | ⬜ pendiente |
| RF-AGT-05 |  | 2 | Si el LLM falla o no responde, el loop cae a una política por reglas o se detiene de… | ⬜ pendiente |
| RF-TRK-01 | sí | 1 | Todo run se registra en MLflow: parámetros, métricas por paso, artefactos… | ✅ MLflow embebido (SQLite + artefactos) |
| RF-TRK-02 | sí | 1 | La UI de Perceptron muestra runs y comparaciones de forma nativa (no depende de la UI… | 🟡 runs y comparación por API; vistas de UI en Capa 3 |
| RF-TRK-03 |  | 1 | Model Registry de MLflow para ModelVersion y stages. | ⬜ pendiente |
| RF-TRK-04 |  | 1 | Comparación de runs: tabla, curvas superpuestas, diff de configuración (ArchSpec,… | ⬜ pendiente |
| RF-EVL-01 | sí | 1 | Métricas por tarea: | ✅ clasificación, regresión, forecasting (MASE, backtesting, naive), anomalías, detección (mAP), segmentación (IoU/Dice), OCR (CER/WER); SED pendiente |
| RF-EVL-02 |  | 1/4 | Explicabilidad: SHAP (tabular, importancia global y local), Integrated Gradients /… | ⬜ pendiente |
| RF-EVL-03 |  | 1/4 | Análisis de errores: explorador de muestras mal predichas con filtros, slices… | ⬜ pendiente |
| RF-EVL-04 |  | 1/4 | Fairness: el usuario marca atributos sensibles; métricas por subgrupo (Fairlearn:… | ⬜ pendiente |
| RF-EVL-05 |  | 1/4 | Robustez: sensibilidad a ruido/perturbaciones por modalidad (ruido gaussiano, blur,… | ⬜ pendiente |
| RF-EVL-06 | sí | 2/4 | Informe final generado por el LLM (o plantilla sin LLM): resumen ejecutivo, qué se… | ⬜ pendiente |
| RF-EXP-01 | sí | 4 | Exportar a ONNX (con verificación numérica vs. PyTorch), torch.export… | ⬜ pendiente |
| RF-EXP-02 | sí | 4 | Playground en la app: cargar un archivo/fila/imagen/audio/texto, ver predicción,… | ⬜ pendiente |
| RF-EXP-03 |  | 4 | API REST de inferencia: generar y levantar un servidor FastAPI (ONNX Runtime o… | ⬜ pendiente |
| RF-EXP-04 |  | 4 | Proyecto de código exportable: repositorio Python standalone generado desde plantillas… | ⬜ pendiente |
| RF-EXP-05 |  | 4 | Firma del modelo: schema de entrada/salida, versión, hash; incluido en todos los formatos. | ⬜ pendiente |
| RF-MON-01 |  | 6 | Registro de predicciones del serving (muestreado, configurable, respetando privacidad)… | ⬜ pendiente |
| RF-MON-02 |  | 6 | Drift de datos: tabular con Evidently (PSI, KS, Jensen-Shannon, chi²) por feature; no… | ⬜ pendiente |
| RF-MON-03 |  | 6 | Drift de concepto / performance: métricas sobre datos etiquetados recientes vs. baseline. | ⬜ pendiente |
| RF-MON-04 |  | 6 | Alertas: en la app, email (SMTP) y webhook (Teams/Slack genérico). | ⬜ pendiente |
| RF-MON-05 |  | 6 | Políticas de reentrenamiento (RetrainPolicy): disparadores por drift, calendario… | ⬜ pendiente |
| RF-MON-06 |  | 6 | Champion/challenger: el nuevo modelo se evalúa contra el productivo en un holdout… | ⬜ pendiente |
| RF-MON-07 |  | 6 | Versionado de datasets: snapshots inmutables por manifiesto de hashes… | ⬜ pendiente |
| RF-SRV-01 |  | 5 | Autenticación: usuarios locales (hash Argon2) y SSO OIDC (Entra ID, Google Workspace,… | ⬜ pendiente |
| RF-SRV-02 |  | 5 | RBAC por workspace y proyecto (§3.2). | ⬜ pendiente |
| RF-SRV-03 |  | 5 | Sincronización desktop ↔ servidor para proyectos de equipo: metadata (PostgreSQL),… | ⬜ pendiente |
| RF-SRV-04 |  | 5 | Cola de jobs con prioridades y cuotas por usuario/workspace; los desktops pueden… | ⬜ pendiente |
| RF-SRV-05 |  | 5 | Modo estación de trabajo: la UI web del servidor ofrece la misma funcionalidad que el… | ⬜ pendiente |
| RF-SRV-06 |  | 5 | Consola de administración: usuarios, grupos, SSO, proveedores LLM y claves, políticas… | ⬜ pendiente |
| RF-SRV-07 |  | 5 | Auditoría: login, acceso a datasets, exportaciones, llamadas LLM, cambios de permisos. | ⬜ pendiente |
| RF-SRV-08 |  | 5 | Backups: guía y scripts para PostgreSQL y object storage. | ⬜ pendiente |
| RF-LIC-01 |  | 0 | Módulo licensing con interfaz LicenseProvider y una implementación DevLicenseProvider… | ✅ `LicenseProvider` + `DevLicenseProvider` |
| RF-LIC-02 |  | 0 | Todas las funciones premium consultan features.is_enabled("<feature_key>"); claves de… | ✅ `features.yaml` + `features.is_enabled()` |
| RF-LIC-03 |  | 7 | La arquitectura debe permitir más adelante: licencia por asiento, por servidor, por… | 🟡 arquitectura lista; sin enforcement en v1 (por diseño) |
