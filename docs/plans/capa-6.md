# Plan — Capa 6 (MLOps)

SPEC §14:
- serving con logging de predicciones y feedback;
- drift de datos y de embeddings;
- drift de performance;
- alertas (app, email, webhook);
- `RetrainPolicy` con disparadores (drift, cron, volumen, degradación), incluidas fuentes API/streaming (RF-ING-05);
- champion/challenger con rollback;
- versionado de datasets con linaje y diff.

**Aceptación §14 (UC-10):**
- al inyectar drift sintético en el stream del fixture, se genera una alerta;
- se reentrena;
- el challenger se evalúa y se promueve solo si mejora;
- el rollback funciona.

Son dos sub-capas con un PR cada una. Todo corre en el Engine (desktop y servidor); en el Team Server, las tareas pesadas van a la cola de la 5b.

## Punto de partida (relevado en el código)
- **Serving:** `serving/runtime.InferenceModel` (ONNX Runtime) sobre `<run>/export/`. El playground predice **por run**, sin registro, y el bundle de serving no tiene log ni feedback.
- **Modelos:** `ModelVersion.stage` existe (candidate/staging/production/archived), pero nada lo cambia: no hay champion, ni promoción, ni rollback.
- **Entidades:** `Deployment`, `DriftReport` y `RetrainPolicy` existen en el dominio sin uso. `DataSourceType.API/STREAM` también.
- **Diff de datasets:** solo `Manifest.diff()` por archivo. `GET /datasets/{a}/diff/{b}` (SPEC §10) no está implementado.
- **Evaluación:** `evaluate_run(run_dir, dataset_dir, split=…)` acepta cualquier dataset, pero sobrescribe `evaluation.json` y la entidad `Evaluation` no registra sobre qué versión se evaluó.
- **Fixture UC-10:** no tiene etiqueta ni las columnas `edad`/`region` de UC-01. Hay que extender el generador, con la misma función de etiquetado de UC-01 y el drift en los lotes ≥ 7. Se regenera desde CI, porque el Python local está bloqueado.
- **Dependencias:** sin Evidently, sin scheduler y sin scipy declarado (llega por scikit-learn).

## 6a — Serving monitoreado, champion/challenger, drift y alertas
- **Deployments** (RF-MON-01, §10 `POST /models/{mv}/deployments`):
  - endpoint de predicción servido por el propio Engine para una `ModelVersion`;
  - **log de predicciones** muestreado (`sample_rate`), en Parquet por día bajo `projects/<id>/monitoring/<dep>/`. Con L1/L2 no se guardan las columnas marcadas como PII: se aplica la misma política de privacidad;
  - `POST /deployments/{id}/feedback` para las etiquetas reales, por `prediction_id` o por clave;
  - el bundle exportado de serving suma el log a archivo y `/feedback`, con el mismo formato.
- **Champion/challenger** (RF-MON-06):
  - `POST /models/{mv}/promote` → production; el anterior pasa a archived y queda como rollback;
  - `POST /projects/{id}/models/rollback`;
  - `POST /models/{challenger}/challenge` evalúa challenger y champion sobre el mismo holdout reciente (feedback etiquetado o una versión de datos) con el criterio configurado (métrica, mejora mínima), y promueve solo si mejora;
  - la evaluación sobre otra versión de datos se guarda aparte y `Evaluation` suma `dataset_version_id`;
  - el deployment sigue siempre al champion.
- **Drift de datos** (RF-MON-02), ADR: **métricas propias con scipy/numpy en lugar de Evidently**:
  - **Motivo:** Evidently suma muchas dependencias (plotly, nltk, etc.) a un producto que se distribuye en desktop; las métricas son pocas y estándar, y así se mantiene el funcionamiento offline.
  - **Tabular:** PSI, KS y Jensen-Shannon para numéricas; χ² y JS para categóricas; con umbrales por severidad.
  - **No estructurado:** drift de embeddings del modelo (penúltima capa vía ONNX) con MMD (kernel RBF), distancia de centroides y un clasificador de dominio (AUC).
  - **Referencia:** el split de entrenamiento de la versión con la que se entrenó el champion.
  - Cada ventana genera un `DriftReport` por deployment, por ventana de tiempo o de N predicciones.
- **Drift de performance** (RF-MON-03): métricas sobre el feedback etiquetado reciente contra las del test del champion.
- **Alertas** (RF-MON-04):
  - entidad `Alert` (en la app, con estado leída o resuelta);
  - canales: email por SMTP (stdlib) y webhook genérico, JSON compatible con Teams/Slack;
  - la configuración de canales va por proyecto y las credenciales SMTP en el almacén de secretos;
  - las alertas llegan a la UI por WS.
- **Versionado de datasets** (RF-MON-07):
  - `GET /datasets/{a}/diff/{b}`: filas agregadas y eliminadas por hash de fila, cambios de esquema, cambios de distribución (con las mismas métricas de drift) y archivos (Manifest);
  - `GET /datasets/{id}/lineage`: grafo de padres y transformaciones;
  - retención configurable;
  - export opcional a formato DVC (`.dvc` con los hashes).
- **UI:**
  - pestaña **Monitoreo** del proyecto: deployments, predicciones y feedback, drift por feature con severidad, alertas;
  - en Modelos: champion, challengers y los botones de promover y rollback;
  - diff entre versiones de datos.

## 6b — Reentrenamiento automático y fuentes streaming
- **Fuentes** (RF-ING-05):
  - interfaz `StreamSource`, extensible;
  - REST con paginación (offset/cursor/link) y auth por header o token (el secreto va al keychain), con mapeo de JSON a tabla;
  - WebSocket;
  - MQTT (paho-mqtt, EPL/EDL: revisar la licencia) y Kafka (kafka-python, Apache-2.0) como extras opcionales;
  - los datos se acumulan en un **buffer** del proyecto, con Parquet por lote y conteo de filas nuevas.
- **RetrainPolicy** (RF-MON-05):
  - disparadores:
    - drift con severidad ≥ X;
    - cron (croniter, MIT, con un scheduler propio en el Engine);
    - volumen (N filas nuevas en el buffer o con feedback);
    - degradación de métrica;
  - acción:
    - una **versión de datos nueva** = la anterior + las filas nuevas etiquetadas, con `parent_id` y `transformation=append:<lote>`. El test del champion queda fijo como holdout comparable (split "congelado" para las filas viejas);
    - después, reentrenar con la última ArchSpec y HPO reducido, o relanzar el agente autónomo;
    - después, champion/challenger;
    - por último, la promoción: automática o con aprobación (`require_approval`), según la política;
  - presupuesto por política;
  - cada ejecución queda registrada (`RetrainRun`) con su bitácora.
- **Aceptación UC-10** (test de API + e2e CLI):
  - UC-01 entrenado y desplegado;
  - el stream del fixture entra por lotes y el feedback llega con etiquetas;
  - en los lotes con drift: alerta → reentrenamiento → challenger evaluado → promovido solo si mejora;
  - rollback al anterior.

## Pendientes y riesgos
- **Licencia de paho-mqtt (EPL-2.0/EDL-1.0):** se evalúa en el ADR. Si no conviene, MQTT queda fuera del empaquetado base.
- **Fixture:** hay que regenerar `uc10_stream` desde un workflow manual, sin tocar los demás fixtures (cada generador tiene su propia semilla).
