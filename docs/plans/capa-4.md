# Plan — Capa 4 (evaluación avanzada, export y serving)

SPEC §14: explicabilidad, análisis de errores, fairness y robustez (RF-EVL-02..05); informe PDF/HTML con marca y model card; export ONNX/torch.export/TorchScript con verificación; playground; API REST de inferencia + Dockerfiles; proyecto de código exportable; etiquetado asistido completo + active learning (§7.5); fuentes DB y datasets públicos (RF-ING-03, 04).

**Aceptación §14:**
- O5 en CI: el proyecto exportado entrena e infiere en un contenedor limpio con `uv sync`.
- ONNX coincide con PyTorch (tolerancia 1e-4; 1e-2 en fp16).
- En el fixture UC-04, etiquetar con pre-etiquetas reduce ≥ 50 % las acciones manuales.

Como en las capas anteriores, hay tres sub-capas con un PR cada una. El Engine se valida en CI; la UI, además, localmente.

## 4a — Export, serving y playground (RF-EXP-01..05)

- **Export** (`engine/perceptron/export/`):
  - **ONNX** (`torch.onnx.export` con dynamo, opset fijo). Se verifica contra PyTorch sobre un batch real de val con tolerancia 1e-4 (fp32) y 1e-2 (fp16), y el reporte de la verificación queda en el artefacto.
  - **torch.export** (ExportedProgram, `.pt2`).
  - **TorchScript**, marcado como *legacy*.
  - **Cuantización dinámica INT8** opcional (ONNX Runtime o `torch.ao`) para CPU.
  - **Firma del modelo** (RF-EXP-05) en todos los formatos: schema de entrada/salida, versión, hash y pipeline serializado.
  - **Runs de código experto:** el export ejecuta el modelo, así que corre dentro del sandbox (ADR-0025), con el mismo camino que la evaluación.
- **Paquete de inferencia:** `FittedPipeline` + modelo exportado. Un `perceptron-runtime` mínimo, sin Lightning ni Optuna, hace el preprocesamiento y la predicción con ONNX Runtime o PyTorch.
- **API REST de inferencia** (RF-EXP-03). Se genera un servidor FastAPI desde plantillas Jinja2 con:
  - `/predict` y `/predict/batch`, `/health`, `/metrics` (Prometheus) y OpenAPI;
  - API key por header;
  - `Dockerfile.cpu` y `Dockerfile.cuda`.

  Se puede levantar desde el desktop como proceso local o bajarlo como zip.
- **Proyecto de código exportable** (RF-EXP-04). Un repo standalone generado con Jinja2:
  - `pyproject.toml` (uv, con lock generado);
  - `src/` con el modelo (`archspec_to_code`), el pipeline, `train.py`, `infer.py` y `serve.py`;
  - `config.yaml`, README, tests mínimos y las licencias de los pesos preentrenados.

  **O5 en CI:** un job construye el proyecto exportado de UC-01 en un contenedor limpio (`python:3.12-slim` + uv) y corre `uv sync`, `train` e `infer`.
- **Playground** (RF-EXP-02). `POST /models/{id}/predict` recibe una fila, archivo, imagen, audio o texto y devuelve predicción, confianza y explicación local. La explicación usa SHAP/IG si ya está la 4b; si no, la importancia por perturbación. En la UI: pestaña Modelos → Probar.
- **API/CLI:**
  - `POST /models/{id}/export` como job (formatos, fp16, int8);
  - `GET /models/{id}/artifacts`;
  - `perceptron export`.

## 4b — Evaluación avanzada e informe (RF-EVL-02..06)

- **Explicabilidad** (RF-EVL-02):
  - SHAP (MIT), con KernelExplainer/DeepExplainer según el caso, para tabular: global y local;
  - Captum (BSD) para imagen, texto y series: Integrated Gradients y Grad-CAM, más atribución sobre espectrograma para audio.

  Los resultados se guardan como artefactos y los muestra la UI.
- **Análisis de errores** (RF-EVL-03):
  - explorador de mal predichos con filtros;
  - slices automáticos de bajo rendimiento, por columnas del ProfileCard (categóricas y bins de numéricas);
  - confusiones frecuentes;
  - posibles errores de etiqueta (confident learning propio, sin dependencia).
- **Fairness** (RF-EVL-04). Con Fairlearn (MIT), sobre los atributos sensibles que marca el usuario:
  - métricas por subgrupo;
  - demographic parity y equalized odds;
  - alertas de disparidad.
- **Robustez** (RF-EVL-05):
  - perturbaciones por modalidad: ruido gaussiano, blur y JPEG en imagen; ruido de fondo en audio; typos en texto; ruido en numéricas tabulares;
  - curva de degradación.
- **Informe PDF/HTML con marca** (RF-EVL-06):
  - se arma el HTML con la plantilla Preteco (Jinja2 + tokens y Titillium Web embebida);
  - el PDF se genera con WeasyPrint (BSD);
  - en Windows, si WeasyPrint no puede cargar Pango, se imprime el mismo HTML desde el webview del desktop;
  - se exporta también a Markdown;
  - incluye la model card completa.

  Decisión a documentar en un ADR: WeasyPrint frente a Chromium headless.

## 4c — Etiquetado asistido y fuentes (RF-LBL-01..06, RF-ING-03, 04)

- **Herramienta de etiquetado en la UI** (RF-LBL-01): clase por muestra, multi-etiqueta, cajas, polígonos/máscaras con pincel y segmentos temporales.
- **Pre-etiquetado** (RF-LBL-02):
  - con el modelo del proyecto;
  - con el LLM (ya existe);
  - zero-shot local con licencias verificadas (CLIP-like, NLI y CLAP-like), sujeto a la auditoría de licencias.
- **Active learning** (RF-LBL-03):
  - por incertidumbre y diversidad;
  - con el ciclo etiquetar → reentrenar → re-priorizar.
- **Calidad del etiquetado** (RF-LBL-05): acuerdo humano-modelo, clases confusas y posibles errores.
- **Import/export de etiquetas** (RF-LBL-06): COCO, YOLO, VOC, CSV y JSONL.
- **Aceptación UC-04:** script de simulación de anotador sobre el fixture. Cuenta las acciones con y sin pre-etiquetas; la reducción debe ser ≥ 50 %.
- **Fuentes** (RF-ING-03, 04):
  - SQL Server, PostgreSQL, MySQL/MariaDB y SQLite (SQLAlchemy + drivers con licencia OK), con query, vista previa, límite y lectura por chunks;
  - Hugging Face Datasets y Kaggle con credenciales del usuario en el keychain, con caché.

## Transversal

- **ADRs:**
  - 0027: formatos de export y paquete de inferencia;
  - 0028: informe PDF;
  - 0029: modelos zero-shot y licencias.
- **Dependencias nuevas**, en extras para no inflar el runtime del desktop:
  - `onnx` y `onnxruntime` (MIT);
  - `shap` (MIT), `captum` (BSD) y `fairlearn` (MIT);
  - `weasyprint` (BSD);
  - drivers de bases de datos (`psycopg` LGPL: usar dinámico; `pymysql` MIT; `pyodbc` MIT).
- **`progress.md`** con las filas RF-EVL, RF-EXP, RF-LBL y RF-ING.
