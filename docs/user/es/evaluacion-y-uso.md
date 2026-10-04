# Evaluación y uso del modelo

Todo esto está en la página de un run (**Experimentos** → tocá el run). El orden natural es: evaluar en
test → analizar → registrar → exportar → probar.

## Evaluación en el test sellado

**Evaluar en test** calcula las métricas sobre el test sellado, que no participó en ninguna decisión del
entrenamiento. Es la estimación honesta de cómo va a rendir el modelo con datos nuevos.

| Tarea | Métricas principales |
|---|---|
| Clasificación | accuracy, precisión / recall / F1 (por clase, macro y micro), ROC-AUC, PR-AUC, calibración, umbral óptimo, **Matriz de confusión** |
| Regresión | MAE, RMSE, MAPE / sMAPE, R², residuos |
| Pronóstico | MAE, RMSE, sMAPE, MASE (comparado con un pronóstico ingenuo), por horizonte y por serie |
| Anomalías | precisión / recall / F1, ROC-AUC, distribución de scores |
| Detección de objetos | mAP@.5 y mAP@[.5:.95], por clase |
| Segmentación | IoU y Dice por clase |
| OCR | CER y WER (tasa de error por carácter y por palabra) |
| Audio | métricas de clasificación |

Cómo leerlas, en corto:

- **Clases desbalanceadas** (como churn): mirá recall, F1 o ROC-AUC, no solo accuracy. Un modelo que
  dice siempre «no se va» puede tener 90 % de accuracy y no servir.
- **MASE < 1** en pronóstico significa que el modelo le gana al pronóstico ingenuo.
- **La matriz de confusión** muestra qué clase se confunde con cuál (filas: real; columnas: predicho).

La detección de eventos sonoros (con inicio y fin) llega **próximamente**.

## Evaluación avanzada

Después de evaluar aparece **Evaluación avanzada**, con cinco pestañas. Todo se calcula sobre el test
sellado ya evaluado, salvo la explicación global, que usa validación.

### Errores

- Resumen: cuántos errores sobre cuántos casos.
- **Dónde rinde peor:** segmentos de datos (columna = valor) con peor rendimiento.
- **Confusiones más frecuentes:** por ejemplo, «“premium” predicho como “básico”: 12».
- **Casos mal predichos**, con la confianza del modelo. Los marcados con «La etiqueta podría estar mal»
  son candidatos a error de etiqueta: revisalos antes de culpar al modelo.

### Explicación

**Calcular** estima qué variables pesan más en las predicciones (valores de Shapley por muestreo sobre
validación). En imágenes, un mapa de calor muestra qué zonas miró el modelo. Para texto, audio y series
la explicación llega **próximamente**.

### Equidad

Marcá los atributos sensibles (por ejemplo, región o edad), elegí la **Clase positiva** y tocá
**Comparar grupos**. Ves, por **Grupo**, la cantidad de casos, la tasa de la clase positiva y las
métricas del modelo, con indicadores de paridad demográfica e igualdad de oportunidades. Si la diferencia
entre grupos supera el umbral, se marca como alerta.

### Robustez

**Probar robustez** mide cuánto empeora el modelo con perturbaciones de severidad creciente: ruido,
categorías cambiadas y valores faltantes (tabular); ruido, desenfoque y compresión JPEG (imagen). Para
texto y audio llega **próximamente**.

### Informe

**Generar informe** (o la pestaña **Informe**) produce un informe con resumen ejecutivo, qué se probó,
por qué ganó el modelo, métricas, limitaciones, riesgos y recomendaciones, más una **model card**. Si
hay LLM lo redacta el LLM; si no, una plantilla. Incluye lo que ya calculaste (explicación, equidad,
robustez). Se descarga en **PDF**, **HTML** o **Markdown**, con la marca Preteco.

## Registrar el modelo

**Registrar modelo** lo agrega a la pestaña **Modelos** con sus métricas en test. Desde ahí se despliega
y se gestiona el champion. Ver [Monitoreo](monitoreo.md).

## Exportar

En **Exportar el modelo** tildás los formatos y tocás **Exportar**:

| Formato | Para qué |
|---|---|
| **ONNX** | El más portable: ONNX Runtime en Python, C#, Java, JavaScript, etc. Opciones **ONNX fp16** (más chico, para GPU) y **ONNX INT8 (CPU)** (cuantizado) |
| **torch.export** | Programa exportado de PyTorch, para seguir en el ecosistema PyTorch |
| **TorchScript** | Formato heredado; usalo solo si tu entorno lo exige |

Cada formato se **verifica** contra PyTorch sobre datos reales de validación: la columna **Verificación**
muestra la diferencia máxima y la tolerancia (ONNX: 1e-4; fp16: 1e-2). Si falla, se marca «falló». Todos
los formatos incluyen la **firma** del modelo: entradas, salidas, versión y hash. El preprocesamiento
(pipeline) viaja con el modelo.

### Servidor de inferencia (Docker)

**Servidor de inferencia (Docker)** (requiere un ONNX verificado) descarga un ZIP con una API REST lista
para desplegar:

- `Dockerfile.cpu` y `Dockerfile.cuda`, o ejecución sin Docker con `uv`.
- Endpoints: `/predict` (JSON), `/predict/batch` (CSV o imagen), `/signature`, `/metrics` (Prometheus)
  y `/docs` (OpenAPI).
- Autenticación con API key: se define con la variable `PERCEPTRON_API_KEY` y se envía en el header
  `X-API-Key`.

El `README.md` del ZIP trae los comandos exactos.

### Proyecto de código

**Proyecto de código** descarga un repositorio Python autónomo (no necesita Perceptron): el modelo
como código PyTorch generado desde la ArchSpec, el pipeline, `config.yaml` con los hiperparámetros del
mejor intento, los pesos entrenados, datos de train y validación, y comandos para reentrenar, predecir,
servir y una prueba de humo. Se instala con `uv sync`. El test sellado no se incluye.

Los detalles de licencias de pesos preentrenados quedan en `LICENSES.md` dentro del proyecto.

## Playground

Cuando hay un ONNX verificado aparece **Probar el modelo**:

- **Tabular:** un formulario con una entrada por columna. **Completar con un ejemplo** carga una fila
  real; cambiá valores y tocá **Predecir**.
- **Imagen:** elegí la **Imagen a clasificar**.

Ves la **Predicción**, la **confianza** y las probabilidades. **Explicar esta predicción** muestra la
**Contribución de cada variable** (tabular) o el **Mapa de calor de la explicación** (imagen). El
playground para audio y texto llega **próximamente**.

## Fórmula sugerida

En **Experimentos**, para regresión con columnas numéricas, **Buscar fórmula** intenta encontrar
una fórmula matemática que explique el objetivo (por ejemplo `multiplo = 3·numero` o
`Salida = Sensor1·(√Sensor2 + 1)`). Es una referencia junto a las redes, no un modelo que se
exporta o despliega.

- Elegí el **Tiempo de búsqueda** (30 s a 10 min) y esperá el resultado.
- Ves la fórmula, su error en validación y en test comparado con la **Mejor red**, y un gráfico
  de valor real vs. fórmula.
- **Calcular** evalúa la fórmula con valores nuevos, también fuera del rango de los datos: una
  fórmula que captura la regla extrapola, una red no.
- **Copiar como Python** o **Copiar como Excel** (las columnas van en A2, B2…).

Si la fórmula avisa que **hay entradas que varían juntas**, muchas fórmulas explican igual esos
datos: para inferir la regla real hacen falta datos donde cada columna varíe por separado.
