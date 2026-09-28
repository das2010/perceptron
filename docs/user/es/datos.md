# Datos

Todo empieza en la pestaña **Datos** del proyecto: cargás una fuente, elegís la columna objetivo y
creás una **versión de datos**. Perceptron la perfila, te avisa de problemas de calidad y la parte en
entrenamiento, validación y test.

## Fuentes

### Archivos y carpetas

En **Agregar datos**:

- **Subir archivo:** CSV, TSV, Excel (`.xlsx`), Parquet, JSON / JSON Lines o un ZIP.
- **Subir carpeta:** una carpeta con **una subcarpeta por clase**, por ejemplo
  `defectos/ok/*.png` y `defectos/falla/*.png`. Sirve para imágenes, audio (WAV, FLAC, MP3, OGG) y
  texto (`.txt`).

Perceptron reconoce el tipo de fuente y lo muestra en la vista previa:

| Tipo | Ejemplo |
|---|---|
| Tabla | CSV o Excel con una fila por caso |
| Carpeta de imágenes | Subcarpetas por clase, o imágenes con anotaciones COCO, Pascal VOC o YOLO (detección) |
| Carpeta de textos | Subcarpetas por clase con archivos `.txt` (o un CSV con una columna de texto) |
| Carpeta de audio | Subcarpetas por clase con archivos de audio |
| Imágenes con máscaras | Imágenes y máscaras PNG (segmentación) |
| Imágenes con texto | Imágenes con un CSV de transcripciones (OCR) |

El CSV de eventos de audio (inicio, fin, etiqueta) llega **próximamente**.

En la **Vista previa** revisás las primeras filas y el tipo inferido de cada columna (Numérica,
Categórica, Sí/No, Fecha, Texto, Identificador, Archivo). En **Columna objetivo** elegís lo que el modelo
tiene que predecir; si la dejás en «Detectar automáticamente», se infiere. Después tocá **Crear versión
de datos**.

### Bases de datos

**Desde una base de datos** conecta con **PostgreSQL**, **MySQL / MariaDB**, **SQL Server** o **SQLite
(archivo)**:

1. Elegí el **Motor** y completá **Servidor**, **Puerto**, **Base**, **Usuario** y **Contraseña**.
2. Escribí la **Consulta SQL**. Es de solo lectura; se lee por lotes y el resultado se guarda como una
   copia dentro del proyecto.
3. Tocá **Ejecutar y previsualizar** y seguí como con un archivo.

La contraseña se guarda en el llavero del sistema, nunca en el proyecto.

### Datasets públicos (Hugging Face / Kaggle)

**Dataset público (Hugging Face / Kaggle)**: elegí el **Origen**, escribí el nombre del **Dataset** (y
el **Split** en Hugging Face) y tocá **Descargar y previsualizar**. El **Token de Hugging Face** es
opcional; para Kaggle necesitás tus **Credenciales de Kaggle (usuario:clave)**. Las credenciales van al
llavero del sistema.

### Fuentes del servidor (Team Server)

En la UI web del Team Server no hay acceso a las carpetas de tu equipo: subís archivos o usás las
**Fuentes del servidor**, carpetas que el Admin monta en el servidor. Navegás la **Ruta** y tocás
**Usar**. Ver [Team Server](team-server.md).

### APIs y streaming

Para alimentar el [reentrenamiento automático](monitoreo.md#reentrenamiento-automatico), un proyecto
puede tener fuentes que acumulan datos nuevos:

- **API REST** con paginación (offset, página, cursor o enlace «next») y autenticación por header o
  token; el JSON se aplana a tabla.
- **WebSocket** con mensajes JSON.
- **Archivo JSON Lines que crece** (por ejemplo, un log): se lee desde el último punto.

Los datos nuevos se acumulan en un buffer y se consultan cada cierto intervalo. Por ahora estas fuentes
se configuran desde la API del Engine; la pantalla para crearlas llega **próximamente**. **Kafka** y
**MQTT** también llegan **próximamente**.

## Versiones de datos

Cada carga crea una **versión de datos** inmutable, identificada por el hash de su contenido. En la
tabla **Versiones de datos** ves la versión, la modalidad, la cantidad de muestras, el objetivo y la
fecha; **Ver perfil** abre su perfil.

- Si los datos cambian, se crea otra versión. Los entrenamientos y modelos anteriores siguen apuntando
  a la suya, así que siempre se puede reproducir un resultado.
- Cada versión guarda su **linaje** (de qué versión viene y con qué transformación, por ejemplo al
  aplicar etiquetas o al sumar filas nuevas para reentrenar).
- Se pueden **comparar dos versiones**: filas agregadas y eliminadas, cambios de esquema, cambios de
  distribución por columna y archivos distintos. Por ahora, desde la API del Engine; la vista en la UI
  llega **próximamente**.

## Particiones y test sellado

Al crear la versión, los datos se parten en **entrenamiento**, **validación** y **test**:

- Por defecto, aleatoria **estratificada** (mantiene la proporción de clases).
- Si el objetivo es numérico y hay una columna de fecha, **temporal** (el test es lo más reciente).
- Las series temporales se parten por tiempo dentro de cada serie.
- Desde la CLI o la API podés pedir partición **por grupo** (para que un mismo cliente no quede en
  train y en test a la vez) o **k-fold**.

!!! important "El test está sellado"
    El test **no se usa** para elegir la arquitectura, los hiperparámetros ni el mejor intento; tampoco
    lo ve el agente autónomo. Solo se abre cuando tocás **Evaluar en test** (o cuando el agente cierra),
    así la métrica final es una estimación honesta.

## Perfil y alertas de calidad

El **Perfil del dataset** resume:

- **Particiones:** tamaño de cada una.
- **Distribución de clases** y la proporción minoritaria/mayoritaria.
- **Columnas:** tipo, nulos, valores distintos y un resumen (media y desvío para numéricas).
- Según la modalidad: resolución, canales, archivos corruptos y casi duplicados (imágenes); idioma,
  largo y duplicados (texto); frecuencia, huecos, estacionalidad y tendencia (series); duración, sample
  rate, silencio y clipping (audio).

Las **Alertas de calidad** tienen tres severidades: **Info**, **Atención** e **Importante**. Las más
comunes:

| Alerta | Qué significa | Qué hacer |
|---|---|---|
| Desbalance | Una clase tiene muchos menos casos que otra | El entrenamiento puede compensarlo (pesos de clase, sobremuestreo); revisá métricas como recall o F1, no solo accuracy |
| Posible fuga (leakage) | Una columna es casi igual al objetivo, es un identificador o trae fechas futuras | Sacala: el modelo parecería perfecto y fallaría en producción |
| Columna constante | Tiene un solo valor | Se descarta en la preparación |
| Datos insuficientes | Hay pocos casos para la tarea | Conseguí más datos o bajá las expectativas |

Este perfil es, además, lo que ve el LLM en el nivel de privacidad L1: solo estadísticas agregadas,
nunca valores individuales. Ver [LLM y privacidad](llm-y-privacidad.md).

## Preparación (pipeline)

**Proponer preparación** (en **Entrenar** o en el wizard) arma un pipeline según el perfil:
imputación, codificación de categorías, escalado, tokenización, redimensionado de imágenes,
espectrogramas, ventanas de series, etc., con una justificación por paso. El pipeline se ajusta solo con
entrenamiento y viaja junto al modelo cuando lo exportás.

En **Diseño → Pipelines de datos** podés abrirlo en el editor visual: **Agregar paso**, **Subir** /
**Bajar**, quitar, cambiar **Columnas** y **Parámetros (JSON)**, y ver la **Vista previa sobre train**
después de cada paso (**Ver resultado tras «paso»**) antes de **Guardar**.

## Etiquetado asistido

Si tus datos no tienen etiquetas (o tienen pocas), usá la pestaña **Etiquetado**:

1. Elegí la **Versión de datos**, el **Tipo** (**Clase**, **Multi-etiqueta** o **Cajas**) y las
   **Clases** separadas por coma. Tocá **Nuevo conjunto**.
2. En la **Cola de revisión** etiquetás muestra por muestra. Elegí el **Orden de la cola**: **Más dudosas
   primero** (active learning), **Diversas** o **Al azar**. Atajos: **Aceptar (Enter)**, **Saltar**,
   **Deshacer**.
3. Cuando ya entrenaste un primer modelo, elegilo en **Modelo para pre-etiquetar** y tocá
   **Pre-etiquetar**: cada muestra trae una sugerencia con su confianza.
4. Con **Confianza mínima** y **Aceptar sugerencias confiables** aceptás en lote las sugerencias seguras
   y revisás a mano solo las dudosas.
5. **Aplicar etiquetas (nueva versión)** crea una versión de datos nueva con las etiquetas. Desde ahí,
   **Reentrenar** y repetir el ciclo.

También podés exportar las etiquetas a CSV o JSONL (y COCO, YOLO o VOC para cajas). Perceptron informa
el acuerdo entre personas y modelo y señala posibles errores de etiqueta.

El pre-etiquetado con el LLM (para texto, con privacidad L2 o L3) y la guía de etiquetado generada por
el LLM están disponibles desde la API. Las máscaras con pincel, los segmentos temporales de audio y
series y el pre-etiquetado con modelos zero-shot locales llegan **próximamente**.
