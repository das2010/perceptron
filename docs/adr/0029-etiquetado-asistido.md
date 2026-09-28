# ADR-0029: Etiquetado asistido y active learning (Capa 4c)
- Estado: aceptado
- Fecha: 2026-09-28
- Contexto: §7.5 pide:
  - herramienta de etiquetado;
  - pre-etiquetado (modelo del proyecto, LLM, zero-shot local);
  - active learning, calidad del etiquetado e import/export en formatos estándar.

  La aceptación de §14 exige que en UC-04 las pre-etiquetas reduzcan al menos un 50 % las acciones manuales. Las versiones de dataset son inmutables (ADR-0016) y el test está sellado para el entrenamiento y la selección de modelos.
- Decisión:
  - **Modelo de datos.**
    - Un `LabelSet` anota una versión de dataset. Cada muestra tiene un id estable, `row:<i>`: su posición en la tabla de la versión, que no cambia porque la versión es inmutable.
    - Los ítems viven en `labels/<id>.jsonl`: etiqueta o cajas normalizadas, origen (humano, modelo o LLM), confianza, estado (sugerida o aceptada) y fecha.
    - Al crear el conjunto, las etiquetas que ya trae el dataset se siembran como humanas aceptadas.
  - **Aplicar etiquetas** crea una versión nueva (`parent_id`, `transformation = labels:<id>`) a través de la ingesta normal:
    - tabla con la columna objetivo;
    - carpetas por clase (imagen y audio);
    - COCO (cajas).

    Así la versión nueva tiene su propio perfil, sus splits y su test sellado: el ciclo es etiquetar → aplicar → reentrenar → pre-etiquetar de nuevo.
  - **Lectura del test.** Hay un propósito de lectura nuevo, `Purpose.LABELING`, que puede leer el test: la persona anota todas las muestras y eso no entrena ni elige modelos. Las sugerencias del modelo sobre muestras de test son solo sugerencias. El origen de cada etiqueta queda registrado para auditar un posible sesgo.
  - **Pre-etiquetado:**
    - con el modelo de un run del proyecto (clase) o con el LLM (texto, con la guía de etiquetado y las mismas reglas de privacidad L2/L3 de la Capa 2);
    - las opiniones del modelo se guardan aparte (`.model.jsonl`) para medir después el acuerdo humano-modelo;
    - los modelos zero-shot locales (CLIP/NLI/CLAP) quedan pendientes de la auditoría de licencias y de la caché de pesos offline.
  - **Active learning:** cola por incertidumbre (menor confianza de la sugerencia), diversidad (ronda entre las clases sugeridas) o al azar, con semilla fija. La aceptación en lote de las sugerencias con confianza ≥ umbral cuenta como una sola acción.
  - **Calidad:** acuerdo humano-modelo sobre las etiquetas humanas que tenían opinión previa del modelo, pares confusos y posibles errores (el modelo discrepa con confianza ≥ 0,9).
  - **Import/export:** CSV y JSONL (clase y multi-etiqueta, por `sample_id` o clave), y COCO, YOLO y VOC (cajas). Los XML con DTD o entidades se rechazan, para evitar la expansión de entidades.
  - **Aceptación UC-04** (test de API): se ocultan la mitad de las etiquetas y se entrena con el resto. El modelo pre-etiqueta, y un anotador simulado acepta en lote lo confiado (≥ 0,8) y corrige a mano lo demás. Se mide la reducción de acciones frente al etiquetado manual (una acción por muestra) y la exactitud final.
- Consecuencias:
  - Los ítems en JSONL alcanzan para decenas de miles de muestras. Con más volumen conviene migrar a Parquet o SQLite sin cambiar la API.
  - Las máscaras por pincel y los segmentos temporales tienen modelo (`LabelKind`) pero no herramienta en la UI todavía.
- Alternativas consideradas:
  - Guardar las etiquetas dentro de la versión: rompe la inmutabilidad.
  - Label Studio embebido: otra app, con otro modelo de datos y otra licencia.
  - Contar como acción cada aceptación individual: mide peor el aporte de las pre-etiquetas en la revisión real, que se hace por lote.
