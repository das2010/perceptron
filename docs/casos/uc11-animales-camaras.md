# UC-11 · Identificación de animales en las cámaras del campo

Caso de prueba de punta a punta con imágenes reales. Sirve para mostrar Perceptron a un cliente y para
validar una versión nueva antes de publicarla.

## El problema

Un establecimiento rural tiene cámaras en los corrales, el gallinero y los accesos. Hoy una persona revisa
las fotos a mano para:

- contar cuántos animales de cada especie pasan por cada corral;
- enterarse cuando entra un **perro** o un **gato** al gallinero o a la zona de conejos;
- detectar fotos que no sirven: de noche la cámara pasa a visión nocturna y las fotos cambian mucho.

El objetivo es que un modelo identifique la especie de cada foto (8 especies) y que el sistema avise cuando
las fotos que llegan dejan de parecerse a las del entrenamiento.

## Los datos

Salen del zip «Dataset Of animal Images» (Roboflow Universe, **CC BY 4.0**). El script
[`scripts/casos/animales/preparar_caso.py`](../../scripts/casos/animales/preparar_caso.py) arma cuatro
conjuntos **disjuntos**, sin copias exactas y siempre iguales (semilla fija):

| Conjunto | Fotos | Para qué |
|---|---|---|
| `entrenamiento/<Especie>/` | 2.000 (250 × 8) | Dataset del proyecto: Cat, Cow, Deer, Dog, Goat, Hen, Rabbit, Sheep |
| `sin_etiquetar/` | 240 | Un lote nuevo sin etiquetas: etiquetado asistido |
| `produccion_dia/` | 200 | Fotos de día que llegan al modelo desplegado |
| `produccion_noche/` | 120 | Fotos con visión nocturna: el cambio de condiciones |
| `respuestas/*.csv` | — | Especie real de las fotos sin etiquetar y de producción (feedback y calificación) |

```bash
python scripts/casos/animales/preparar_caso.py "C:\Dataset Of animal Images.zip" fixtures/caso_animales
```

En el Team Server local, `fixtures/` se monta como fuente del servidor: los conjuntos quedan en
`/sources/caso_animales/…`.

## Recorrido en Perceptron

1. **Proyecto.** Nuevo proyecto, con el objetivo en lenguaje natural (el del apartado anterior).
2. **Datos.** *Fuentes del servidor* → `caso_animales/entrenamiento` → la vista previa muestra
   «2000 archivos en 8 clases» → *Crear versión de datos* → *Ver perfil*.
3. **Diseño.** *Entrenar* → *Proponer preparación* (224×224, normalización, augmentations) →
   *Proponer arquitecturas*: el arquitecto (LLM) propone 2–3 redes preentrenadas livianas y el sistema mide
   el tiempo por época en el equipo. Elegí una.
4. **Búsqueda de hiperparámetros.** 3 intentos × 6 épocas → *Recomendar estrategia* (LLM) → *Entrenar ahora*.
   En *Experimentos* se ven las curvas en vivo y el worker que entrena.
5. **Evaluación.** En el mejor run: *Evaluar* (test sellado), matriz de confusión, recall por especie,
   *Explicación* con una foto (mapa de calor sobre el animal), *Robustez* e *Informe*.
6. **Despliegue.** *Exportar* ONNX → *Registrar* → en *Modelos*, *Desplegar*.
7. **Producción de día.** Mandar las fotos de `produccion_dia/` al deployment (`POST
   /deployments/{id}/predict/file`) y luego la especie real como feedback (`/feedback`). *Monitoreo* →
   *Chequear ahora*: la performance con feedback y el drift de los embeddings (debe ser bajo).
8. **Producción de noche.** Mandar `produccion_noche/` → *Chequear ahora*: el drift de los embeddings
   sube a alto y aparece la alerta «Drift de datos».
9. **Etiquetado asistido.** Nueva versión de datos con `caso_animales/sin_etiquetar` → *Etiquetado* →
   conjunto de tipo *Clase* con las 8 especies → *Zero-shot local* (SigLIP, sin entrenar) → revisar la cola
   (las más dudosas primero) → *Aceptar sugerencias confiables* → *Aplicar etiquetas*.
10. **Costo.** *Auditoría LLM*: todas las llamadas, con el payload filtrado (privacidad L1) y su costo.

El recorrido completo por la API lo automatiza
[`scripts/casos/animales/validar_caso.py`](../../scripts/casos/animales/validar_caso.py), que además
califica los criterios:

```bash
set PERCEPTRON_EMAIL=usuario@empresa.com
set PERCEPTRON_PASSWORD=...
python scripts/casos/animales/validar_caso.py --url http://localhost:8080 --caso fixtures/caso_animales
```

## Criterios de aceptación

| Criterio | Objetivo |
|---|---|
| Accuracy en el test sellado | ≥ 0,85 |
| Recall de la peor especie en test | ≥ 0,70 |
| Accuracy con las fotos de producción de día | ≥ 0,85 |
| Acierto del zero-shot sobre el lote sin etiquetar | ≥ 0,85 |
| Drift con fotos de día | bajo o ninguno |
| Drift con fotos de noche | medio o alto, con alerta |
| Gasto del LLM en todo el caso | ≤ USD 1 |

## Resultado de referencia

Corrida del 29/09/2026 en un Team Server local **solo con CPU** (8 núcleos, 7,7 GB), LLM gpt-5,
privacidad L1. El arquitecto propuso EfficientNet-B0 preentrenada (con descongelado progresivo) y la
estrategia *single* (con este presupuesto no conviene buscar hiperparámetros); 6 épocas.

| Criterio | Resultado | Objetivo |
|---|---|---|
| Accuracy en el test sellado (300 fotos) | **0,94** (F1 macro 0,94, ROC-AUC 0,99) | ≥ 0,85 |
| Recall de la peor especie (Dog) | **0,87** | ≥ 0,70 |
| Accuracy en producción de día (200 fotos) | **0,95** | ≥ 0,85 |
| Drift con fotos de día | **ninguno** (AUC de dominio 0,46) | bajo o ninguno |
| Drift con fotos de noche | **alto** (AUC de dominio 0,93) y alerta | medio o alto |
| Acierto del zero-shot (SigLIP, 200 fotos) | **0,91** | ≥ 0,85 |
| Gasto del LLM | **USD 0,15** (10 llamadas) | ≤ USD 1 |

Tiempos en CPU: datos y perfil 50 s, propuestas del LLM 81 s, entrenamiento 40 min, evaluación 28 s,
zero-shot de 200 fotos 3 min (la primera vez baja el modelo, ~800 MB).

Con 3 épocas el modelo queda en 0,82 y no aprueba: el presupuesto del caso es 6 épocas
(`validar_caso.py --epocas 6`, el valor por defecto).

## Qué mirar en la demo

- **Confusiones esperables:** cabra ↔ oveja y ciervo ↔ cabra, por el pelaje y el entorno parecidos.
  La matriz de confusión y los mapas de calor muestran si el modelo mira al animal o al fondo.
- **El drift no necesita etiquetas:** la alerta de noche aparece antes de tener feedback, porque se mide
  sobre los embeddings internos del modelo.
- **El zero-shot ahorra etiquetado:** con los nombres de las especies alcanza para pre-etiquetar el lote
  nuevo. Con *Aceptar sugerencias confiables* (confianza ≥ 0,9) se aceptaron 150 de 200 fotos con 98 % de
  acierto; la persona revisa solo las 50 dudosas, que la cola muestra primero.
- **Lo que sigue:** etiquetar fotos nocturnas (zero-shot + revisión) y reentrenar. El challenger se compara
  con el champion sobre datos que ninguno vio y se promueve solo si mejora.
