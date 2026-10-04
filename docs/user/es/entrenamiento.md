# Entrenamiento

Hay tres formas de entrenar, de más guiada a más automática:

| Camino | Dónde | Cuándo conviene |
|---|---|---|
| **Wizard** | Pestaña **Wizard** | Primera vez con un problema; querés entender cada decisión |
| **Entrenar** | Pestaña **Entrenar** | Ya sabés lo que querés: preparación → arquitectura → búsqueda → **Entrenar ahora** |
| **Agente** | Pestaña **Agente** | Querés que el LLM itere solo dentro de un presupuesto |

## Wizard y copiloto

El wizard tiene nueve pasos: **Objetivo → Datos → Calidad → Etiquetado → Tarea y métrica →
Arquitectura → Búsqueda de hiperparámetros → Presupuesto y hardware → Revisión y lanzamiento**. Avanzás
con **Siguiente** y **Anterior**. El estado se guarda solo: podés cerrar y seguir otro día.

- **Objetivo:** «¿Qué querés lograr?» Contalo como se lo contarías a un colega.
- **Tarea y métrica:** tipo de problema, **Métrica a optimizar** (si no elegís, «Automática
  (val_loss)») y un **Umbral de éxito** opcional.
- **Presupuesto y hardware:** **Tiempo máximo (minutos)**, **Dispositivo** («El recomendado» o uno
  concreto) y **Modo**: **Guiado (vos decidís)** o **Agente autónomo**.
- **Revisión y lanzamiento:** resume todo antes de entrenar.

Cada paso tiene un botón **¿Por qué?** que le pregunta al copiloto por qué conviene lo recomendado con
tus datos.

**El copiloto** es el panel de la derecha. Le escribís en lenguaje natural («quiero priorizar no perder
clientes premium») y responde en streaming. Cuando propone cambios al borrador (objetivo, columna
objetivo, tarea, métrica, umbral, intentos, épocas, tiempo, modo autónomo), aparecen en **Cambios
sugeridos al borrador** con **Aceptar** / **Descartar**.

Sin LLM configurado, o con privacidad L0, el copiloto no está disponible y el wizard sigue funcionando
con recomendaciones por reglas.

### Ficha del caso y plan adaptado

En **Objetivo** está la **Ficha del caso**: qué tipo de problema es, qué se predice, qué error es
peor (y cuánto), si hay fechas o entidades repetidas, si las entradas varían por separado, si vas a
predecir fuera del rango de los datos y dónde se va a usar el resultado. Podés completarla a mano o
**contarle tu caso al asistente**: propone cómo completarla, te pregunta lo que falta y nada se
aplica hasta que tocás **Aceptar cambios**.

Con la ficha y los datos, el wizard **adapta el plan**:

- saltea pasos que no hacen falta (por ejemplo, **Etiquetado** si ya hay etiquetas) y dice por qué;
- sugiere la tarea, la métrica y el tipo de arquitectura, cada uno con su **Por qué** y un botón
  **Usar**;
- avisa antes de entrenar si algo no cierra: entradas que varían juntas cuando querés descubrir una
  regla, partición aleatoria con datos temporales o con entidades repetidas, pocas filas.

Si el caso no encaja en lo que Perceptron resuelve, el wizard lo dice y sigue con los pasos
estándar.

Además:

- En **Calidad**, **Revisar la ficha** compara lo que dijiste con lo que muestran los datos (por
  ejemplo, entradas declaradas independientes que en realidad varían juntas) y propone
  correcciones para aceptar o descartar.
- Si el objetivo es descubrir una regla, aparece el paso **Fórmula sugerida** antes de entrenar.
- Si un error es peor que el otro en una clasificación de dos clases, aparece **Umbral de
  decisión**: indicás cuántas veces peor es, y al evaluar se elige con validación el umbral que
  minimiza el costo y se muestra en el test frente al del 50 % (**Umbral por costo de los
  errores**, en la página del run).
- Cuando el plan cambia, un aviso **El plan cambió** dice qué pasos se agregaron o quitaron y qué
  avisos son nuevos o se resolvieron.

## Arquitecturas

### Propuestas: por reglas o por LLM

**Proponer arquitecturas** genera 2 a 4 propuestas. Cada una trae justificación, cantidad de parámetros,
memoria y tiempo estimado por época.

- **Por reglas:** salen del catálogo de Perceptron según modalidad, tarea y tamaño de los datos. Se usan
  sin LLM, con L0, si se agotó el presupuesto del LLM o si el LLM falló la validación tres veces. El
  motivo se muestra («Se usaron reglas: …»).
- **Por LLM** (**Sugerido por IA**): el arquitecto lee el perfil (según tu nivel de privacidad), el
  objetivo y el hardware, y compone bloques del catálogo.

Toda propuesta se **valida** antes de mostrarse: esquema, compatibilidad de formas, memoria contra la
disponible, y disponibilidad y licencia de pesos preentrenados.

El catálogo cubre, entre otras: MLP, ResNet-MLP y FT-Transformer (tabular); una CNN compacta o
modelos preentrenados curados, como EfficientNet (imagen); detección, U-Net y CRNN para OCR (visión
avanzada); TextCNN, BiLSTM y encoders preentrenados (texto); N-BEATS, LSTM/GRU, TCN, PatchTST y
autoencoders (series); CRNN sobre espectrograma (audio).

### Editor visual

En **Diseño → Arquitecturas** (o **Abrir en el editor visual** desde **Entrenar**) ves la arquitectura
como un grafo de bloques:

- **Agregar bloque** → **Elegí un bloque del catálogo**. Conectás bloques arrastrando desde un borde;
  **Supr** borra el seleccionado.
- Tocá un bloque para editar sus parámetros; marcá **Ajustable por HPO** los que la búsqueda debe
  explorar.
- La validación es en vivo: **Válida** o la lista de **Problemas de validación**, más el resumen de
  parámetros y memoria.
- **Ver como código** muestra el código PyTorch equivalente.
- **Guardar como nueva** crea otra arquitectura (origen *manual*); la original no se toca.

En el paso **Arquitectura** del wizard, **Definir paso a paso** te lleva por **Familia → Backbone o
tamaño → Cabeza y pérdida → Regularización**, con la opción **Recomendada** marcada, **Explicame las
opciones** (pregunta al copiloto), **Completar con lo recomendado** y **Crear esta arquitectura**.

### Modo experto (código)

Desde el editor, **Modo experto (código)** abre un editor donde escribís (o el LLM propone) el modelo
como código PyTorch con una función `build_model(config)`.

- El código **no es declarativo**: Perceptron no puede explicarlo ni validarlo como una ArchSpec.
- Corre en un **sandbox** sin red, sin procesos y sin acceso a archivos fuera del run, con límites de
  memoria y tiempo. El lint marca problemas en vivo.
- Para guardarlo tildás «Entiendo que el código no es declarativo y corre en un sandbox» y tocás
  **Probar y guardar**: se construye el modelo en el sandbox y se informa su cantidad de parámetros.
- El run queda marcado como **Código experto (no declarativo)**. Para cambiarlo, **Editar como nueva**.

## Búsqueda de hiperparámetros (HPO)

**Recomendar estrategia** analiza tu escenario (tamaño de datos, costo por intento, presupuesto,
hardware) y devuelve una **Estrategia de búsqueda**, la **poda** y los **Hiperparámetros a ajustar**.
Con LLM la recomienda el estratega; sin LLM, las reglas. Podés descartarla.

| Estrategia | Cuándo conviene |
|---|---|
| Evaluación única | Presupuesto muy bajo o receta conocida |
| Random search | Espacios grandes con poco presupuesto; baseline |
| Grid search | Pocos hiperparámetros discretos |
| **TPE** (por defecto) | Caso general |
| CMA-ES | Espacios continuos, presupuesto medio o alto |
| NSGA-II (multi-objetivo) | Balancear calidad contra tamaño o latencia |
| Poda: mediana, ASHA, Hyperband | Entrenamientos largos donde las primeras épocas ya anticipan el resultado |

**Presupuesto:** **Intentos (trials)**, **Épocas máximas por intento** y, en el wizard, **Tiempo máximo**
y la métrica objetivo. Lo que ocurra primero corta la búsqueda. Un estudio interrumpido se puede
reanudar.

Por defecto cada entrenamiento usa precisión mixta si el hardware lo permite, early stopping,
checkpoints del mejor y del último, semillas fijas y tamaño de batch automático.

## Experimentos

En **Experimentos**:

- **Entrenamiento en vivo:** época, métricas y **Curvas de validación por intento** mientras corre.
- **Entrenamientos:** cada run con su **Estado** (En cola, Entrenando, En pausa, Terminado, Falló,
  Cancelado) e **Inicio**.
- **Comparar:** marcá dos o más runs y tocá **Comparar**. Ves las curvas superpuestas (elegí la
  **Métrica de las curvas**) y una tabla donde se resaltan los hiperparámetros que difieren.

En la página de cada run, el **Diagnóstico del entrenamiento** detecta sobreajuste, subajuste,
divergencia o tasa de aprendizaje mal calibrada y sugiere acciones (por LLM o **por reglas**).

## Agente autónomo

En **Agente** el LLM actúa como ingeniero de ML: propone arquitectura y estrategia, entrena, diagnostica
y vuelve a proponer, hasta alcanzar el objetivo o agotar el presupuesto.

1. Fijá los límites: **Intentos máximos (total)**, **Iteraciones máximas** y **Costo máximo de LLM
   (USD)**.
2. Elegí la **Aprobación humana**: **Nunca**, **Antes de cada iteración**, **Si cambia de familia de
   arquitectura** o **Al superar el 50 % del presupuesto**.
3. Tocá **Lanzar agente**.

Mientras trabaja ves la **Bitácora** en vivo (decisiones, sugerencias, límites, aprobaciones), las
iteraciones, los intentos y el costo. Si pide permiso, aparece **El agente pide tu aprobación**.
**Detener** lo corta conservando el mejor modelo hasta el momento.

- Los límites los aplica el sistema, no el LLM.
- El agente **nunca ve el test sellado**: se abre solo al cerrar, en **Resultado en el test sellado**.
  **Ver el mejor run** te lleva a ese run para registrarlo.
- Si el LLM falla, el agente hace una iteración por reglas («Usó reglas») o cierra de forma segura.

El mini-torneo de arquitecturas (entrenar cada propuesta con un presupuesto corto y seguir con la
mejor) está disponible por CLI y API; en la UI llega **próximamente**.

## Dónde corre el entrenamiento

- **En tu equipo:** en el dispositivo recomendado (GPU NVIDIA CUDA, AMD ROCm en Linux, Intel XPU o CPU)
  o el que elijas en **Dispositivo**. Cada run corre en un proceso separado; si se queda sin memoria o
  falla, se informa con diagnóstico.
- **En un Team Server desde el desktop:** conectá el servidor en **Configuración → Servidores de
  equipo**. En el paso **Entrenar**, **Dónde entrenar** ofrece **En este equipo** o **En «servidor»**;
  tildá **Usar la GPU del servidor** para ir a un worker con GPU. El proyecto se sube al servidor y ves
  el progreso en vivo igual que en local.
- **En la UI web del Team Server:** los entrenamientos van directamente a la cola de workers. Ver
  [Team Server](team-server.md#cola-de-entrenamiento).

Multi-GPU en un mismo equipo y un intento por GPU en paralelo llegan **próximamente**.
