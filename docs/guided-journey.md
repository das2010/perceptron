# Asistente guiado — recorrido principal

Experiencia principal de Perceptron para personas sin conocimientos de machine learning
(ADR-0042). Reutiliza el wizard (RF-WIZ-01/03/04), la ingesta y el perfilado (RF-ING-01/07/08),
el diseño guiado por requisitos (ADR-0041), el HPO, la evaluación con test sellado, el registro,
la exportación y el Playground (RF-EXP-02). No reemplaza al modo avanzado: todas las herramientas
técnicas siguen en «Opciones avanzadas» del proyecto.

## Qué casos cubre

| Datos | Objetivo | Cómo se resuelve |
|---|---|---|
| Tabla (CSV / Excel) | elegir una categoría | clasificación tabular |
| Tabla | estimar un número | regresión tabular |
| Tabla | descubrir la regla o fórmula | regresión + búsqueda de fórmula simbólica (ADR-0039) si los datos son casi determinísticos |
| Imágenes (ZIP por carpetas) | elegir una categoría | clasificación de imágenes |
| Textos (tabla con columna de texto) | elegir una categoría | clasificación de textos |
| Audios (ZIP por carpetas) | elegir una categoría | clasificación de audio |
| Tabla con fechas | pronosticar los próximos valores | series de tiempo (N-BEATS/PatchTST), horizonte elegido por la persona; resultado en % de error y contra repetir el último valor |
| Tabla con fechas | detectar momentos raros | autoencoder de series; con casos marcados se mide cuántos detecta, sin marcas aprende lo normal (umbral por cuantil alto) |
| Fotos + anotaciones (COCO, VOC, YOLO) | ubicar objetos | detección (CenterNet); resultado en % de objetos bien ubicados (mAP@50) |
| Fotos + máscaras | marcar la zona exacta | segmentación (U-Net); coincidencia de la zona (IoU) |
| Imágenes + `labels.csv` | leer texto | OCR (CRNN + CTC); % de caracteres bien leídos y textos exactos |

Nada queda en un callejón sin salida:
- **Ante la duda, pregunta.** Si la persona habla de algo «en el tiempo», el asistente pregunta si quiere estimarlo según otros datos o pronosticar los próximos valores. Con fotos, distingue entre decir el tipo, ubicar o marcar la zona.
- **Si la combinación no tiene recorrido, propone un replanteo que sí lo tiene.** Por ejemplo, agrupar en rangos un número que se quiere estimar con fotos, o exportar a una planilla.
- **Si faltan las respuestas correctas, guía a Etiquetado.** Excepción: los casos raros, que se pueden entrenar sin marcas.

## Recorrido

0. **Archivos primero (opcional).** El usuario puede adjuntar uno o más archivos en Inicio o con
   el clip de la conversación, antes de contar el caso o en el medio. `journey_files.read` los
   lee sin valores individuales (columnas con tipo, cantidades, carpetas) y arma objetivos
   posibles con reglas:
   - **Planillas:** predecir cada columna de pocas categorías, estimar cada número, pronosticar
     si hay fechas regulares (por serie) y detectar momentos raros (marcados o no).
   - **Fotos:** ubicar (con anotaciones COCO/VOC/YOLO), marcar zonas (`images/` + `masks/`),
     leer texto (`images/` + CSV) o clasificar por carpeta.
   - **Grabaciones y textos:** clasificar por carpeta.
   - **Combinaciones:** varias planillas con las mismas columnas se juntan. Si una columna es
     única en una planilla y aparece en la más larga, se unen por esa columna. La tabla
     combinada se materializa al elegir.

   Cada propuesta trae los hechos del caso que implica. El LLM (prompt `journey_goals`) elige
   cuáles mostrar primero y lo cuenta con los nombres de las columnas. No inventa: su orden se
   valida contra los ids. El usuario elige con un clic o con sus palabras, nombrando la columna
   o diciendo algo que encaja con una sola propuesta (`mentioned`, `consistent`). Lo demás se
   pregunta como siempre, y al confirmar los datos se revisan sin volver a subirlos.
1. **El caso.** El usuario describe el problema en Inicio. El asistente pregunta, de a una
   pregunta por vez y con un ejemplo, por el próximo dato que falta, en tantas iteraciones como
   haga falta. Debajo de la conversación hay un control de respaldo para ese mismo dato, útil sin
   LLM o en L0. «Lo que entendí» muestra los datos con un botón para corregir cada uno. Con todo
   completo, el usuario confirma el resumen.
2. **Los datos.** El motor genera la guía (`data_guide`): pasos, una tabla de ejemplo con las
   columnas del caso (o el árbol de carpetas del ZIP), una plantilla CSV y consejos (no incluir
   datos que se conocen después, cantidad mínima de filas, rango a cubrir, la categoría
   importante). El selector de archivos acepta solo los formatos del caso.
3. **Revisión.** Ingesta, perfil, alertas de la ficha y chequeos del plan del wizard. Los
   bloqueos (sin columna resultado, menos de 20 ejemplos, una sola categoría, carpetas que no
   coinciden con las categorías, archivos ilegibles, ninguna columna utilizable) impiden seguir;
   las advertencias se muestran. Si la columna resultado no se detecta, se elige de una lista.
4. **Entrenamiento.** `Wizard.design` con torneo automático: requisitos, arquitecturas, mini
   torneo y estrategia. Después estudio, evaluación en test sellado, registro y exportación. La
   UI muestra las etapas reales y «Prueba N de M»; se puede salir y volver. Al terminar queda un
   aviso persistente hasta que el usuario lo reconoce.
5. **Resultados.** Explicación sin jerga y determinística (aciertos y error típico en % sobre
   casos que el modelo nunca vio, aciertos por categoría, casos importantes no detectados y
   falsas alarmas), si cumple el objetivo acordado y los próximos pasos. Si encontró una fórmula
   exacta, se ofrece para copiar en Excel o Python.
6. **Validación.** El asistente explica cómo elegir casos nuevos (otro día, otros objetos u
   otras personas) con su resultado real. El modelo predice sin ver el resultado y después se
   compara. En tablas y textos también se puede subir un CSV con varios casos. Se necesitan al
   menos 10 casos nuevos y todas las categorías para quedar listo para usar.
7. **Uso.** Un formulario con solo las columnas que el modelo usa: sin el resultado ni los
   identificadores descartados, con las opciones de cada categoría y un ejemplo de cada campo.
   En imágenes y audio, solo el archivo. La predicción va por el Playground con el run
   registrado.

## Conversación

Toda la experiencia es una sola conversación. El asistente cuenta cada etapa en el mismo hilo,
y el control de la acción actual (guía y subida de datos, revisión, etapas de entrenamiento,
validación, formulario de uso) va pegado al mensaje que lo presentó. Siempre se le puede
escribir: lo que se pregunta después queda debajo, como en cualquier chat.

- **Cada respuesta tiene contestación.** Con IA, el asistente acusa lo que entendió con las
  palabras de la persona, responde dudas y hace una sola pregunta adaptada al caso, con 2 a 4
  respuestas rápidas propuestas (prompt `copilot/journey-v3`). Si la IA no responde, pregunta
  por un dato que ya se contestó o hace una pregunta sin campo, se usa la pregunta redactada
  para el tipo de datos (`journey_talk.question`). Así no hay silencios ni preguntas
  repetidas.
- **Sin IA** (sin proveedor, L0 o error): las preguntas están redactadas y traen respuestas
  rápidas. Lo escrito a mano se entiende cuando responde la pregunta anterior: texto libre,
  listas («con fisura y sin fisura»), porcentajes y «8 de cada 10», sí/no
  (`journey_talk.understand`).
- **Después de la entrevista**, `Coordinator.ask` responde con el estado real del recorrido:
  datos pedidos, revisión, avance, resultado y validaciones (prompt `copilot/journey_chat`). Si
  corresponde, ofrece una acción permitida en esa etapa («Preparar y entrenar», «Detener»,
  «Corregir el caso») para aceptar con un botón; nunca cambia el estado por su cuenta.
- **Inicio** es una caja de conversación con ejemplos para tocar. El primer mensaje se envía al
  abrir el proyecto, con «escribiendo…». Enter envía y Shift+Enter agrega una línea.
- **Aviso al terminar:** queda en la conversación, con un punto en la pestaña del navegador y,
  si la persona lo permite, una notificación del sistema.

**Cómo se mide.** `engine/tests/golden/test_golden_journey.py` conversa con proveedores reales:
personas simuladas que responden en lenguaje natural, cuentan todo de entrada, preguntan «¿qué
querés decir?» o piden acciones. Verifica que el caso quede completo y sin jerga, y guarda las
transcripciones como artefacto (`llm.yml`, filtro `golden_filter=journey`). Con el prompt
anterior, en tres casos el asistente quedó callado varios turnos y repitió seis veces la misma
pregunta; con el actual, cada turno acusa, responde y pregunta una sola cosa.

## Arquitectura

- `services/journey_guide.py` (puro): `Facts`, `missing`, `unsupported_reason`, `to_brief`,
  `data_guide`, `summarize`, `next_steps` e `input_form`. Las reglas deciden qué falta y qué se
  soporta; el LLM no controla transiciones.
- `services/journey.py`: `Coordinator` y `Journey` (`schema_version=2`), guardado en la tabla
  documental con compare-and-swap por versión. Los recorridos de la versión 1 (solo imágenes) se
  retoman desde la entrevista, conservando la conversación.
- **Entrevista:** el mensaje se guarda antes de llamar a `llm.gateway` (propósito COPILOT, prompt
  `copilot/journey-v2`, schema `InterviewTurn`, PrivacyFilter y auditoría habituales). Un dato
  propuesto por el LLM se aplica solo si trae una cita literal del usuario y valida; los
  supuestos van aparte. Los datos que elige el usuario en los controles se aplican directamente.
- **Confirmación:** los datos se escriben en la ficha del caso del `ProjectDraft` del wizard
  (`UseCaseBrief`), así el arquitecto, los requisitos de diseño y el plan usan la misma fuente.
- **Trabajo en segundo plano:** reserva durable antes de lanzar (doble clic → 409), PID y momento
  de creación del proceso para recuperar interrupciones, y cancelación de workers remotos.
  Detener deja el recorrido en `interrupted` con los datos conservados.
- **API:** `GET/POST /projects/{id}/journey` (acciones message, confirm, prepare, train, stop,
  acknowledge, revise), `POST /journey/validate` (JSON) y `POST /journey/validate/file`
  (archivo + resultado real, o CSV de hasta 500 casos). Lectores: solo `GET`; validar y las
  acciones requieren editor.
- **Runtime:** `required_columns` exige solo las columnas que usa el modelo; las descartadas
  (IDs) ya no son obligatorias al predecir.
- **UI:** `features/journey/JourneyPage.tsx` muestra un paso por vez (stepper) y
  `JourneyPanels.tsx` tiene los paneles. Consulta el estado cada 2 s solo durante revisión y
  entrenamiento. Textos en `lib/i18n/journey.ts` (es/en).

## Límites

- La validación con casos nuevos es una comprobación práctica, no una certificación
  estadística. Los hashes detectan duplicados exactos de casos ya vistos, no casos parecidos.
- El objetivo es un porcentaje: aciertos en categorías, o error típico máximo en números. El
  costo distinto de los errores se informa y orienta el diseño; no se calibra un umbral.
- La recuperación usa el proceso de la misma instancia; no es un coordinador distribuido.
- Probado con FakeLLMProvider y con entrenamiento real en los tests; el proveedor externo se
  prueba en el workflow LLM.

## Pruebas

`engine/tests/api/test_journey.py`: reglas, guía, citas, entrevista sin LLM y con FakeLLM,
confirmación en la ficha, casos no soportados, permisos entre proyectos, CAS, recuperación,
cancelación remota, recorridos legados, bloqueos de carpetas y columna, y el recorrido completo
de tabla (churn) con entrenamiento, validación individual y por CSV y predicción.
`server/tests/test_server_rbac.py::test_journey_permissions` cubre los permisos.
`ui/src/features/journey/JourneyPage.test.tsx` cubre que cada paso muestre solo sus controles.
