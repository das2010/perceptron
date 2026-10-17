# ADR-0042: Asistente guiado como experiencia principal
- Estado: aceptado (2026-10-17, pedido explícito: «Convertí la experiencia principal en un
  recorrido guiado por un asistente de IA para usuarios sin conocimientos de machine learning»)
- Contexto:
  - La pantalla del proyecto mostraba once pestañas técnicas (datos, diseño, entrenamiento,
    experimentos…). Alguien sin conocimientos de ML no sabe por dónde empezar ni qué decidir.
  - El wizard (ADR-0040/0041) ya diseña a partir de la ficha del caso, pero sus preguntas
    asumen vocabulario técnico y deja al usuario el resto del circuito.
  - Una primera entrega cubría solo clasificación de imágenes y duplicaba la entrevista y el
    diseño fuera del wizard.
- Decisión:
  - La ruta del proyecto (`/projects/:id`) es el asistente guiado. La vista general pasa a
    `/projects/:id/overview`. Las pestañas técnicas quedan en «Opciones avanzadas»: plegadas en
    el asistente y abiertas en cualquier ruta avanzada. Las URLs existentes se conservan.
  - El asistente es un coordinador sobre los servicios existentes, no un motor paralelo. La
    entrevista completa la misma ficha del caso (`UseCaseBrief`) del wizard, y el
    entrenamiento usa `Wizard.design`, `new_study`/`launch_study`, `evaluate`, `register` y
    `export`.
  - Reglas puras deciden qué falta, qué casos se soportan y cómo se explica el resultado. El
    LLM solo conversa y propone datos con cita literal del usuario; sin LLM, cada pregunta
    tiene un control equivalente.
  - Los casos que el asistente no cubre (series, anomalías, detección, sin etiquetas) se
    derivan con el motivo al wizard avanzado o a Etiquetado.
- Consecuencias:
  - Un usuario nuevo llega a un modelo validado sin ver términos técnicos. Un usuario experto
    sigue teniendo todo a un clic.
  - Agregar un caso al asistente requiere su regla en `journey_guide`, su guía de datos y su
    formulario de entrada. Mientras eso no exista, el caso se deriva al modo avanzado.
  - La predicción por el runtime ya no exige columnas que el modelo descarta (IDs). Es un
    cambio compatible: antes esas columnas eran obligatorias y se ignoraban.
- Alternativas descartadas:
  - Un asistente solo de imágenes: no cumple el objetivo y duplica el wizard.
  - Reemplazar el wizard: perdería el control fino que necesitan los usuarios expertos.
