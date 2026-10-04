# ADR-0040: Wizard adaptativo — ficha del caso (LLM) y plan compilado (reglas)
- Estado: aceptado (2026-10-04)
- Contexto:
  - El wizard (SPEC §7.6, RF-WIZ-01..04) tiene 9 pasos fijos. El copiloto conversa y sugiere
    un `DraftPatch` sobre un puñado de campos (objetivo, target, tarea, métrica, umbral,
    presupuesto), pero no construye una comprensión estructurada del caso y no cambia qué pasos
    se muestran ni sus defaults.
  - Casos reales que el wizard actual no supo guiar:
    - «Tabla 3» (`multiplo = 3 × numero`): el objetivo era una regla y había que extrapolar. El
      flujo propuso redes y 15 épocas; la regresión lineal y la fórmula sugerida (ADR-0039)
      llegaron después, a pedido.
    - «Sensores» (`Salida = S1·(1+√S2)`): las entradas eran colineales (`S2 = √S1`), así que la
      regla no se podía inferir. Se descubrió después de entrenar 100 trials.
  - Lo que el usuario sabe del caso (costo de los errores, si hay que extrapolar, dónde se va a
    usar el modelo, si las entradas son independientes) hoy queda en un texto libre que solo
    leen el copiloto y el arquitecto.
- Decisión:
  - **Tres responsabilidades separadas:** el LLM *entiende* el caso, reglas determinísticas
    *deciden* el plan y la persona *aprueba*. El LLM no inventa pasos ni pantallas.
  - **Ficha del caso (`UseCaseBrief`)** en `services/wizard.py`, validada con Pydantic
    (RF-LLM-04, ADR-0021):
    - `problem`: predecir un valor, predecir una categoría, detectar anomalías, pronosticar,
      descubrir una regla u «otro» (con texto).
    - `prediction`: qué se predice y en qué unidad.
    - `error_costs`: simétricos, o qué error es peor y cuánto (p. ej. «no detectar una falla es
      10× peor»).
    - `business_metric` → `target_metric` y `success_threshold`, que ya existen en `DraftValues`.
    - `data_facts`: fechas, entidades repetidas (máquinas, clientes), etiquetas disponibles,
      entradas declaradas independientes.
    - `usage`: extrapolar fuera del rango visto, explicabilidad, dónde corre (desktop,
      servidor, equipo chico, planilla) y latencia.
    - `assumptions` y `open_questions`, cada una con su confianza.
    - Cada campo es opcional y registra su origen (usuario, LLM, perfil) para el historial.
  - **Entrevista (paso Objetivo):**
    - Nuevo prompt `copilot/intake` del propósito `copilot`. A partir de la conversación propone
      un `BriefPatch` y la próxima pregunta, dirigida a los campos que faltan o tienen baja
      confianza.
    - Pocos casos resueltos sirven de ejemplo: las plantillas de proyecto y los UC-01..11.
    - La persona ve la ficha en un panel, la edita y la acepta. Nada entra a la ficha sin
      aceptación.
  - **Compilador de plan (`compile_plan(brief, card) -> WizardPlan`)**, puro y determinístico:
    - **Pasos:** cuáles van y en qué orden.
      - Etiquetado solo si faltan etiquetas.
      - Split temporal si hay fechas y por grupo si hay entidades.
      - Fórmula sugerida si `problem` es «descubrir una regla».
      - Calibración del umbral si los costos de error son asimétricos.
      - Explicabilidad si se pide.
    - **Defaults por paso:** métrica y umbral, estrategia de split, familias de arquitectura a
      preferir o evitar (p. ej. lineal y fórmula primero si hay que extrapolar), épocas y
      presupuesto.
    - **Chequeos con severidad:** p. ej. «extrapolar + solo redes», «regla + entradas
      colineales», «pocas filas para la métrica pedida».
    - **«¿Por qué?» de cada elemento** que cita el campo de la ficha o del perfil que lo originó
      (RF-WIZ-02).
    - Usa solo hechos medibles (ficha y Profile Card), nunca nombres de columnas ni de datasets.
    - Un caso que ninguna regla reconoce produce el plan estándar de 9 pasos, que es el
      comportamiento actual.
  - **Reconciliación con los datos:**
    - Al ingerir o perfilar, el LLM compara la ficha con la Profile Card y propone correcciones
      o preguntas (p. ej. «dijiste que los sensores son independientes, pero su correlación de
      rangos es 1,0»).
    - Las contradicciones medibles (colinealidad, fechas o grupos no declarados, desbalance
      contra los costos declarados) también las detecta el compilador sin LLM.
    - El plan se recompila y la UI muestra qué cambió.
  - **Los demás roles consumen la ficha:**
    - El arquitecto, el estratega de HPO, el diagnosticador y el informante la reciben en el
      `LLMContext` junto al objetivo.
    - El diagnosticador compara contra `success_threshold`.
    - El informante escribe en términos de negocio («detecta el 96 % de las fallas»).
  - **Persistencia:** `ProjectDraft.values` suma `brief` y `plan`, versionados con el historial
    existente (RF-WIZ-04). Es aditivo: los borradores viejos siguen abriendo con el plan
    estándar.
  - **UI:**
    - El wizard dibuja los pasos y defaults de `plan` en vez de la lista fija.
    - El panel «Ficha del caso» es editable.
    - Cada paso tiene su «¿por qué está este paso?» y hay una vista de cambios del plan.
  - **Sin LLM o en L0 (RF-WIZ-03):** la ficha se completa como formulario y corre el mismo
    compilador. En L1 el LLM ve la Profile Card y no valores individuales (ADR-0007), suficiente
    para la entrevista y la reconciliación.
- Alcance y límites:
  - Es genérico en el mecanismo y está acotado por el catálogo: se adapta a cualquier dataset
    de las modalidades y tareas que el Engine resuelve (tabular, imagen, texto, series, audio,
    detección, segmentación, OCR).
  - Ante problemas fuera de ese catálogo (recomendación, grafos, refuerzo, generativos,
    multimodal combinado), la ficha lo registra como «otro» y el wizard lo dice
    explícitamente. No simula un plan.
  - Solo detecta las contradicciones de datos que se miden: colinealidad, fuga, fechas, grupos,
    desbalance, tamaño y nulos. Un sesgo de muestreo, por ejemplo, queda fuera.
  - Si la persona describe mal el caso, la ficha sale mal. Por eso se muestra y se aprueba, y
    la reconciliación atrapa una parte.
  - Ampliar la cobertura es agregar una regla al compilador, con su test.
- Verificación:
  - Tests del compilador: tablas ficha + perfil → plan esperado (pasos, defaults, chequeos).
  - Conversaciones grabadas con `FakeLLMProvider` de «Tabla 3», «Sensores», UC-01..11 y un
    caso fuera de catálogo. Validan la ficha, el plan y que no se aplique nada sin aceptación.
  - Aceptación con proveedores reales en `llm.yml` (manual y nocturno).
- Consecuencias:
  - Cambia `DraftValues` (aditivo), un prompt nuevo del copiloto, endpoints del wizard para la
    ficha y el plan, y la página del wizard. El contrato OpenAPI → TS se regenera.
  - Costo de LLM: una entrevista de 5 a 10 turnos más una reconciliación por proyecto, del
    orden de centavos. El caché y los presupuestos existentes (RF-LLM-06/07) aplican.
  - El copiloto sigue disponible en todos los pasos con su `DraftPatch`. La ficha suma
    contexto; no lo reemplaza.
- Fases:
  1. Ficha, entrevista, panel editable y compilador sobre los pasos actuales (incluir, saltear,
     ordenar, defaults y chequeos), con sus tests. ~1–1,5 semanas.
  2. Reconciliación con la Profile Card, pasos condicionales nuevos (calibración del umbral,
     fórmula sugerida dentro del wizard), vista de cambios del plan y aceptación con LLM real.
     ~1 semana.
- Alternativas consideradas:
  - **Que el LLM genere el plan directamente:** más flexible, pero no reproducible, difícil de
    testear, puede proponer pasos que no existen y no funciona sin LLM (RF-WIZ-03).
  - **Árbol de preguntas fijo, sin LLM:** reproducible, pero rígido. No entiende descripciones
    de dominio ni repregunta lo ambiguo. Queda como modo L0 del mismo diseño.
  - **Ampliar solo `DraftPatch`:** suma campos sueltos sin cambiar pasos ni detectar
    contradicciones. No resuelve los casos del contexto.
