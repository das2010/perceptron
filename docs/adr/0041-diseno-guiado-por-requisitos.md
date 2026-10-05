# ADR-0041: Wizard que diseña — requisitos por escenario, recomendación y cadena guiada
- Estado: aceptado (2026-10-04, pedido explícito: «rediseñá iterativamente el wizard… sin
  interrupciones»)
- Contexto:
  - El arquitecto (RF-ARC-01) devuelve 2–4 propuestas, pero el wizard las mostraba como
    iguales: no decía cuál convenía ni por qué. Elegir quedaba a cargo de la persona.
  - «Tubos»: tres clases con pocas imágenes. El arquitecto propuso redes preentrenadas
    (MobileNetV3, EfficientNet), pero no hubo recomendación y se eligió «CNN compacta desde
    cero», la peor opción para tan pocos datos.
  - «Tabla 3» y «Sensores» (ADR-0040): la ficha decía «regla / extrapolar», pero las propuestas
    no garantizaban una opción lineal.
  - El conocimiento de diseño (transfer learning con pocos datos, modelos chicos en tablas
    chicas, compensar el desbalance, topes de tamaño en equipos embebidos) estaba disperso en
    las reglas de Capa 1 y en el prompt, sin forma de verificarlo.
- Decisión:
  - **Requisitos de diseño (`services/design.py`).** Se derivan de hechos medibles (modalidad,
    tarea, ejemplos de entrenamiento, desbalance, dispositivo, conexión, tokenizer) y de la
    ficha del caso (regla/extrapolar, despliegue, latencia, explicabilidad, costo de errores).
    Es una función pura: mismos hechos, mismos requisitos.
    - Cada requisito tiene `level` `must|should` y `scope` `each|any`. `any` alcanza con que
      lo cumpla una propuesta (p. ej. «una preentrenada»); `each`, todas (p. ej. tope de
      parámetros para edge).
    - Primer set: `pretrained_backbone`, `no_pretrained_offline`, `small_model`,
      `linear_option`, `edge_size`, `low_latency`, `imbalance_handling`, `explainable`,
      `epoch_time`.
  - **El arquitecto los recibe y se lo verifica.** Van en `constraints.design_requirements`
    (prompt `architect/main-v2`). Si falla un `must`, el validador le devuelve el motivo y el
    arquitecto reintenta, con el mismo mecanismo de RF-ARC-02.
  - **Si el `must` sigue sin cubrirse,** el sistema agrega una propuesta de plantilla que lo
    cumple (lineal; MobileNetV3 o EfficientNet-B0 preentrenada), con `origin=rules`.
  - **Evaluación y recomendación.** Cada propuesta recibe un puntaje: 100 menos 45 por cada
    `must` y 15 por cada `should` incumplidos, con la confianza del LLM como desempate. La de
    mayor puntaje queda primera y marcada `recommended`. El quickstart, el agente y el
    mini-torneo usan ese orden.
  - **UI.** El paso Arquitectura (wizard y Entrenar) muestra:
    - los requisitos del escenario;
    - la insignia «Recomendada» y qué requisitos cumple cada opción;
    - un aviso al elegir una opción no recomendada.
  - **Iteraciones implementadas (mismo ADR):**
    1. Requisitos, evaluación y recomendada (arriba).
    2. **Diseño guiado** (`services/autodesign.py`, `POST /projects/{id}/draft/design`, job). La
       cadena es:
       - preparación;
       - propuestas evaluadas;
       - mini-torneo de hasta 3 que cumplan los obligatorios, con:
         - una métrica que todas registran (las pérdidas no son comparables entre focal y
           entropía cruzada);
         - el 15 % de las épocas, con mínimo 2;
         - todo train por época si hay menos de 5000 ejemplos;
         - en modo `auto`, solo si se estima en 15 minutos o menos;
       - elegida con evidencia;
       - épocas por trial;
       - estrategia de HPO.

       El resultado queda en `values.design` del borrador (campo del sistema) y la persona lo
       acepta. Se descarta si cambian los datos, la ficha o la preparación. El quickstart usa el
       mismo criterio de torneo.
    3. **Próximo paso desde el diagnóstico** (`services/improve.py`,
       `GET/POST /runs/{id}/improvements`). Cada acción tipada del diagnosticador se traduce
       en un cambio determinístico. Se parte de los hiperparámetros del mejor trial como
       defaults y se ajusta:
       - el learning rate;
       - la regularización (dropout y luego weight decay);
       - las épocas;
       - el desbalance (pesos por clase y luego sobremuestreo).

       Lo que no es de arquitectura (aumentar datos, cambiar de familia, más datos) indica dónde
       hacerlo. La UI ofrece «Aplicar» y luego «Entrenar con esta mejora», con el presupuesto
       del estudio.
    4. **Golden de escenarios con LLM real** (`tests/golden/test_golden_design.py`):
       - «Tabla 3» → la recomendada es lineal;
       - «Tubos» → el LLM propone una preentrenada y queda recomendada;
       - edge → todas bajo el tope.

       Pasaron con OpenAI sin ajustar el prompt.
    5. Memo «Por qué esta arquitectura» en la revisión y la búsqueda del diseño en el paso HPO.
- Consecuencias:
  - El LLM sigue proponiendo y explicando; quien decide qué es obligatorio es el sistema, de
    forma determinística y testeable. Con reglas (L0) la recomendación funciona igual.
  - La persona puede elegir cualquier opción: la recomendación nunca bloquea.
  - Agregar conocimiento de diseño es sumar un requisito con su chequeo y su test, no editar
    prompts.
