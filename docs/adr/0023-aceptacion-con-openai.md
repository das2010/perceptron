# ADR-0023: Aceptación de la Capa 2 también con OpenAI
- Estado: aceptado (pedido del usuario, 2026-09-27)
- Fecha: 2026-09-27
- Contexto: SPEC §14 pide que el agente complete UC-01, UC-04 y UC-09 "con Claude y con un
  modelo local". El equipo dispone de una clave de OpenAI y no de una de Anthropic para el CI.
- Decisión: el workflow `llm.yml` corre la matriz de aceptación y los golden tests con
  Claude, OpenAI (perfil `openai` del catálogo: GPT-5 / GPT-5 mini) y Ollama; cada proveedor
  en la nube se saltea si falta su secret (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). El
  benchmark O2 elige el proveedor con `benchmark_provider`. Claude sigue siendo el proveedor
  por defecto del producto (SPEC §2); esto solo cambia con qué proveedor se valida.
- Consecuencias: el criterio de §14 queda cubierto con OpenAI + Ollama hasta que haya clave de
  Anthropic; los resultados registran el perfil usado. Los modelos de razonamiento de OpenAI
  van sin temperatura y con `reasoning_effort: low` (catálogo) para que no agoten el tope de
  tokens; si igual pasa, el adaptador lo informa como error del proveedor.
- Alternativas consideradas: esperar la clave de Anthropic (bloquea la aceptación); cambiar el
  default del producto a OpenAI (contradice §2).
