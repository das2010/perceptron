# ADR-0005: ArchSpec declarativa + código experto en sandbox
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: El LLM propone arquitecturas; el sistema debe validarlas (SPEC §4.2 p.4–5, §9).
- Decisión: ArchSpec (JSON validado por Pydantic ⇄ JSON Schema) como formato por defecto, restringido a bloques del catálogo. El código PyTorch libre es opt-in (modo experto), corre en sandbox (§13.2) y queda marcado como no declarativo. En el dominio, `ArchSpecRecord` guarda `spec` o `code_path`.
- Consecuencias: Validación estricta y reproducibilidad; el catálogo limita la expresividad del LLM a propósito.
- Alternativas consideradas: Solo código libre (inseguro y difícil de validar).
