# ADR-0007: Niveles de privacidad L0–L3 con PrivacyFilter central
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: SPEC §7.7.3 y objetivo O4: cero bytes de datos crudos al LLM en ≤ L1.
- Decisión: `PrivacyLevel` por proyecto con default **L1**. El filtro se aplica en el Gateway, nunca en cada llamador; cada `LLMCall` registra el payload ya filtrado y el nivel aplicado (auditoría RF-PRV-03).
- Consecuencias: Tests de propiedad (hypothesis) en Capa 2 verifican que ningún valor individual aparece en payloads L1.
- Alternativas consideradas: Filtrado por llamador (descartado: fácil de olvidar).
