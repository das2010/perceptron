# ADR-0006: LLM Gateway pluggable con IDs de modelo en configuración
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: SPEC §2 y §7.7.1: Anthropic (default), OpenAI, Gemini, Kimi, OpenAI-compatible y Ollama.
- Decisión: Toda llamada pasa por `perceptron.llm.gateway` con propósito (`LLMPurpose`), schema de salida y PrivacyFilter. Los IDs de modelo no se hardcodean: viven en configuración (perfiles por propósito, RF-LLM-03). En tests se usa `FakeLLMProvider`.
- Consecuencias: Se implementa en Capa 2; la Capa 0 ya modela `LLMSession`/`LLMCall` y el enum de propósitos.
- Alternativas consideradas: Llamar SDKs de proveedores directamente desde cada módulo (descartado: rompe privacidad y auditoría).
- Implementación (Capa 2a): `llm/gateway.py`, adaptadores en `llm/providers/`, catálogo y perfiles en `llm/catalog.yaml` (ADR-0020), salidas estructuradas con reintento (ADR-0021).
