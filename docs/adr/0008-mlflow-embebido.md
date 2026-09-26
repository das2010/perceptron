# ADR-0008: Tracking con MLflow embebido
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: SPEC §2: MLflow local (SQLite) y en servidor (PostgreSQL + S3).
- Decisión: MLflow como backend de tracking; la UI de Perceptron muestra runs nativamente. En el workspace local se reserva `mlflow/`.
- Consecuencias: Integración en Capa 1.
- Alternativas consideradas: Tracking propio (descartado: reinventar registry y artefactos).
