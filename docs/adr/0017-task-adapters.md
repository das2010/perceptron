# ADR-0017: Adaptadores de tarea (TaskAdapter)
- Estado: aceptado
- Fecha: 2026-09-27
- Contexto: La Capa 1a solo tenía clasificación y regresión, con ramas `if REGRESSION / else` repartidas en el módulo de Lightning, la inferencia, la evaluación y el builder. La Capa 1b suma forecasting, anomalías, detección, segmentación, OCR y eventos sonoros (SPEC §2), cada una con salidas, loss, métricas y evaluación distintas.
- Decisión: Un adaptador por `TaskType` en `perceptron/tasks/`, con una interfaz única: `num_outputs`, `check_output` (etapa 4 de §9.3), `build_loss`, `build_metrics`, `step`, `predict` y `evaluate`. `training.module.PerceptronModule`, `training.inference.predict`, `evaluation.evaluate.evaluate_run` y `archspec.builder` solo delegan (`tasks.get_adapter`). El `EvaluationReport` conserva `classification`/`regression` y agrega `details` y `curves` genéricos.
- Consecuencias: agregar una tarea es un archivo nuevo más su registro, sin tocar el runner ni la API. Los tests de la 1a garantizan que el refactor no cambia el comportamiento.
- Alternativas consideradas: un LightningModule por tarea (duplica optimizadores, schedulers y fine-tuning); seguir con condicionales (no escala a 8 tareas).
