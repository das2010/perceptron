# ADR-0004: PyTorch + PyTorch Lightning
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: SPEC §2 fija el framework de deep learning.
- Decisión: PyTorch + Lightning; el `LightningModule` se genera desde ArchSpec. Las dependencias pesadas (torch, lightning, mlflow, optuna) entran en la Capa 1 como extras del engine, no en la Capa 0.
- Consecuencias: Variantes de wheel por hardware (CUDA/ROCm/XPU/CPU), RF-TRN-02.
- Alternativas consideradas: JAX / TensorFlow (descartados en §2).
