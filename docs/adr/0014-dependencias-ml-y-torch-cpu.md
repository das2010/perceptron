# ADR-0014: Dependencias de ML, índice CPU de PyTorch y licencias
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: La Capa 1 incorpora el stack de ML (SPEC §5.1). El producto es comercial (§16), PyTorch se distribuye en variantes por hardware (RF-TRN-02) y el CI no tiene GPU.
- Decisión: Extra `ml` de `perceptron-engine` con torch 2.14, torchvision 0.29, lightning 2.6, torchmetrics 1.9, timm 1.0, optuna 5.0, mlflow 3.16, polars 1.44, pyarrow 25, fastexcel 0.21, scikit-learn 1.9, pillow 12.3, psutil 7.2 y jinja2 3.1 (versiones del `uv.lock` al 2026-09-26). En desarrollo y CI, `torch`/`torchvision` salen del índice explícito `pytorch-cpu` (`https://download.pytorch.org/whl/cpu`). La app detecta el hardware (`GET /system/hardware`) y recomienda la variante (cpu/cuda/rocm/xpu); su instalación en el runtime embebido es Capa 3.
- Licencias directas: BSD-3 (torch, torchvision, scikit-learn, psutil, jinja2), Apache-2.0 (lightning, torchmetrics, timm, mlflow, pyarrow), MIT (optuna, polars, fastexcel), MIT-CMU/HPND (pillow). Todas compatibles con distribución comercial. La auditoría transitiva completa es Capa 7. Las licencias de **pesos** preentrenados se registran por modelo en el catálogo (§8).
- Consecuencias: el CI descarga torch CPU (~200 MB, con caché de uv). Quien tenga GPU local debe instalar la variante correspondiente hasta que exista el instalador de Capa 3.
- Alternativas consideradas: torch desde PyPI (en Linux trae CUDA, +2 GB en CI); dependencias de ML como obligatorias (impediría usar la CLI liviana sin ML).
