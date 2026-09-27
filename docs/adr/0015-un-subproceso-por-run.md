# ADR-0015: Un subproceso por run/trial con eventos JSONL
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: RF-TRN-05 exige aislar cada run (estado de CUDA, OOM, crashes nativos) y reportar progreso en vivo (RF-TRN-06).
- Decisión: Cada run o trial de HPO se ejecuta como `python -m perceptron.training.worker <run.json>`. El worker escribe eventos JSON, uno por línea, en stdout (`started`, `epoch`, `batch`, `metric`, `checkpoint`, `finished`, `error`); stderr queda para logs. El `runner` del Engine supervisa el proceso, publica los eventos en el `EventBus` (de ahí a WebSocket y tracking), detecta OOM y salidas anómalas, y cancela o pausa (fin de época + checkpoint). HPO lanza los trials secuencialmente y usa los eventos `metric` para el pruning de Optuna.
- Consecuencias: el aislamiento es total y un crash no tumba el Engine; hay un costo de arranque (~1–3 s por trial, importar torch). En Windows el spawn obliga a que todo lo que cruza procesos sea serializable (config JSON).
- Alternativas consideradas: hilos o procesos `multiprocessing` en el mismo intérprete (sin aislamiento de CUDA); Ray (dependencia pesada, se evalúa para el servidor en D3).
