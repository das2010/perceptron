# ADR-0039: Regresión simbólica como modelo de referencia («fórmula sugerida»)
- Estado: aceptado (2026-10-04)
- Contexto:
  - Perceptron entrena redes neuronales (SPEC §2, §8). En problemas tabulares donde la salida
    sale de una regla de cálculo, una red solo aproxima la función dentro del rango visto y no
    explica cuál es la regla.
  - Casos «Tabla 3» (`multiplo = 3 × numero`) y «Sensores» (`Salida = S1·(1+√S2)`). El mejor
    FT-Transformer de «Sensores» tiene R² 0,99998 en test, pero con S1 = 5 predice 2,24 en vez
    de 12,48.
  - El SPEC ya prevé un modelo de referencia fuera de las redes: LightGBM «solo para comparar»
    (§2 Tabular, §8).
  - Prueba de concepto con PyOperon, sobre 300 filas con los sensores independientes:
    - encontró `S1·(1+√S2)` en 9 s, con error máximo 2e-7, y extrapola exacto;
    - con los sensores colineales (S2 = √S1), encontró otra fórmula equivalente sobre la curva
      que falla fuera de ella.
- Decisión:
  - **Qué es.** Para regresión tabular, además de la red, Perceptron puede buscar una fórmula
    cerrada sobre los datos: la «fórmula sugerida». Es un modelo de referencia, como el baseline
    de LightGBM. No es una familia de arquitecturas: no entra en ArchSpec, HPO, export ONNX,
    serving ni monitoreo.
  - **Biblioteca:** `pyoperon` (MIT; ruedas nativas para Windows, Linux y macOS) en el extra `ml`.
    `sympy` (BSD) simplifica y traduce la fórmula; ya viene como dependencia de PyTorch.
    Descartadas:
    - PySR: exige un runtime de Julia de ~0,5 GB con compilación en la primera corrida. Complica
      el instalador de escritorio y entornos corporativos.
    - gplearn: menor calidad.
  - **Alcance v1:**
    - Regresión con un target y solo columnas numéricas de entrada (hasta 20, según el esquema
      del dataset).
    - Valores en unidades reales, sin el escalado del pipeline: la fórmula se lee directo.
    - Filas con nulos fuera; hasta 10 000 filas de train por muestreo.
    - Operadores: + − × ÷, constantes, √ y x² (exp y log desestabilizaban la búsqueda en la
      prueba de concepto; quedan para una versión con más tiempo). Tope de longitud y de tiempo
      configurables (por defecto 60 s). Semilla fija: mismos datos y parámetros, misma fórmula.
  - **Uso de los datos:** búsqueda en train, elección del frente de Pareto (error vs. complejidad)
    con validación y métricas finales en test. Es el mismo criterio que la evaluación de un run:
    el test sellado no participa de ninguna decisión.
  - **Producto:**
    - Job `202` desde Experimentos: «Buscar fórmula».
    - Resultado guardado como entidad `SymbolicFit` del proyecto: fórmula simplificada, frente
      de Pareto, métricas en validación y test comparadas con el mejor run, y gráfico real vs.
      predicho.
    - Se prueba con valores nuevos y se copia como Python o fórmula de Excel.
  - **Avisos:**
    - Si hay entradas muy correlacionadas entre sí, la fórmula puede no ser única: se avisa
      (caso «Sensores»).
    - Si el R² en validación queda bajo, se informa que no se encontró una fórmula simple.
  - **LLM:** no interviene en v1. Que el LLM proponga hipótesis de fórmula queda para después.
- Consecuencias:
  - Se agrega una dependencia nativa (C++). Su licencia y la de lo que empaqueta pasan por el job
    de licencias del CI antes de distribuir.
  - El tiempo de CPU queda acotado por el tope del job. No usa GPU ni LLM.
  - Clasificación, entradas categóricas y usar la fórmula como modelo desplegable quedan fuera.
    Si se piden, van en otro ADR.
  - Contrato: endpoints y entidad nuevos (OpenAPI → TS regenerado).
- Alternativas consideradas:
  - **Modelo de primera clase** (ArchSpec, HPO, export, serving): 2–3 semanas. Toca decisiones
    vinculantes del SPEC. Se evalúa después de usar la v1.
  - **No hacer nada** y recomendar un modelo lineal: no cubre reglas no lineales como la de
    «Sensores».
