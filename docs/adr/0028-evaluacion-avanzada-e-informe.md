# ADR-0028: Evaluación avanzada e informe con marca (Capa 4b)
- Estado: aceptado
- Fecha: 2026-09-28
- Contexto: RF-EVL-02..06 piden explicabilidad (SHAP en tabular, IG/Grad-CAM en imagen), análisis de errores, fairness (el SPEC nombra Fairlearn), robustez e informe PDF/HTML con la marca Preteco. El Engine corre en el desktop con un runtime embebido (ADR-0026) y en el servidor: las dependencias no deben exigir librerías del sistema ni forzar a bajar versiones de numpy o torch.
- Decisión:
  - **Explicabilidad con Captum** (BSD, puro PyTorch):
    - tabular: valores de Shapley por muestreo (ShapleyValueSampling), una feature por columna del pipeline, con una línea base típica (mediana y moda de validación), en versión global y local;
    - imagen: Integrated Gradients respecto de la clase predicha, como mapa de calor superpuesto (lima, marca).

    Se descarta `shap`: arrastra numba/llvmlite, que suelen ir atrasados respecto de numpy y agregan binarios al runtime del desktop. Los valores de Shapley son el mismo concepto.
  - **Fairness nativa**, con las definiciones de Fairlearn: paridad demográfica = máx − mín de la tasa de selección; igualdad de oportunidades = máx(diferencia de TPR, diferencia de FPR). Se evita agregar pandas solo para estas métricas. Los atributos numéricos continuos se agrupan por cuartiles. Las alertas saltan con un umbral de 0,1, configurable.
  - **Análisis de errores:** sobre las predicciones por muestra que ahora guarda la evaluación final (`evaluation/predictions.parquet`). Incluye:
    - slices de bajo rendimiento, por valores o cuartiles de las columnas originales;
    - confusiones;
    - confident learning simplificado para los posibles errores de etiqueta;
    - explorador de mal predichos.
  - **Robustez:** perturbaciones determinísticas a tres severidades sobre el test.
    - tabular: ruido, categorías cambiadas y faltantes;
    - imagen: ruido, desenfoque y JPEG.

    Texto, audio y series quedan para una iteración posterior.
  - **Informe:**
    - HTML autocontenido (Titillium Web embebida en base64, licencia OFL incluida; gráficos en SVG);
    - PDF con reportlab (BSD), en Helvetica: no depende de Pango/GTK, a diferencia de WeasyPrint, y funciona igual en Windows y Linux;
    - Markdown.

    El Markdown del LLM se renderiza sin HTML crudo (§13.2).
  - **Límites:** los modelos de código experto no se analizan en el proceso del Engine (ADR-0025). Estos análisis se agregarán al sandbox más adelante.
- Consecuencias: el PDF no usa la tipografía de marca (reportlab necesita TTF y el paquete de fuentes trae WOFF2). Se puede sumar una TTF de Titillium Web (OFL) si Marketing la provee. El extra `eval` suma captum, reportlab y markdown-it-py.
- Alternativas consideradas: SHAP + Fairlearn (dependencias pesadas y frágiles); WeasyPrint (dependencias del sistema); imprimir a PDF desde el navegador (no sirve para la API ni la CLI).
