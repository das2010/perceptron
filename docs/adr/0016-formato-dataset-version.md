# ADR-0016: Formato en disco de DatasetVersion
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: RF-ING-07 (versiones inmutables con hash), RF-ING-09 (hasta ~10 GB sin cargar todo en memoria) y RF-MON-07 (diff y linaje).
- Decisión: Cada versión es un directorio `projects/<id>/datasets/<content_hash>/` con: `table.parquet` (tabular, escrito por Polars) o `index.parquet` + `files/` (imágenes, hardlinks del origen o copia si está en otro volumen); `schema.json` (tipos semánticos y target); columna `__split__` (`train`/`val`/`test` o fold) y `manifest.json`. `content_hash` es el sha256 del manifiesto canónico (`storage.manifest`), calculado sobre los datos materializados (no sobre `manifest.json` ni metadatos). Re-ingestar los mismos datos con el mismo split produce el mismo hash. El directorio no se modifica nunca: un cambio de split o de datos crea una versión hija (`parent_id`).
- Consecuencias: lectura por lotes con Polars/Arrow; las imágenes se sirven desde disco. El hash incluye el split, así que la reproducibilidad (O3) cubre la partición.
- Alternativas consideradas: guardar solo referencias a los archivos originales (no inmutable); DVC como formato nativo (se ofrecerá como export, RF-MON-07).
