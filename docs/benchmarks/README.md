# Benchmark O2 (SPEC §1.3, §15.4)

**Objetivo O2:** el mejor modelo del ciclo autónomo queda dentro del 5 % de la mejor
configuración manual conocida.

```bash
uv run perceptron bench run --datasets adult,fashion,fsdd --out bench-report --max-cost 2
```

o, en CI, `gh workflow run llm.yml -f benchmark=true` (necesita `ANTHROPIC_API_KEY`).

| Dataset | Modalidad | Licencia | Métrica | Referencia manual |
|---|---|---|---|---|
| Adult (UCI), 6000 filas | tabular | CC BY 4.0 | ROC-AUC | reglas + `lr=1e-3`, 30 épocas |
| Fashion-MNIST, 150 imágenes × 10 clases | imagen | MIT | accuracy | reglas + `lr=1e-3`, 15 épocas |
| Free Spoken Digit Dataset, 50 clips × 10 dígitos | audio | CC BY-SA 4.0 | accuracy | reglas + `lr=2e-3`, 30 épocas |

- Los datos se descargan en el job (no se versionan) y se verifica su SHA-256 cuando está
  fijado en `perceptron/benchmark/datasets.py` (el primer run informa el hash a fijar).
- FSDD reemplaza a Speech Commands (2,3 GB) para que el job quepa en un runner estándar.
- La referencia es una configuración manual fija y documentada, no el estado del arte:
  mide si el agente iguala lo que haría una persona con experiencia con el mismo presupuesto.
- Cada corrida deja `benchmark-<fecha>.md` y `benchmark.json` como artefacto; los resultados
  por release se copian a esta carpeta.
