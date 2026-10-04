import { describe, expect, it } from "vitest";

import { DEFAULT_EPOCHS, suggestEpochs } from "./epochs";

const spec = (epochs: unknown) => ({ training: { epochs } });

describe("épocas sugeridas al elegir una arquitectura", () => {
  it("una arquitectura barata usa las épocas que propone (caso «Tabla 3»)", () => {
    // Regresión lineal: pide 80 y tarda milisegundos por época.
    expect(suggestEpochs(spec({ hp: "epochs", default: 80 }), 0.02, 10)).toBe(80);
    expect(suggestEpochs(spec(60), 1, 10)).toBe(60);
  });

  it("si no entra en el presupuesto de tiempo, usa lo que entra, nunca menos que antes", () => {
    expect(suggestEpochs(spec(100), 4, 10)).toBe(30); // 20 min / (10 × 4 s)
    expect(suggestEpochs(spec(40), 30, 10)).toBe(DEFAULT_EPOCHS); // red pesada: como siempre
  });

  it("sin estimación ni épocas propuestas, el default de siempre", () => {
    expect(suggestEpochs(spec(80), null, 10)).toBe(DEFAULT_EPOCHS);
    expect(suggestEpochs({}, 0.1, 10)).toBe(DEFAULT_EPOCHS);
    expect(suggestEpochs(spec(5), 0.1, 10)).toBe(5); // si pide menos, se respeta
  });
});
