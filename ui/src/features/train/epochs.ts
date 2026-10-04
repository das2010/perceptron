/** Presupuesto de tiempo para el default de épocas: el estudio entero, a la estimación del motor. */
export const EPOCH_BUDGET_S = 20 * 60;
export const DEFAULT_EPOCHS = 15;
const MAX_EPOCHS = 500;

function proposedEpochs(spec: unknown): number | null {
  const raw = (spec as { training?: { epochs?: unknown } } | null | undefined)?.training?.epochs;
  if (typeof raw === "number") return raw;
  const def = (raw as { default?: unknown } | null | undefined)?.default;
  return typeof def === "number" ? def : null;
}

/** Épocas por trial al elegir una arquitectura: las que ella propone, si el estudio entra en
 * `EPOCH_BUDGET_S` según el tiempo estimado por época; si no, lo que entre (nunca menos que
 * el default de siempre). Antes eran 15 fijas: una regresión lineal que pedía 80 y tarda
 * milisegundos por época quedaba sin converger (caso «Tabla 3»). */
export function suggestEpochs(
  spec: unknown,
  epochTimeS: number | null | undefined,
  trials: number,
): number {
  const proposed = proposedEpochs(spec);
  if (!proposed || !epochTimeS || epochTimeS <= 0) return DEFAULT_EPOCHS;
  const affordable = Math.floor(EPOCH_BUDGET_S / (Math.max(trials, 1) * epochTimeS));
  const floor = Math.min(proposed, DEFAULT_EPOCHS);
  return Math.min(MAX_EPOCHS, Math.max(floor, Math.min(proposed, affordable)));
}
