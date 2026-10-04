import type { Run } from "@/lib/api/hooks";

/** El mejor run de cada estudio (menor `val_loss`, el objetivo por defecto): sin esta marca,
 * en la tabla no se distinguía y se evaluaba cualquiera. */
export function bestRunIds(runs: Run[]): Set<string> {
  const best = new Map<string, Run>();
  for (const r of runs) {
    const loss = r.metrics?.val_loss;
    if (r.status !== "succeeded" || loss == null || !r.study_id) continue;
    const current = best.get(r.study_id);
    if (!current || loss < (current.metrics?.val_loss ?? Infinity)) best.set(r.study_id, r);
  }
  return new Set([...best.values()].map((r) => r.id));
}
