/** Operaciones puras sobre ArchSpec y PipelineSpec que usan los editores visuales. */

/** Nodo implícito de entrada del grafo (engine: `archspec.schema.INPUT_NODE`). */
export const INPUT_NODE = "input";

export interface HP {
  hp: string;
  default: unknown;
}
export interface SpecNode {
  id: string;
  block: string;
  params: Record<string, unknown>;
}
export interface Spec {
  name: string;
  modality: string;
  nodes: SpecNode[];
  edges: [string, string][];
  [key: string]: unknown;
}

export const isHp = (v: unknown): v is HP => typeof v === "object" && v !== null && "hp" in v;

/** Nombre de hiperparámetro válido para el engine (`^[a-z][a-z0-9_]*$`). */
export const hpName = (node: string, param: string) =>
  `${node}_${param}`
    .toLowerCase()
    .replace(/[^a-z0-9_]/g, "_")
    .replace(/^[^a-z]+/, "") || "hp";

/** Agrega un bloque al final de la cadena con un id único. */
export function addBlock(spec: Spec, block: string): { spec: Spec; id: string } {
  const base = (block.split(".").at(-1) ?? "bloque").toLowerCase().replace(/[^a-z0-9_]/g, "_");
  let id = base;
  for (let i = 2; spec.nodes.some((n) => n.id === id); i++) id = `${base}_${i}`;
  const last = spec.nodes.at(-1)?.id ?? INPUT_NODE;
  return {
    id,
    spec: {
      ...spec,
      nodes: [...spec.nodes, { id, block, params: {} }],
      edges: [...spec.edges, [last, id]],
    },
  };
}

/** Quita un bloque y reconecta predecesores con sucesores para no cortar la cadena. */
export function removeBlock(spec: Spec, id: string): Spec {
  const preds = spec.edges.filter(([, b]) => b === id).map(([a]) => a);
  const succs = spec.edges.filter(([a]) => a === id).map(([, b]) => b);
  const kept = spec.edges.filter(([a, b]) => a !== id && b !== id);
  const bridged = preds
    .flatMap((p) => succs.map((s) => [p, s] as [string, string]))
    .filter(([a, b]) => !kept.some(([x, y]) => x === a && y === b));
  return { ...spec, nodes: spec.nodes.filter((n) => n.id !== id), edges: [...kept, ...bridged] };
}

/** Mueve un elemento de una lista `delta` posiciones (reordenar pasos del pipeline). */
export function move<T>(items: T[], index: number, delta: number): T[] {
  const to = index + delta;
  if (to < 0 || to >= items.length) return items;
  const next = [...items];
  const [item] = next.splice(index, 1) as [T];
  next.splice(to, 0, item);
  return next;
}

/** Pasos tabulares del engine (`data/pipeline/steps.py`). */
export const STEP_KINDS = [
  "drop",
  "to_numeric",
  "date_features",
  "impute_numeric",
  "impute_categorical",
  "log1p",
  "scale",
  "ordinal",
  "one_hot",
] as const;
