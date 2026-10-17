/** Etiquetas legibles de los datos del caso. */
import type { Schemas } from "@/lib/api/hooks";

type Facts = Schemas["Facts"];

export function fieldLabel(t: (k: string) => string, name: string, facts: Facts): string {
  if (name === "success_target")
    return t(
      facts.problem === "category"
        ? "journey.fields.success_target_category"
        : "journey.fields.success_target_value",
    );
  return t(`journey.fields.${name}`);
}

export function factValue(t: (k: string) => string, value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return t(value ? "journey.options.yes" : "journey.options.no");
  const key = `journey.options.${String(value)}`;
  const label = t(key);
  return label === key ? String(value) : label;
}
