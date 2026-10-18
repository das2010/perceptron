/** Etiquetas legibles del caso y respuestas rápidas de cada pregunta del asistente. */
import type { TFunction } from "i18next";

import type { Schemas } from "@/lib/api/hooks";

type Facts = Schemas["Facts"];
type T = TFunction;

export function fieldLabel(t: T, name: string, facts: Facts): string {
  if (name === "success_target")
    return t(
      facts.problem === "category"
        ? "journey.fields.success_target_category"
        : "journey.fields.success_target_value",
    );
  return t(`journey.fields.${name}`);
}

export function factValue(t: T, value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return t(value ? "journey.options.yes" : "journey.options.no");
  const key = `journey.options.${String(value)}`;
  const label = t(key);
  return label === key ? String(value) : label;
}

/** Una respuesta rápida: el texto que «dice» la persona y el dato que confirma. */
export interface QuickReply {
  label: string;
  facts?: Facts;
}

const CHOICES: Record<string, string[]> = {
  data_kind: ["table", "images", "text", "audio", "other"],
  error_costs: ["false_negative_worse", "false_positive_worse", "symmetric"],
};
/** Lo que se puede obtener según el tipo de datos (lo que el motor resuelve de punta a punta). */
const PROBLEMS: Record<string, string[]> = {
  table: ["category", "value", "forecast", "anomaly", "rule"],
  images: ["category", "locate", "segment", "read_text"],
  text: ["category"],
  audio: ["category"],
};
/** Problemas cuyo objetivo es un mínimo de aciertos (el resto, un máximo de error). */
const HITS = new Set(["category", "anomaly", "locate", "segment", "read_text"]);
const BOOLEANS = new Set(["labels_available", "extrapolate", "grouped"]);

/** Respuestas rápidas para el dato que pregunta el asistente (sin depender del LLM). */
export function quickReplies(t: T, field: string, facts: Facts): QuickReply[] {
  if (field === "problem")
    return (PROBLEMS[facts.data_kind ?? ""] ?? PROBLEMS.table ?? []).map((v) => ({
      label: t(`journey.quick.problem.${v}`),
      facts: { problem: v } as Facts,
    }));
  if (field === "labels_available" && facts.problem === "anomaly")
    return [true, false].map((v) => ({
      label: t(`journey.quick.anomalyMarks.${v ? "yes" : "no"}`),
      facts: { labels_available: v },
    }));
  if (
    field === "labels_available" &&
    ["locate", "segment", "read_text"].includes(facts.problem ?? "")
  )
    return [true, false].map((v) => ({
      label: t(`journey.quick.photoMarks.${v ? "yes" : "no"}`),
      facts: { labels_available: v },
    }));
  if (field === "horizon")
    return [1, 4, 12].map((n) => ({
      label: t("journey.quick.horizon", { count: n }),
      facts: { horizon: n } as Facts,
    }));
  const choices = CHOICES[field];
  if (choices)
    return choices.map((v) => ({
      label: t(`journey.quick.${field}.${v}`),
      facts: { [field]: v } as Facts,
    }));
  if (BOOLEANS.has(field))
    return [true, false].map((v) => ({
      label: t(`journey.quick.${field}.${v ? "yes" : "no"}`),
      facts: { [field]: v } as Facts,
    }));
  if (field === "positive_class")
    return (facts.classes ?? []).map((c) => ({ label: c, facts: { positive_class: c } }));
  if (field === "success_target") {
    const category = HITS.has(facts.problem ?? "");
    return (category ? [70, 80, 90, 95] : [5, 10, 20]).map((n) => ({
      label: t(category ? "journey.quick.hits" : "journey.quick.error", {
        n,
        of: n % 10 === 0 ? n / 10 : n / 5,
        den: n % 10 === 0 ? 10 : 20,
      }),
      facts: { success_target: n },
    }));
  }
  return [];
}

/** «antiguedad_meses» → «Antiguedad meses»: los nombres de columna, legibles en formularios. */
export function humanize(name: string): string {
  const words = name.replace(/[_\-.]+/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : name;
}
