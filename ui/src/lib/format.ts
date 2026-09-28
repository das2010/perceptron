/** Formatos según el locale (SPEC §11.2: nada de números/fechas con formato fijo). */
export function formatDate(value: string | null | undefined, locale: string): string {
  if (!value) return "";
  const d = new Date(value);
  return Number.isNaN(d.getTime())
    ? ""
    : d.toLocaleString(locale, { dateStyle: "medium", timeStyle: "short" });
}

export function formatNumber(value: number | null | undefined, locale: string, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString(locale, { maximumFractionDigits: digits });
}

export function formatPercent(value: number | null | undefined, locale: string): string {
  if (value === null || value === undefined) return "—";
  return value.toLocaleString(locale, { style: "percent", maximumFractionDigits: 1 });
}
