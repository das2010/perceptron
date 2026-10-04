/**
 * Umbral de decisión por costo de los errores (ADR-0040, fase 2). Se eligió en validación con
 * el costo de la ficha del caso; acá se muestra su efecto en el test frente al umbral del 50 %.
 */
import { useTranslation } from "react-i18next";

import { Card, CardTitle, Table, Td, Th } from "@/components/ui";
import type { EvaluationReport } from "@/lib/api/hooks";
import { formatNumber, formatPercent } from "@/lib/format";

type Summary = {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  cost: number;
};
type CostThresholdInfo = {
  positive_class: string;
  worse: "false_negative" | "false_positive";
  ratio: number;
  threshold: number;
  test: Summary;
  test_at_50: Summary;
};

export function CostThreshold({ report }: { report: EvaluationReport }) {
  const { t, i18n } = useTranslation();
  const info = (report.details as Record<string, unknown> | undefined)?.cost_threshold as
    CostThresholdInfo | undefined;
  if (!info) return null;
  const lang = i18n.language;
  const rows: [string, Summary][] = [
    [t("run.cost.atThreshold", { threshold: formatNumber(info.threshold, lang, 3) }), info.test],
    [t("run.cost.at50"), info.test_at_50],
  ];
  return (
    <Card>
      <CardTitle>{t("run.cost.title")}</CardTitle>
      <p className="mb-3 text-sm text-muted">
        {t(`run.cost.explain_${info.worse}`, {
          ratio: formatNumber(info.ratio, lang, 2),
          positive: info.positive_class,
        })}
      </p>
      <Table>
        <thead>
          <tr>
            <Th>{t("run.cost.threshold")}</Th>
            <Th>{t("run.cost.recall")}</Th>
            <Th>{t("run.cost.precision")}</Th>
            <Th>{t("run.cost.missed")}</Th>
            <Th>{t("run.cost.falseAlarms")}</Th>
            <Th>{t("run.cost.cost")}</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, s]) => (
            <tr key={label}>
              <Td className="font-semibold">{label}</Td>
              <Td>{formatPercent(s.recall, lang)}</Td>
              <Td>{formatPercent(s.precision, lang)}</Td>
              <Td>{s.fn}</Td>
              <Td>{s.fp}</Td>
              <Td>{formatNumber(s.cost, lang, 1)}</Td>
            </tr>
          ))}
        </tbody>
      </Table>
    </Card>
  );
}
