/** Comparación de runs (SPEC §11.2): curvas superpuestas, métricas finales e hiperparámetros. */
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { EChart } from "@/components/charts/EChart";
import { Button, Card, CardTitle, Field, Select, Table, Td, Th } from "@/components/ui";
import { type Run, useRunHistories } from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

const short = (id: string) => id.split("-").at(-1) ?? id;

export function CompareRuns({ runs, onClear }: { runs: Run[]; onClear: () => void }) {
  const { t, i18n } = useTranslation();
  const histories = useRunHistories(runs.map((r) => r.id));
  const metrics = useMemo(() => {
    const keys = new Set<string>();
    for (const r of runs) for (const k of Object.keys(r.metrics ?? {})) keys.add(k);
    return [...keys].sort((a, b) =>
      a === "val_loss" ? -1 : b === "val_loss" ? 1 : a.localeCompare(b),
    );
  }, [runs]);
  const params = useMemo(() => {
    const keys = new Set<string>();
    for (const r of runs) for (const k of Object.keys(r.hyperparams ?? {})) keys.add(k);
    return [...keys].sort();
  }, [runs]);
  const [metric, setMetric] = useState("val_loss");
  const option = useMemo(
    () => ({
      legend: { bottom: 0 },
      tooltip: { trigger: "axis" },
      xAxis: { type: "value", name: t("experiments.epoch") },
      yAxis: { type: "value", scale: true, name: metric },
      series: runs.map((r, i) => ({
        name: short(r.id),
        type: "line",
        showSymbol: false,
        data: (histories[i]?.data ?? [])
          .filter((h) => metric in h)
          .map((h) => [(h.epoch ?? 0) + 1, h[metric]]),
      })),
    }),
    [runs, histories, metric, t],
  );
  const differs = (k: string) =>
    new Set(runs.map((r) => JSON.stringify(r.hyperparams?.[k]))).size > 1;

  return (
    <Card>
      <CardTitle className="flex items-center justify-between gap-2">
        {t("experiments.compareTitle", { count: runs.length })}
        <Button size="sm" variant="ghost" onClick={onClear}>
          {t("experiments.clearCompare")}
        </Button>
      </CardTitle>
      <Field label={t("experiments.compareMetric")}>
        <Select value={metric} onChange={(e) => setMetric(e.target.value)} className="w-56">
          {[...new Set(["val_loss", "train_loss", ...metrics])].map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </Select>
      </Field>
      <EChart option={option} label={t("experiments.compareTitle", { count: runs.length })} />
      <div className="mt-4 overflow-x-auto">
        <Table>
          <thead>
            <tr>
              <Th />
              {runs.map((r) => (
                <Th key={r.id} className="font-mono">
                  {short(r.id)}
                </Th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => (
              <tr key={m}>
                <Td className="font-semibold">{m}</Td>
                {runs.map((r) => (
                  <Td key={r.id}>{formatNumber(r.metrics?.[m], i18n.language)}</Td>
                ))}
              </tr>
            ))}
            {params.map((p) => (
              <tr key={p} className={differs(p) ? "bg-canvas" : undefined}>
                <Td className="text-muted">{p}</Td>
                {runs.map((r) => (
                  <Td key={r.id} className="font-mono text-xs">
                    {r.hyperparams?.[p] === undefined ? "—" : String(r.hyperparams[p])}
                  </Td>
                ))}
              </tr>
            ))}
          </tbody>
        </Table>
        <p className="mt-2 text-xs text-muted">{t("experiments.compareHint")}</p>
      </div>
    </Card>
  );
}
