/** Visualizaciones del HPO (RF-HPO-06): historia, importancia de hiperparámetros,
 * coordenadas paralelas y frente de Pareto de un estudio. */
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { EChart } from "@/components/charts/EChart";
import { Card, CardTitle, EmptyState, Field, Select, Spinner } from "@/components/ui";
import { type Run, type StudyAnalysis, useStudyAnalysis } from "@/lib/api/hooks";

function History({ data }: { data: StudyAnalysis }) {
  const { t } = useTranslation();
  const metric = data.objectives[0]?.metric ?? "";
  const option = useMemo(
    () => ({
      tooltip: { trigger: "axis" },
      legend: { bottom: 0 },
      xAxis: { type: "value", name: t("hpo.trial"), minInterval: 1 },
      yAxis: { type: "value", scale: true, name: metric },
      series: [
        {
          name: t("hpo.trialValue"),
          type: "scatter",
          data: data.trials.filter((p) => p.values[0] != null).map((p) => [p.number, p.values[0]]),
        },
        {
          name: t("hpo.bestSoFar"),
          type: "line",
          step: "end",
          showSymbol: false,
          data: data.trials
            .filter((p) => p.best_so_far != null)
            .map((p) => [p.number, p.best_so_far]),
        },
      ],
    }),
    [data, metric, t],
  );
  return <EChart option={option} label={t("hpo.history")} />;
}

function Importance({ data }: { data: StudyAnalysis }) {
  const { t } = useTranslation();
  const entries = Object.entries(data.importance).sort((a, b) => a[1] - b[1]);
  const option = useMemo(
    () => ({
      grid: { left: 120 },
      xAxis: { type: "value", max: 1 },
      yAxis: { type: "category", data: entries.map(([k]) => k) },
      series: [{ type: "bar", data: entries.map(([, v]) => Number(v.toFixed(3))) }],
    }),
    [entries],
  );
  if (!entries.length) return <p className="text-xs text-muted">{t("hpo.noImportance")}</p>;
  return <EChart option={option} label={t("hpo.importance")} />;
}

function Parallel({ data }: { data: StudyAnalysis }) {
  const { t } = useTranslation();
  const metric = data.objectives[0]?.metric ?? "";
  const option = useMemo(() => {
    const dims = [...data.params, metric];
    const categories = (k: string) =>
      [
        ...new Set(data.trials.map((p) => p.params[k]).filter((v) => typeof v === "string")),
      ] as string[];
    return {
      parallelAxis: dims.map((k, i) => {
        const cats = i < data.params.length ? categories(k) : [];
        return cats.length
          ? { dim: i, name: k, type: "category", data: cats }
          : { dim: i, name: k };
      }),
      series: [
        {
          type: "parallel",
          lineStyle: { opacity: 0.5 },
          data: data.trials
            .filter((p) => p.values[0] != null)
            .map((p) => [...data.params.map((k) => p.params[k] ?? null), p.values[0]]),
        },
      ],
    };
  }, [data, metric]);
  if (!data.params.length) return null;
  return <EChart option={option} label={t("hpo.parallel")} />;
}

function Pareto({ data }: { data: StudyAnalysis }) {
  const { t } = useTranslation();
  const [a, b] = data.objectives;
  const option = useMemo(() => {
    const point = (p: StudyAnalysis["trials"][number]) => [p.values[0], p.values[1]];
    const full = data.trials.filter((p) => p.values[0] != null && p.values[1] != null);
    return {
      legend: { bottom: 0 },
      tooltip: {},
      xAxis: { type: "value", scale: true, name: a?.metric },
      yAxis: { type: "value", scale: true, name: b?.metric },
      series: [
        {
          name: t("hpo.dominated"),
          type: "scatter",
          data: full.filter((p) => !p.pareto).map(point),
        },
        {
          name: t("hpo.pareto"),
          type: "scatter",
          symbolSize: 12,
          data: full.filter((p) => p.pareto).map(point),
        },
      ],
    };
  }, [data, a, b, t]);
  if (!a || !b) return null;
  return <EChart option={option} label={t("hpo.pareto")} />;
}

export function StudyInsights({ runs }: { runs: Run[] }) {
  const { t } = useTranslation();
  const studies = useMemo(() => {
    const count = new Map<string, number>();
    for (const r of runs) if (r.study_id) count.set(r.study_id, (count.get(r.study_id) ?? 0) + 1);
    return [...count.entries()].filter(([, n]) => n >= 2).map(([id]) => id);
  }, [runs]);
  const [chosen, setChosen] = useState<string | null>(null);
  const studyId = chosen ?? studies[0] ?? null;
  const analysis = useStudyAnalysis(studyId);
  if (!studies.length) return null;
  const data = analysis.data;

  return (
    <Card>
      <CardTitle>{t("hpo.title")}</CardTitle>
      {studies.length > 1 && (
        <Field label={t("hpo.study")}>
          <Select
            value={studyId ?? ""}
            onChange={(e) => setChosen(e.target.value)}
            className="w-72"
          >
            {studies.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </Field>
      )}
      {analysis.isPending && <Spinner />}
      {data && data.trials.length === 0 && <EmptyState>{t("hpo.empty")}</EmptyState>}
      {data && data.trials.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          <section>
            <h3 className="mb-1 text-sm font-semibold">{t("hpo.history")}</h3>
            <History data={data} />
          </section>
          <section>
            <h3 className="mb-1 text-sm font-semibold">{t("hpo.importance")}</h3>
            <Importance data={data} />
          </section>
          <section className="lg:col-span-2">
            <h3 className="mb-1 text-sm font-semibold">{t("hpo.parallel")}</h3>
            <Parallel data={data} />
          </section>
          {data.objectives.length > 1 && (
            <section className="lg:col-span-2">
              <h3 className="mb-1 text-sm font-semibold">{t("hpo.pareto")}</h3>
              <Pareto data={data} />
            </section>
          )}
        </div>
      )}
    </Card>
  );
}
