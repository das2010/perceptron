import { Link, useSearch } from "@tanstack/react-router";
import { useCallback, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { EChart } from "@/components/charts/EChart";
import { Badge, Card, CardTitle, EmptyState, ErrorNote, Spinner, Table, Td, Th } from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { useJob, useRuns, type Run } from "@/lib/api/hooks";
import { useEngineSocket } from "@/lib/api/ws";
import { formatDate, formatNumber } from "@/lib/format";

interface JobMessage {
  job_id?: string;
  kind?: string;
  data?: { run_id?: string; epoch?: number; metrics?: Record<string, number>; status?: string };
}

type Curves = Record<string, { epoch: number; value: number }[]>;

const STATUS_TONE = {
  queued: "neutral",
  running: "brand",
  succeeded: "ok",
  failed: "bad",
  cancelled: "warn",
  paused: "warn",
} as const;

/** Curvas en vivo de un estudio (RF-TRN-06) por el WebSocket del job. */
function LiveStudy({ jobId, metric }: { jobId: string; metric: string }) {
  const { t } = useTranslation();
  const job = useJob(jobId);
  const [curves, setCurves] = useState<Curves>({});

  const onMessage = useCallback((msg: JobMessage) => {
    const d = msg.data;
    if (msg.kind !== "epoch" || !d?.run_id || d.epoch === undefined || !d.metrics) return;
    const value = d.metrics[metric];
    if (value === undefined) return;
    const runId = d.run_id;
    const point = { epoch: d.epoch, value };
    setCurves((prev) => ({ ...prev, [runId]: [...(prev[runId] ?? []), point] }));
  }, [metric]);

  const connected = useEngineSocket<JobMessage>(`/api/v1/jobs/${jobId}`, onMessage);

  const option = useMemo(
    () => ({
      legend: { type: "scroll", bottom: 0 },
      xAxis: { type: "value", name: t("experiments.epoch") },
      yAxis: { type: "value", name: metric, scale: true },
      series: Object.entries(curves).map(([runId, pts]) => ({
        name: runId.split("-").at(-1),
        type: "line",
        showSymbol: false,
        data: pts.map((p) => [p.epoch + 1, p.value]),
      })),
    }),
    [curves, metric, t],
  );

  const status = job.data?.status ?? "queued";
  return (
    <Card>
      <CardTitle className="flex items-center gap-2">
        {t("experiments.live")}
        <Badge tone={STATUS_TONE[status]}>{t(`status.${status}`)}</Badge>
        {connected && <span className="text-xs text-muted">{t("experiments.connected")}</span>}
      </CardTitle>
      {Object.keys(curves).length === 0 ? (
        <Spinner label={t("experiments.waiting")} />
      ) : (
        <EChart option={option} label={t("experiments.curves")} />
      )}
      <ErrorNote error={job.data?.error ? new Error(String((job.data.error as { message?: string }).message ?? "")) : null} />
    </Card>
  );
}

function metricCols(runs: Run[]): string[] {
  const all = new Set<string>();
  for (const r of runs) for (const k of Object.keys(r.metrics ?? {})) if (k.startsWith("val_")) all.add(k);
  return [...all].sort((a, b) => (a === "val_loss" ? -1 : b === "val_loss" ? 1 : a.localeCompare(b))).slice(0, 4);
}

export function ExperimentsPage() {
  const { t, i18n } = useTranslation();
  const projectId = useProjectId();
  const { job } = useSearch({ from: "/projects/$projectId/experiments" });
  const jobState = useJob(job);
  const running = jobState.data && !["succeeded", "failed", "cancelled"].includes(jobState.data.status);
  const { data, isPending, error } = useRuns(projectId, running ? 4000 : undefined);
  const runs = useMemo(() => [...(data ?? [])].reverse(), [data]);
  const cols = metricCols(runs);

  return (
    <div className="space-y-6">
      {job && <LiveStudy jobId={job} metric="val_loss" />}
      <Card>
        <CardTitle>{t("experiments.runs")}</CardTitle>
        {isPending && <Spinner />}
        <ErrorNote error={error} />
        {!isPending && runs.length === 0 && <EmptyState>{t("experiments.empty")}</EmptyState>}
        {runs.length > 0 && (
          <Table>
            <thead>
              <tr>
                <Th>{t("experiments.run")}</Th>
                <Th>{t("experiments.status")}</Th>
                {cols.map((c) => <Th key={c}>{c}</Th>)}
                <Th>{t("experiments.started")}</Th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <Td>
                    <Link
                      to="/projects/$projectId/runs/$runId"
                      params={{ projectId, runId: r.id }}
                      className="font-mono text-xs underline"
                    >
                      {r.id.split("-").at(-1)}
                    </Link>
                  </Td>
                  <Td>
                    <Badge tone={STATUS_TONE[r.status]}>{t(`status.${r.status}`)}</Badge>
                  </Td>
                  {cols.map((c) => (
                    <Td key={c}>{formatNumber(r.metrics?.[c], i18n.language)}</Td>
                  ))}
                  <Td className="text-xs text-muted">{formatDate(r.started_at, i18n.language)}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}
