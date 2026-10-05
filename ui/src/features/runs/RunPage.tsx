import { useParams } from "@tanstack/react-router";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { EChart } from "@/components/charts/EChart";
import {
  AiSuggestion,
  Badge,
  Button,
  Card,
  CardTitle,
  ErrorNote,
  Spinner,
  Table,
  Td,
  Th,
} from "@/components/ui";
import {
  useRunHistory,
  useDiagnosis,
  useEvaluate,
  useEvaluation,
  useExportReport,
  useRegister,
  useReport,
  useOpenInMlflow,
  useRun,
  type EvaluationReport,
} from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

import { AnalysisPanel } from "./AnalysisPanel";
import { CostThreshold } from "./CostThreshold";
import { ExportPanel } from "./ExportPanel";
import { Improvements } from "./Improvements";
import { PlaygroundPanel } from "./PlaygroundPanel";

/** Error del worker guardado en `Run.diagnosis.error` (tipo, mensaje, pista). */
function formatRunError(diagnosis: unknown): string {
  const error = (diagnosis as { error?: unknown } | null | undefined)?.error;
  if (!error) return "";
  if (typeof error === "string") return error;
  const e = error as {
    code?: string;
    type?: string;
    message?: string;
    hint?: string;
    log_tail?: string;
  };
  const parts = [e.code ?? e.type, e.message, e.hint, e.log_tail].filter(Boolean);
  return parts.length ? parts.join("\n") : JSON.stringify(error, null, 2);
}

function Curves({ runId }: { runId: string }) {
  const { t } = useTranslation();
  const { data } = useRunHistory(runId);
  const history = useMemo(() => data ?? [], [data]);
  const option = useMemo(() => {
    const keys = ["train_loss", "val_loss"].filter((k) => history.some((h) => k in h));
    return {
      legend: { bottom: 0 },
      xAxis: { type: "value", name: t("experiments.epoch") },
      yAxis: { type: "value", scale: true },
      series: keys.map((k) => ({
        name: k,
        type: "line",
        showSymbol: false,
        data: history.filter((h) => k in h).map((h) => [(h.epoch ?? 0) + 1, h[k]]),
      })),
    };
  }, [history, t]);
  if (!history.length) return null;
  return (
    <Card>
      <CardTitle>{t("run.curves")}</CardTitle>
      <EChart option={option} label={t("run.curves")} />
    </Card>
  );
}

function Confusion({ report }: { report: EvaluationReport }) {
  const { t, i18n } = useTranslation();
  const cls = report.classification;
  const option = useMemo(() => {
    if (!cls) return {};
    const data = cls.confusion_matrix.flatMap((row, i) => row.map((v, j) => [j, i, v]));
    const max = Math.max(1, ...cls.confusion_matrix.flat());
    return {
      tooltip: { position: "top" },
      xAxis: { type: "category", data: cls.labels, name: t("run.predicted") },
      yAxis: { type: "category", data: cls.labels, name: t("run.actual"), inverse: true },
      visualMap: {
        min: 0,
        max,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        show: false,
      },
      series: [{ type: "heatmap", data, label: { show: true } }],
    };
  }, [cls, t]);
  if (!cls) return null;
  return (
    <Card>
      <CardTitle>{t("run.confusion")}</CardTitle>
      <EChart option={option} height={320} label={t("run.confusion")} />
      <Table>
        <thead>
          <tr>
            <Th>{t("run.class")}</Th>
            <Th>precision</Th>
            <Th>recall</Th>
            <Th>f1</Th>
            <Th>{t("run.support")}</Th>
          </tr>
        </thead>
        <tbody>
          {cls.per_class.map((c) => (
            <tr key={c.label}>
              <Td className="font-semibold">{c.label}</Td>
              <Td>{formatNumber(c.precision, i18n.language, 3)}</Td>
              <Td>{formatNumber(c.recall, i18n.language, 3)}</Td>
              <Td>{formatNumber(c.f1, i18n.language, 3)}</Td>
              <Td>{c.support}</Td>
            </tr>
          ))}
        </tbody>
      </Table>
    </Card>
  );
}

export function RunPage() {
  const { t, i18n } = useTranslation();
  const { projectId, runId } = useParams({ from: "/projects/$projectId/runs/$runId" });
  const run = useRun(runId);
  const evaluation = useEvaluation(runId);
  const evaluate = useEvaluate(runId);
  const register = useRegister(runId, projectId);
  const done = run.data?.status === "succeeded";
  const diagnosis = useDiagnosis(runId, done);
  const report = useReport(runId);
  const exported = useExportReport(runId, done);
  const mlflow = useOpenInMlflow(runId);

  if (run.isPending) return <Spinner />;
  if (run.error || !run.data) return <ErrorNote error={run.error} />;
  const r = run.data;
  const evaluated = evaluation.data;
  const d = diagnosis.data;

  return (
    <div className="space-y-4">
      <Card>
        <CardTitle className="flex items-center gap-2">
          {t("run.title")} <span className="font-mono text-xs">{r.id}</span>
          <Badge>{t(`status.${r.status}`)}</Badge>
          {r.mlflow_run_id && (
            <Button
              variant="ghost"
              size="sm"
              className="ml-auto"
              loading={mlflow.isPending}
              onClick={() => mlflow.mutate()}
              title={t("run.mlflowHint")}
            >
              {t("run.openMlflow")}
            </Button>
          )}
        </CardTitle>
        <ErrorNote error={mlflow.error} />
        <dl className="grid gap-2 text-sm sm:grid-cols-3">
          {Object.entries(r.metrics ?? {})
            .filter(([k]) => k.startsWith("val_"))
            .map(([k, v]) => (
              <div key={k}>
                <dt className="text-muted">{k}</dt>
                <dd className="font-semibold">{formatNumber(v, i18n.language)}</dd>
              </div>
            ))}
        </dl>
        {Object.keys(r.hyperparams ?? {}).length > 0 && (
          <p className="mt-3 text-xs text-muted">
            {Object.entries(r.hyperparams ?? {})
              .map(([k, v]) => `${k}=${String(v)}`)
              .join(" · ")}
          </p>
        )}
      </Card>

      {r.status === "failed" && (
        <Card className="border-bad">
          <CardTitle className="text-bad">{t("run.failed")}</CardTitle>
          <pre className="whitespace-pre-wrap text-xs" data-testid="run-error">
            {formatRunError(r.diagnosis)}
          </pre>
        </Card>
      )}

      <Curves runId={runId} />

      {d && (
        <AiSuggestion title={t("run.diagnosis")}>
          <p>{d.summary}</p>
          <ul className="mt-2 list-disc pl-5">
            {(d.problems ?? []).map((p, i) => (
              <li key={i}>
                <strong>{p.kind}</strong> — {p.evidence}
              </li>
            ))}
            {(d.actions ?? []).map((a, i) => (
              <li key={`a${i}`}>→ {a.rationale}</li>
            ))}
          </ul>
          {d.origin !== "llm" && <p className="mt-2 text-xs text-muted">{t("run.byRules")}</p>}
          <Improvements
            runId={runId}
            projectId={projectId}
            datasetVersionId={r.dataset_version_id}
            pipelineId={r.pipeline_id}
          />
        </AiSuggestion>
      )}

      <Card>
        <CardTitle>{t("run.evaluation")}</CardTitle>
        {!evaluated && done && (
          <>
            <p className="mb-3 text-sm text-muted">{t("run.sealedHint")}</p>
            <Button loading={evaluate.isPending} onClick={() => evaluate.mutate()}>
              {t("run.evaluate")}
            </Button>
          </>
        )}
        <ErrorNote error={evaluate.error} />
        {evaluated && (
          <>
            <dl className="grid gap-2 text-sm sm:grid-cols-4">
              {Object.entries(evaluated.metrics).map(([k, v]) => (
                <div key={k}>
                  <dt className="text-muted">{k}</dt>
                  <dd className="font-semibold">{formatNumber(v, i18n.language)}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-4 flex flex-wrap gap-2">
              <Button
                variant="secondary"
                loading={register.isPending}
                disabled={register.isSuccess}
                onClick={() => register.mutate()}
              >
                {register.isSuccess ? t("run.registered") : t("run.register")}
              </Button>
              <Button variant="ai" loading={report.isPending} onClick={() => report.mutate()}>
                {t("run.report")}
              </Button>
            </div>
            <ErrorNote error={register.error ?? report.error} />
          </>
        )}
      </Card>

      {evaluated && <Confusion report={evaluated} />}
      {evaluated && <CostThreshold report={evaluated} />}
      {evaluated && (
        <AnalysisPanel
          runId={runId}
          datasetVersionId={r.dataset_version_id}
          classes={evaluated.classification?.labels ?? []}
        />
      )}

      {done && <ExportPanel runId={runId} />}
      {done &&
        exported.data?.artifacts.some(
          (a) => a.format === "onnx" && !a.error && a.verification?.passed,
        ) && (
          <PlaygroundPanel
            runId={runId}
            datasetVersionId={r.dataset_version_id}
            report={exported.data}
          />
        )}

      {report.data && (
        <Card>
          <CardTitle className="flex items-center gap-2">
            {report.data.title}
            {report.data.origin === "llm" && <Badge tone="brand">IA</Badge>}
          </CardTitle>
          <pre className="whitespace-pre-wrap font-sans text-sm">{report.data.markdown}</pre>
        </Card>
      )}
    </div>
  );
}
