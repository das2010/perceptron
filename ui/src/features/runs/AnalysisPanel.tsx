/**
 * Evaluación avanzada del run (RF-EVL-02..06): errores, explicación, equidad, robustez e
 * informe descargable. Todo se calcula sobre el test sellado ya evaluado.
 */
import { AlertTriangle, Download } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { EChart } from "@/components/charts/EChart";
import {
  Badge,
  Button,
  Card,
  CardTitle,
  ErrorNote,
  Field,
  Select,
  Spinner,
  Table,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Td,
  Th,
} from "@/components/ui";
import { downloadFromEngine } from "@/lib/api/download";
import {
  useErrorAnalysis,
  useExplanation,
  useFairness,
  useProfile,
  useRobustness,
} from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

function Errors({ runId }: { runId: string }) {
  const { t, i18n } = useTranslation();
  const q = useErrorAnalysis(runId, true);
  if (q.isPending) return <Spinner />;
  if (q.error || !q.data) return <ErrorNote error={q.error} />;
  const e = q.data;
  const featureCols = Object.keys(e.samples[0]?.features ?? {});
  return (
    <div className="space-y-4">
      <p className="text-sm">
        {t("analysis.errors.summary", {
          errors: e.num_errors,
          total: e.num_samples,
          metric: e.metric,
          value: formatNumber(e.overall, i18n.language),
        })}{" "}
        {e.label_issues > 0 && (
          <Badge tone="warn">{t("analysis.errors.labelIssues", { count: e.label_issues })}</Badge>
        )}
      </p>
      {e.slices.length > 0 && (
        <div>
          <h4 className="mb-1 font-semibold">{t("analysis.errors.slices")}</h4>
          <Table>
            <thead>
              <tr>
                <Th>{t("analysis.errors.column")}</Th>
                <Th>{t("analysis.errors.value")}</Th>
                <Th>{t("analysis.errors.support")}</Th>
                <Th>{e.metric}</Th>
              </tr>
            </thead>
            <tbody>
              {e.slices.map((s) => (
                <tr key={`${s.column}:${s.value}`}>
                  <Td>{s.column}</Td>
                  <Td>{s.value}</Td>
                  <Td>{s.support}</Td>
                  <Td className="text-bad">{formatNumber(s.metric, i18n.language)}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </div>
      )}
      {e.confusions.length > 0 && (
        <div>
          <h4 className="mb-1 font-semibold">{t("analysis.errors.confusions")}</h4>
          <ul className="text-sm">
            {e.confusions.map((c) => (
              <li key={`${c.actual}>${c.predicted}`}>
                {t("analysis.errors.confusion", {
                  actual: c.actual,
                  predicted: c.predicted,
                  count: c.count,
                })}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="max-h-96 overflow-auto">
        <h4 className="mb-1 font-semibold">{t("analysis.errors.samples")}</h4>
        <Table>
          <thead>
            <tr>
              <Th>{t("analysis.errors.actual")}</Th>
              <Th>{t("analysis.errors.predicted")}</Th>
              <Th>{t("analysis.errors.confidence")}</Th>
              {featureCols.map((c) => (
                <Th key={c}>{c}</Th>
              ))}
            </tr>
          </thead>
          <tbody>
            {e.samples.slice(0, 50).map((s) => (
              <tr key={s.row}>
                <Td>
                  {String(s.actual)}
                  {s.label_issue && (
                    <span title={t("analysis.errors.labelIssue")}>
                      <AlertTriangle className="ml-1 inline h-3 w-3 text-warn" aria-hidden="true" />
                    </span>
                  )}
                </Td>
                <Td>{String(s.predicted)}</Td>
                <Td>{s.confidence == null ? "—" : formatNumber(s.confidence, i18n.language, 2)}</Td>
                {featureCols.map((c) => (
                  <Td key={c} className="max-w-40 truncate text-xs">
                    {String(s.features?.[c] ?? "")}
                  </Td>
                ))}
              </tr>
            ))}
          </tbody>
        </Table>
      </div>
    </div>
  );
}

function Explanation({ runId }: { runId: string }) {
  const { t } = useTranslation();
  const [on, setOn] = useState(false);
  const q = useExplanation(runId, on);
  const option = useMemo(() => {
    const top = [...(q.data?.features ?? [])].slice(0, 15).reverse();
    return {
      grid: { left: 140, right: 20, top: 10, bottom: 20 },
      xAxis: { type: "value" },
      yAxis: { type: "category", data: top.map((f) => f.feature) },
      series: [
        { type: "bar", data: top.map((f) => f.importance), itemStyle: { color: "#334000" } },
      ],
    };
  }, [q.data]);
  if (!on)
    return (
      <div className="space-y-2">
        <p className="text-sm text-muted">{t("analysis.explain.hint")}</p>
        <Button onClick={() => setOn(true)}>{t("analysis.explain.compute")}</Button>
      </div>
    );
  if (q.isPending) return <Spinner label={t("analysis.computing")} />;
  if (q.error || !q.data) return <ErrorNote error={q.error} />;
  return (
    <div>
      <p className="mb-2 text-xs text-muted">
        {t("analysis.explain.method", { samples: q.data.samples })}
      </p>
      <EChart
        option={option}
        label={t("analysis.tabs.explain")}
        height={Math.max(200, 24 * Math.min(15, q.data.features.length))}
      />
    </div>
  );
}

function Fairness({
  runId,
  datasetVersionId,
  classes,
}: {
  runId: string;
  datasetVersionId: string;
  classes: string[];
}) {
  const { t, i18n } = useTranslation();
  const profile = useProfile(datasetVersionId);
  const fair = useFairness(runId);
  const columns = (profile.data?.columns ?? [])
    .map((c) => c.name)
    .filter((c) => c !== profile.data?.target?.name);
  const [attrs, setAttrs] = useState<string[]>([]);
  const [positive, setPositive] = useState<string>("");
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted">{t("analysis.fairness.hint")}</p>
      <div className="flex flex-wrap gap-3 text-sm">
        {columns.map((c) => (
          <label key={c} className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={attrs.includes(c)}
              onChange={() =>
                setAttrs((a) => (a.includes(c) ? a.filter((x) => x !== c) : [...a, c]))
              }
            />
            {c}
          </label>
        ))}
      </div>
      {classes.length > 2 && (
        <Field label={t("analysis.fairness.positive")}>
          <Select value={positive} onChange={(e) => setPositive(e.target.value)} className="w-48">
            <option value="">—</option>
            {classes.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </Select>
        </Field>
      )}
      <Button
        disabled={!attrs.length || (classes.length > 2 && !positive)}
        loading={fair.isPending}
        onClick={() => fair.mutate({ attributes: attrs, positive_class: positive || null })}
      >
        {t("analysis.fairness.compute")}
      </Button>
      <ErrorNote error={fair.error} />
      {fair.data?.map((f) => (
        <div key={f.attribute}>
          <h4 className="font-semibold">{f.attribute}</h4>
          {(f.alerts ?? []).map((a) => (
            <p key={a} role="alert" className="text-sm text-bad">
              <AlertTriangle className="mr-1 inline h-4 w-4" aria-hidden="true" />
              {a}
            </p>
          ))}
          <Table>
            <thead>
              <tr>
                <Th>{t("analysis.fairness.group")}</Th>
                <Th>{t("analysis.errors.support")}</Th>
                {f.task === "regression" ? (
                  <Th>MAE</Th>
                ) : (
                  <>
                    <Th>{t("analysis.fairness.selection", { cls: f.positive_class })}</Th>
                    <Th>accuracy</Th>
                    <Th>TPR</Th>
                    <Th>FPR</Th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {f.groups.map((g) => (
                <tr key={g.group}>
                  <Td>{g.group}</Td>
                  <Td>{g.support}</Td>
                  {f.task === "regression" ? (
                    <Td>{formatNumber(g.mae ?? null, i18n.language)}</Td>
                  ) : (
                    <>
                      <Td>{formatNumber(g.selection_rate ?? null, i18n.language, 3)}</Td>
                      <Td>{formatNumber(g.accuracy ?? null, i18n.language, 3)}</Td>
                      <Td>{formatNumber(g.tpr ?? null, i18n.language, 3)}</Td>
                      <Td>{formatNumber(g.fpr ?? null, i18n.language, 3)}</Td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </Table>
        </div>
      ))}
    </div>
  );
}

function Robustness({ runId }: { runId: string }) {
  const { t, i18n } = useTranslation();
  const [on, setOn] = useState(false);
  const q = useRobustness(runId, on);
  const option = useMemo(() => {
    const r = q.data;
    if (!r) return {};
    const kinds = [...new Set(r.results.map((x) => x.kind))];
    return {
      legend: { bottom: 0 },
      xAxis: { type: "category", data: ["0", "1", "2", "3"], name: t("analysis.robustness.level") },
      yAxis: { type: "value", scale: true, name: r.metric },
      series: kinds.map((k) => ({
        name: k,
        type: "line",
        data: [r.baseline, ...r.results.filter((x) => x.kind === k).map((x) => x.metric)],
      })),
    };
  }, [q.data, t]);
  if (!on)
    return (
      <div className="space-y-2">
        <p className="text-sm text-muted">{t("analysis.robustness.hint")}</p>
        <Button onClick={() => setOn(true)}>{t("analysis.robustness.compute")}</Button>
      </div>
    );
  if (q.isPending) return <Spinner label={t("analysis.computing")} />;
  if (q.error || !q.data) return <ErrorNote error={q.error} />;
  return (
    <div className="space-y-2">
      <p className="text-sm">
        {t("analysis.robustness.baseline", {
          metric: q.data.metric,
          value: formatNumber(q.data.baseline, i18n.language),
          samples: q.data.samples,
        })}
      </p>
      <EChart option={option} label={t("analysis.tabs.robustness")} />
    </div>
  );
}

function ReportDownloads({ runId }: { runId: string }) {
  const { t } = useTranslation();
  const [error, setError] = useState<unknown>(null);
  const get = (format: "html" | "pdf" | "md") => {
    setError(null);
    downloadFromEngine(
      `/api/v1/runs/${runId}/report/document?format=${format}`,
      `informe-${runId}.${format}`,
    ).catch(setError);
  };
  return (
    <div className="space-y-2">
      <p className="text-sm text-muted">{t("analysis.report.hint")}</p>
      <div className="flex flex-wrap gap-2">
        {(["pdf", "html", "md"] as const).map((f) => (
          <Button key={f} variant="secondary" onClick={() => get(f)}>
            <Download className="h-4 w-4" aria-hidden="true" />
            {t(`analysis.report.${f}`)}
          </Button>
        ))}
      </div>
      <ErrorNote error={error} />
    </div>
  );
}

export function AnalysisPanel({
  runId,
  datasetVersionId,
  classes,
}: {
  runId: string;
  datasetVersionId: string;
  classes: string[];
}) {
  const { t } = useTranslation();
  return (
    <Card>
      <CardTitle>{t("analysis.title")}</CardTitle>
      <Tabs defaultValue="errors">
        <TabsList>
          {(["errors", "explain", "fairness", "robustness", "report"] as const).map((k) => (
            <TabsTrigger key={k} value={k}>
              {t(`analysis.tabs.${k}`)}
            </TabsTrigger>
          ))}
        </TabsList>
        <TabsContent value="errors" className="pt-4">
          <Errors runId={runId} />
        </TabsContent>
        <TabsContent value="explain" className="pt-4">
          <Explanation runId={runId} />
        </TabsContent>
        <TabsContent value="fairness" className="pt-4">
          <Fairness runId={runId} datasetVersionId={datasetVersionId} classes={classes} />
        </TabsContent>
        <TabsContent value="robustness" className="pt-4">
          <Robustness runId={runId} />
        </TabsContent>
        <TabsContent value="report" className="pt-4">
          <ReportDownloads runId={runId} />
        </TabsContent>
      </Tabs>
    </Card>
  );
}
