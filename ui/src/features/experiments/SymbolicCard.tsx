/**
 * Fórmula sugerida (ADR-0039): regresión simbólica como modelo de referencia. Busca una
 * fórmula cerrada para el target, la compara con la mejor red, la deja probar con valores
 * nuevos y copiarla como Python o como fórmula de Excel.
 */
import { useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Sigma } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { EChart } from "@/components/charts/EChart";
import {
  Button,
  Card,
  CardTitle,
  ErrorNote,
  Field,
  Input,
  PendingHint,
  Select,
  Table,
  Td,
  Th,
} from "@/components/ui";
import {
  keys,
  type Run,
  type SymbolicFit,
  useDatasets,
  useJob,
  useStartSymbolic,
  useSymbolicFits,
  useSymbolicPredict,
} from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

const TIME_LIMITS = [30, 60, 180, 600];

function bestRun(runs: Run[], datasetVersionId: string): Run | undefined {
  return runs
    .filter((r) => r.dataset_version_id === datasetVersionId && r.metrics?.val_loss != null)
    .sort((a, b) => (a.metrics?.val_loss ?? Infinity) - (b.metrics?.val_loss ?? Infinity))[0];
}

export function SymbolicCard({
  projectId,
  runs,
  datasetVersionId,
}: {
  projectId: string;
  runs: Run[];
  /** Fija el dataset (wizard): sin selector. */
  datasetVersionId?: string | undefined;
}) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const datasets = (useDatasets(projectId).data ?? []).filter((d) => d.modality === "tabular");
  const fits = useSymbolicFits(projectId);
  const start = useStartSymbolic(projectId);
  const [dvId, setDvId] = useState<string>("");
  const [limit, setLimit] = useState(60);
  const [jobId, setJobId] = useState<string>();
  const job = useJob(jobId);
  const current = datasetVersionId || dvId || datasets[0]?.id || "";
  const running =
    Boolean(jobId) && !["succeeded", "failed", "cancelled"].includes(job.data?.status ?? "");
  const done = job.data?.status === "succeeded";
  useEffect(() => {
    if (done) void qc.invalidateQueries({ queryKey: keys.symbolic(projectId) });
  }, [done, qc, projectId]);

  if (datasets.length === 0) return null;
  const fit = (fits.data ?? []).find((f) => f.dataset_version_id === current);
  const jobError = job.data?.error
    ? new Error(String((job.data.error as { message?: string }).message ?? ""))
    : null;

  return (
    <Card>
      <CardTitle className="flex items-center gap-2">
        <Sigma className="h-4 w-4" aria-hidden="true" />
        {t("symbolic.title")}
      </CardTitle>
      <p className="mb-3 text-sm text-muted">{t("symbolic.hint")}</p>
      <div className="flex flex-wrap items-end gap-3">
        {!datasetVersionId && datasets.length > 1 && (
          <Field label={t("symbolic.dataset")}>
            <Select value={current} onChange={(e) => setDvId(e.target.value)}>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.content_hash.slice(0, 10)} · {d.target ?? "—"}
                </option>
              ))}
            </Select>
          </Field>
        )}
        <Field label={t("symbolic.timeLimit")}>
          <Select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
            {TIME_LIMITS.map((s) => (
              <option key={s} value={s}>
                {s < 60 ? `${s} s` : `${s / 60} min`}
              </option>
            ))}
          </Select>
        </Field>
        <Button
          loading={start.isPending || running}
          onClick={() =>
            start.mutate(
              { dataset_version_id: current, time_limit_s: limit },
              { onSuccess: (out) => setJobId(out.job.id) },
            )
          }
        >
          {t("symbolic.search")}
        </Button>
        <PendingHint active={running}>{t("symbolic.searching", { seconds: limit })}</PendingHint>
      </div>
      <ErrorNote error={start.error ?? jobError} />
      {fit && <FitView fit={fit} best={bestRun(runs, current)} />}
    </Card>
  );
}

function FitView({ fit, best }: { fit: SymbolicFit; best: Run | undefined }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language;
  const val = fit.metrics?.val ?? {};
  const test = fit.metrics?.test ?? {};
  const parity = useMemo(() => fit.parity ?? [], [fit.parity]);
  const option = useMemo(() => {
    const values = parity.flat();
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    return {
      xAxis: { type: "value", name: t("symbolic.real"), min: lo, max: hi },
      yAxis: { type: "value", name: t("symbolic.predicted"), min: lo, max: hi },
      series: [
        { type: "scatter", symbolSize: 6, data: parity },
        {
          type: "line",
          showSymbol: false,
          data: [
            [lo, lo],
            [hi, hi],
          ],
          lineStyle: { type: "dashed" },
        },
      ],
    };
  }, [parity, t]);
  const excel = lang.startsWith("es") ? fit.excel_es : fit.excel_en;
  const legend = fit.features
    .map((f, i) => `${String.fromCharCode(65 + (i % 26))}2 = ${f}`)
    .join(" · ");

  return (
    <div className="mt-4 space-y-4">
      <p className="rounded-pt bg-canvas p-3 font-mono text-lg" aria-label={t("symbolic.formula")}>
        {fit.target} = {fit.formula}
      </p>
      {(fit.warnings ?? []).map((w) => (
        <p key={w} className="flex items-start gap-2 text-sm text-warn">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          {w}
        </p>
      ))}
      <Table>
        <thead>
          <tr>
            <Th />
            <Th>{t("symbolic.valMae")}</Th>
            <Th>{t("symbolic.valR2")}</Th>
            <Th>{t("symbolic.testMae")}</Th>
            <Th>{t("symbolic.testR2")}</Th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <Td className="font-semibold">{t("symbolic.formula")}</Td>
            <Td>{formatNumber(val.mae, lang, 4)}</Td>
            <Td>{formatNumber(val.r2, lang, 5)}</Td>
            <Td>{formatNumber(test.mae, lang, 4)}</Td>
            <Td>{formatNumber(test.r2, lang, 5)}</Td>
          </tr>
          {best && (
            <tr>
              <Td className="font-semibold">
                {t("symbolic.bestNetwork", { run: best.id.split("-").at(-1) })}
              </Td>
              <Td>{formatNumber(best.metrics?.val_mae, lang, 4)}</Td>
              <Td>{formatNumber(best.metrics?.val_r2, lang, 5)}</Td>
              <Td className="text-muted">—</Td>
              <Td className="text-muted">—</Td>
            </tr>
          )}
        </tbody>
      </Table>
      {parity.length > 0 && <EChart option={option} height={260} label={t("symbolic.parity")} />}
      <TryFormula fit={fit} />
      <div className="flex flex-wrap items-center gap-3">
        <CopyButton text={fit.python} label={t("symbolic.copyPython")} />
        <CopyButton text={excel} label={t("symbolic.copyExcel")} />
        <span className="text-xs text-muted">{t("symbolic.excelLegend", { legend })}</span>
      </div>
      {(fit.candidates ?? []).length > 1 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-muted">{t("symbolic.candidates")}</summary>
          <ul className="mt-2 space-y-1 font-mono text-xs">
            {(fit.candidates ?? []).map((c, i) => (
              <li key={i}>
                {t("symbolic.candidate", {
                  complexity: c.length,
                  rmse: formatNumber(c.val_rmse as number, lang, 6),
                })}{" "}
                {String(c.formula)}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function TryFormula({ fit }: { fit: SymbolicFit }) {
  const { t, i18n } = useTranslation();
  const predict = useSymbolicPredict(fit.id);
  const [values, setValues] = useState<Record<string, string>>({});
  const row = Object.fromEntries(
    fit.features.map((f) => [
      f,
      values[f] === undefined || values[f] === "" ? null : Number(values[f]),
    ]),
  );
  const ready = fit.features.every((f) => row[f] !== null && Number.isFinite(row[f]));
  const result = predict.data?.[0];

  return (
    <div className="flex flex-wrap items-end gap-3">
      {fit.features.map((f) => (
        <Field key={f} label={f}>
          <Input
            type="number"
            step="any"
            value={values[f] ?? ""}
            onChange={(e) => setValues((prev) => ({ ...prev, [f]: e.target.value }))}
          />
        </Field>
      ))}
      <Button
        variant="secondary"
        disabled={!ready}
        loading={predict.isPending}
        onClick={() => predict.mutate([row])}
      >
        {t("symbolic.try")}
      </Button>
      {predict.isSuccess && (
        <span className="text-sm" role="status">
          {fit.target} ={" "}
          <strong>{result == null ? "—" : formatNumber(result, i18n.language, 6)}</strong>
        </span>
      )}
      <ErrorNote error={predict.error} />
    </div>
  );
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  return (
    <Button
      size="sm"
      variant="ghost"
      onClick={() =>
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        })
      }
    >
      {copied ? t("symbolic.copied") : label}
    </Button>
  );
}
