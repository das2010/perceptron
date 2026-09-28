/**
 * Playground (RF-EXP-02): probar el modelo exportado con una fila (tabular) o una imagen y ver
 * la predicción, la confianza y las probabilidades por clase.
 */
import { FlaskConical } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, Card, CardTitle, ErrorNote, Field, Input } from "@/components/ui";
import {
  type ExportReport,
  type PlaygroundResult,
  useDatasetSample,
  useExplainImage,
  useExplainRow,
  usePredictFile,
  usePredictRows,
} from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

function Result({ result }: { result: PlaygroundResult }) {
  const { t, i18n } = useTranslation();
  const p = result.predictions[0];
  if (!p) return null;
  const probs = Object.entries((p.probabilities ?? {}) as Record<string, number>).sort(
    (a, b) => b[1] - a[1],
  );
  return (
    <div className="mt-4 space-y-2" aria-live="polite">
      <p className="text-sm">
        {t("playground.prediction")} <strong className="text-base">{String(p.prediction)}</strong>
        {typeof p.confidence === "number" && (
          <span className="ml-2 text-muted">
            {t("playground.confidence", {
              value: formatNumber(p.confidence * 100, i18n.language, 1),
            })}
          </span>
        )}
      </p>
      {probs.length > 0 && (
        <ul className="space-y-1 text-xs">
          {probs.map(([cls, v]) => (
            <li key={cls} className="flex items-center gap-2">
              <span className="w-24 truncate">{cls}</span>
              <span className="h-2 flex-1 rounded bg-canvas">
                <span
                  className="block h-2 rounded bg-primary"
                  style={{ width: `${Math.round(v * 100)}%` }}
                />
              </span>
              <span className="w-12 text-right">{formatNumber(v * 100, i18n.language, 1)}%</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function PlaygroundPanel({
  runId,
  datasetVersionId,
  report,
}: {
  runId: string;
  datasetVersionId: string;
  report: ExportReport;
}) {
  const { t } = useTranslation();
  const inputs = report.signature.inputs as { kind: string; columns?: string[] };
  const columns = inputs.columns ?? [];
  const sample = useDatasetSample(inputs.kind === "tabular" ? datasetVersionId : undefined);
  const [values, setValues] = useState<Record<string, string>>({});
  const rows = usePredictRows(runId);
  const file = usePredictFile(runId);
  const explainRow = useExplainRow(runId);
  const explainImage = useExplainImage(runId);
  const [lastFile, setLastFile] = useState<File | null>(null);
  const result = rows.data ?? file.data;
  const currentRow = () => Object.fromEntries(columns.map((c) => [c, values[c] ?? ""]));

  const fillExample = () => {
    const row = (sample.data?.[0] ?? {}) as Record<string, unknown>;
    setValues(Object.fromEntries(columns.map((c) => [c, row[c] == null ? "" : String(row[c])])));
  };

  return (
    <Card>
      <CardTitle className="flex items-center gap-2">
        <FlaskConical className="h-4 w-4" aria-hidden="true" />
        {t("playground.title")}
      </CardTitle>
      {inputs.kind === "tabular" ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            file.reset();
            explainRow.reset();
            rows.mutate([currentRow()]);
          }}
        >
          <div className="grid gap-3 sm:grid-cols-3">
            {columns.map((c) => (
              <Field key={c} label={c}>
                <Input
                  value={values[c] ?? ""}
                  onChange={(e) => setValues((v) => ({ ...v, [c]: e.target.value }))}
                />
              </Field>
            ))}
          </div>
          <div className="mt-3 flex gap-2">
            <Button type="button" variant="secondary" onClick={fillExample} disabled={!sample.data}>
              {t("playground.example")}
            </Button>
            <Button type="submit" loading={rows.isPending}>
              {t("playground.predict")}
            </Button>
          </div>
        </form>
      ) : (
        <Field label={t("playground.image")}>
          <input
            type="file"
            accept="image/*"
            aria-label={t("playground.image")}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (!f) return;
              rows.reset();
              explainImage.reset();
              setLastFile(f);
              file.mutate(f);
            }}
          />
        </Field>
      )}
      <ErrorNote error={rows.error ?? file.error ?? explainRow.error ?? explainImage.error} />
      {result && <Result result={result} />}
      {result && (
        <Button
          className="mt-3"
          variant="ai"
          size="sm"
          loading={explainRow.isPending || explainImage.isPending}
          onClick={() =>
            inputs.kind === "tabular"
              ? explainRow.mutate(currentRow())
              : lastFile && explainImage.mutate(lastFile)
          }
        >
          {t("playground.explain")}
        </Button>
      )}
      {explainRow.data && (
        <ul className="mt-3 space-y-1 text-xs" aria-label={t("playground.contributions")}>
          {explainRow.data.contributions.slice(0, 10).map((c) => {
            const max = Math.max(...explainRow.data.contributions.map((x) => Math.abs(x.attribution)), 1e-9);
            return (
              <li key={c.feature} className="flex items-center gap-2">
                <span className="w-32 truncate">{c.feature}</span>
                <span className="relative h-2 flex-1 rounded bg-canvas">
                  <span
                    className={`absolute h-2 rounded ${c.attribution >= 0 ? "left-1/2 bg-primary" : "right-1/2 bg-bad"}`}
                    style={{ width: `${(Math.abs(c.attribution) / max) * 50}%` }}
                  />
                </span>
                <span className="w-16 text-right">{c.attribution.toFixed(3)}</span>
              </li>
            );
          })}
        </ul>
      )}
      {explainImage.data?.heatmap_png && (
        <img
          className="mt-3 max-h-80 rounded-pt border border-line"
          src={`data:image/png;base64,${explainImage.data.heatmap_png}`}
          alt={t("playground.heatmap")}
        />
      )}
    </Card>
  );
}
