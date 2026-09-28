/**
 * Export verificado del modelo (RF-EXP-01/05), servidor de inferencia (RF-EXP-03) y proyecto
 * de código (RF-EXP-04). Cada formato muestra su verificación numérica contra PyTorch.
 */
import { useQueryClient } from "@tanstack/react-query";
import { Download, PackageOpen } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { Badge, Button, Card, CardTitle, ErrorNote, Table, Td, Th } from "@/components/ui";
import { downloadFromEngine } from "@/lib/api/download";
import { useExportReport, useExportRun, useJob } from "@/lib/api/hooks";

const FORMATS = ["onnx", "torch_export", "torchscript"] as const;

function formatBytes(n: number, locale: string): string {
  const units = ["B", "KB", "MB", "GB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toLocaleString(locale, { maximumFractionDigits: 1 })} ${units[i]}`;
}

export function ExportPanel({ runId }: { runId: string }) {
  const { t, i18n } = useTranslation();
  const qc = useQueryClient();
  const report = useExportReport(runId, true);
  const exportRun = useExportRun(runId);
  const [formats, setFormats] = useState<string[]>(["onnx"]);
  const [fp16, setFp16] = useState(false);
  const [int8, setInt8] = useState(false);
  const [jobId, setJobId] = useState<string | undefined>();
  const job = useJob(jobId);
  const [error, setError] = useState<unknown>(null);
  const status = job.data?.status;
  const running = Boolean(jobId) && status !== "succeeded" && status !== "failed";

  useEffect(() => {
    if (status === "succeeded" || status === "failed") {
      void qc.invalidateQueries({ queryKey: ["runs", runId, "export"] });
    }
  }, [status, qc, runId]);

  const toggle = (f: string) =>
    setFormats((prev) => (prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]));
  const download = (path: string, name: string) => {
    setError(null);
    downloadFromEngine(path, name).catch(setError);
  };
  const hasOnnx = report.data?.artifacts.some(
    (a) => a.format === "onnx" && !a.error && a.verification?.passed,
  );

  return (
    <Card>
      <CardTitle>{t("export.title")}</CardTitle>
      <p className="mb-3 text-sm text-muted">{t("export.hint")}</p>
      <div className="flex flex-wrap items-center gap-4 text-sm">
        {FORMATS.map((f) => (
          <label key={f} className="flex items-center gap-2">
            <input type="checkbox" checked={formats.includes(f)} onChange={() => toggle(f)} />
            {t(`export.format.${f}`)}
          </label>
        ))}
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={fp16}
            disabled={!formats.includes("onnx")}
            onChange={(e) => setFp16(e.target.checked)}
          />
          {t("export.fp16")}
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={int8}
            disabled={!formats.includes("onnx")}
            onChange={(e) => setInt8(e.target.checked)}
          />
          {t("export.int8")}
        </label>
        <Button
          disabled={!formats.length}
          loading={exportRun.isPending || running}
          onClick={() =>
            exportRun.mutate(
              {
                formats: formats as (typeof FORMATS)[number][],
                fp16: fp16 && formats.includes("onnx"),
                int8: int8 && formats.includes("onnx"),
              },
              { onSuccess: (launch) => setJobId(launch.job.id) },
            )
          }
        >
          <PackageOpen className="h-4 w-4" aria-hidden="true" />
          {t("export.run")}
        </Button>
      </div>
      {running && <p className="mt-2 text-xs text-muted">{t("export.running")}</p>}
      <ErrorNote
        error={
          exportRun.error ??
          (status === "failed" ? new Error(String(job.data?.error?.message ?? "")) : null) ??
          error
        }
      />
      {report.data && (
        <>
          <div className="mt-4 overflow-x-auto">
            <Table>
              <thead>
                <tr>
                  <Th>{t("export.col.format")}</Th>
                  <Th>{t("export.col.size")}</Th>
                  <Th>{t("export.col.check")}</Th>
                  <Th />
                </tr>
              </thead>
              <tbody>
                {report.data.artifacts.map((a) => (
                  <tr key={a.format}>
                    <Td>
                      {t(`export.format.${a.format}`, { defaultValue: a.format })}
                      {a.legacy && (
                        <Badge className="ml-2" tone="warn">
                          legacy
                        </Badge>
                      )}
                    </Td>
                    <Td className="text-xs">
                      {a.error ? "—" : formatBytes(a.size_bytes, i18n.language)}
                    </Td>
                    <Td className="text-xs">
                      {a.error ? (
                        <span className="text-bad" title={a.error}>
                          {t("export.failed")}
                        </span>
                      ) : a.verification ? (
                        <Badge tone={a.verification.passed ? "ok" : "bad"}>
                          {t("export.diff", {
                            diff: a.verification.max_abs_diff.toExponential(1),
                            tol:
                              a.verification.tolerance != null
                                ? a.verification.tolerance.toExponential(0)
                                : "—",
                          })}
                        </Badge>
                      ) : null}
                    </Td>
                    <Td>
                      {!a.error && (
                        <Button
                          size="sm"
                          variant="ghost"
                          aria-label={t("export.download", { file: a.file })}
                          onClick={() =>
                            download(`/api/v1/runs/${runId}/export/files/${a.file}`, a.file)
                          }
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      )}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button
              variant="secondary"
              disabled={!hasOnnx}
              title={hasOnnx ? undefined : t("export.needsOnnx")}
              onClick={() =>
                download(`/api/v1/runs/${runId}/export/serving.zip`, `${runId}-serving.zip`)
              }
            >
              <Download className="h-4 w-4" aria-hidden="true" />
              {t("export.serving")}
            </Button>
            <Button
              variant="secondary"
              onClick={() =>
                download(`/api/v1/runs/${runId}/export/project.zip`, `${runId}-proyecto.zip`)
              }
            >
              <Download className="h-4 w-4" aria-hidden="true" />
              {t("export.project")}
            </Button>
          </div>
        </>
      )}
    </Card>
  );
}
