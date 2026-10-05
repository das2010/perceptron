/**
 * Progreso de un trial en curso: «Época 7 de 20 · ~2 min restantes» con una barra. Sale del
 * último evento de época del job del estudio (`job.progress`). Antes de la primera época la
 * barra es indeterminada: armar datos y modelo puede tardar.
 */
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";

import { useJob } from "@/lib/api/hooks";

type EpochProgress = {
  last?: string;
  run_id?: string;
  epoch?: number;
  max_epochs?: number | null;
  eta_s?: number | null;
};

function formatEta(seconds: number, t: TFunction): string {
  if (seconds < 60) return t("experiments.etaSeconds", { n: Math.max(1, Math.round(seconds)) });
  if (seconds < 3600) return t("experiments.etaMinutes", { n: Math.round(seconds / 60) });
  return t("experiments.etaHours", { n: Math.round((seconds / 3600) * 10) / 10 });
}

export function RunProgress({ jobId, runId }: { jobId: string; runId?: string }) {
  const { t } = useTranslation();
  const job = useJob(jobId);
  const p = (job.data?.progress ?? {}) as EpochProgress;
  const current = (runId === undefined || p.run_id === runId) && typeof p.epoch === "number";
  const done = current ? (p.epoch ?? 0) + 1 : 0;
  const total = current && p.max_epochs ? p.max_epochs : null;
  const pct = total ? Math.min(100, Math.round((done / total) * 100)) : null;
  const label = !current
    ? t("experiments.preparing")
    : total
      ? t("experiments.epochOf", { done, total })
      : t("experiments.epochN", { done });
  return (
    <div className="mt-1 w-48 max-w-full" data-testid={`progress-${runId ?? jobId}`}>
      <div
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        {...(pct !== null ? { "aria-valuenow": pct } : {})}
        className="h-1.5 w-full overflow-hidden rounded-full bg-line"
      >
        <div
          className={pct === null ? "h-full w-1/3 animate-pulse bg-brand" : "h-full bg-brand"}
          style={pct === null ? undefined : { width: `${pct}%` }}
        />
      </div>
      <p className="mt-1 text-xs text-muted">
        {label}
        {current && p.eta_s != null && p.eta_s > 0 && ` · ${formatEta(p.eta_s, t)}`}
      </p>
    </div>
  );
}
