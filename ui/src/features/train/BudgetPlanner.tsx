/**
 * Intentos y épocas por intento propuestos por el sistema (ADR-0041): según cuántos
 * hiperparámetros se buscan, el tiempo medido por época y cuánto querés esperar. Se calcula
 * solo al elegir la arquitectura y se puede recalcular con otro tiempo; los números quedan
 * editables y la duración estimada se actualiza con lo que pongas.
 */
import { Clock } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, ErrorNote, Field, Input } from "@/components/ui";
import { type BudgetPlan, usePlanHpo } from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

const DEFAULT_MINUTES = 20;

function minutes(seconds: number): number {
  return Math.max(1, Math.round(seconds / 60));
}

export function BudgetPlanner({
  projectId,
  archspecId,
  datasetVersionId,
  trials,
  epochs,
  timeBudgetS,
  onPlan,
  onTimeBudget,
}: {
  projectId: string;
  archspecId: string;
  datasetVersionId: string;
  trials: number;
  epochs: number;
  timeBudgetS?: number | null | undefined;
  onPlan: (plan: BudgetPlan) => void;
  onTimeBudget?: ((seconds: number) => void) | undefined;
}) {
  const { t, i18n } = useTranslation();
  const plan = usePlanHpo(projectId);
  const [wait, setWait] = useState(timeBudgetS ? minutes(timeBudgetS) : DEFAULT_MINUTES);
  const requested = useRef("");

  const run = (mins: number) =>
    plan.mutate(
      { archspec_id: archspecId, dataset_version_id: datasetVersionId, time_budget_s: mins * 60 },
      { onSuccess: onPlan },
    );

  // Al elegir otra arquitectura (o datos), el plan se recalcula solo.
  useEffect(() => {
    const key = `${archspecId}|${datasetVersionId}`;
    if (!archspecId || !datasetVersionId || requested.current === key) return;
    requested.current = key;
    run(wait);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [archspecId, datasetVersionId]);

  const p = plan.data;
  const perEpoch = p?.epoch_time_s ?? null;
  const current = perEpoch ? trials * epochs * perEpoch : null;
  const edited = p && (trials !== p.max_trials || epochs !== p.max_epochs_per_trial);

  return (
    <section aria-label={t("plan.title")} className="mb-4 rounded-pt border border-line p-4 text-sm">
      <div className="flex flex-wrap items-end gap-3">
        <Field label={t("plan.wait")} hint={t("plan.waitHint")}>
          <Input
            type="number"
            min={1}
            max={1440}
            value={wait}
            onChange={(e) => setWait(Number(e.target.value))}
            onBlur={() => onTimeBudget?.(wait * 60)}
          />
        </Field>
        <Button variant="secondary" loading={plan.isPending} onClick={() => run(wait)}>
          {t("plan.propose")}
        </Button>
      </div>
      <ErrorNote error={plan.error} />
      {p && (
        <div className="mt-3">
          <p className="font-semibold">
            {t("plan.summary", { trials: p.max_trials, epochs: p.max_epochs_per_trial })}
          </p>
          <ul className="mt-1 list-disc pl-5 text-xs text-muted">
            {(p.reasons ?? []).map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
          {current !== null && (
            <p className="mt-2 flex items-center gap-1 text-xs">
              <Clock className="h-3 w-3" aria-hidden="true" />
              {t(edited ? "plan.estimateEdited" : "plan.estimate", {
                minutes: formatNumber(current / 60, i18n.language, 1),
                budget: minutes(p.time_budget_s),
              })}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
