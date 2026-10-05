/**
 * Mejoras con un clic (ADR-0041, iteración 3): cada acción del diagnóstico se muestra como un
 * cambio concreto («lr: 0.01 → 0.003»). Aplicar guarda una arquitectura nueva desde el mejor
 * punto del run; entrenar la lanza con el presupuesto del estudio. Lo que no es de la
 * arquitectura indica dónde hacerlo.
 */
import { useNavigate } from "@tanstack/react-router";
import { ArrowRight, Wand2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, ErrorNote } from "@/components/ui";
import { useApplyImprovement, useCreateStudy, useImprovements } from "@/lib/api/hooks";

type Applied = NonNullable<ReturnType<typeof useApplyImprovement>["data"]>;

export function Improvements({
  runId,
  projectId,
  datasetVersionId,
  pipelineId,
}: {
  runId: string;
  projectId: string;
  datasetVersionId: string;
  pipelineId: string;
}) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const options = useImprovements(runId, true);
  const apply = useApplyImprovement(runId);
  const createStudy = useCreateStudy(projectId);
  const [applied, setApplied] = useState<Record<number, Applied>>({});
  const rows = options.data ?? [];
  if (rows.length === 0) return null;

  const train = (a: Applied) =>
    createStudy.mutate(
      {
        dataset_version_id: datasetVersionId,
        pipeline_id: pipelineId,
        archspec_id: a.archspec.id ?? "",
        budget: {
          max_trials: Number(a.budget?.max_trials ?? 5),
          max_epochs_per_trial: Number(a.budget?.max_epochs_per_trial ?? 15),
        },
      },
      {
        onSuccess: (launch) =>
          void navigate({
            to: "/projects/$projectId/experiments",
            params: { projectId },
            search: { job: launch.job.id },
          }),
      },
    );

  return (
    <section aria-label={t("improve.title")} className="mt-3 border-t border-line pt-3">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Wand2 className="h-4 w-4" aria-hidden="true" />
        {t("improve.title")}
      </h4>
      <ul className="mt-2 space-y-2 text-sm">
        {rows.map((o) => {
          const done = applied[o.index];
          return (
            <li key={o.index} className="flex flex-wrap items-center gap-2">
              <span className="min-w-60 flex-1">
                {o.rationale}
                {o.change && <span className="ml-1 font-mono text-xs">({o.change})</span>}
                {!o.applicable && o.hint && (
                  <span className="ml-1 text-xs text-muted">{t(`improve.hint.${o.hint}`)}</span>
                )}
              </span>
              {o.applicable && !done && (
                <Button
                  size="sm"
                  variant="secondary"
                  loading={apply.isPending && apply.variables === o.index}
                  onClick={() =>
                    apply.mutate(o.index, {
                      onSuccess: (a) => setApplied((m) => ({ ...m, [o.index]: a })),
                    })
                  }
                >
                  {t("improve.apply")}
                </Button>
              )}
              {done && (
                <Button size="sm" loading={createStudy.isPending} onClick={() => train(done)}>
                  {t("improve.train")}
                  <ArrowRight className="h-3 w-3" aria-hidden="true" />
                </Button>
              )}
            </li>
          );
        })}
      </ul>
      <ErrorNote error={apply.error ?? createStudy.error ?? options.error} />
    </section>
  );
}
