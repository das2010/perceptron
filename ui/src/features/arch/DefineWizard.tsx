/**
 * Sub-wizard de definición de arquitectura (SPEC §7.6, paso 6): familia → backbone → cabeza →
 * regularización. Las opciones vienen del catálogo con la recomendación por reglas marcada; el
 * copiloto ("guía de definición") explica las del paso actual.
 */
import { Check, Sparkles } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { useUiStore } from "@/app/store";
import { Badge, Button, ErrorNote, Spinner } from "@/components/ui";
import { type ArchSpecRecord, useBuildDefinition, useDefinitionPlan } from "@/lib/api/hooks";
import { cn } from "@/lib/cn";

export function DefineWizard({
  projectId,
  datasetVersionId,
  pipelineId,
  onBuilt,
}: {
  projectId: string;
  datasetVersionId: string;
  pipelineId: string;
  onBuilt: (record: ArchSpecRecord) => void;
}) {
  const { t } = useTranslation();
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [current, setCurrent] = useState(0);
  const plan = useDefinitionPlan(projectId, datasetVersionId, pipelineId, choices);
  const build = useBuildDefinition(projectId);
  const askCopilot = useUiStore((s) => s.askCopilot);

  if (plan.error) return <ErrorNote error={plan.error} />;
  if (!plan.data) return <Spinner />;
  const steps = plan.data.steps;
  const step = steps[Math.min(current, steps.length - 1)];
  if (!step) return null;

  const choose = (id: string) => {
    // Cambiar la familia invalida el backbone: sus opciones dependen de ella.
    const next = { ...choices, [step.step]: id };
    if (step.step === "family" && choices.family !== id) delete next.backbone;
    setChoices(next);
    if (current < steps.length - 1) setCurrent(current + 1);
  };
  const acceptRecommended = () => {
    const next: Record<string, string> = { ...choices };
    for (const s of steps) {
      const rec = s.options.find((o) => o.recommended && o.available);
      if (!next[s.step] && rec) next[s.step] = rec.id;
    }
    setChoices(next);
    setCurrent(steps.length - 1);
  };

  return (
    <div className="space-y-4 rounded-pt border border-line p-4">
      <ol className="flex flex-wrap gap-2" aria-label={t("define.steps")}>
        {steps.map((s, i) => (
          <li key={s.step}>
            <button
              type="button"
              onClick={() => setCurrent(i)}
              aria-current={i === current ? "step" : undefined}
              className={cn(
                "flex items-center gap-1 rounded-full border px-3 py-1 text-xs",
                i === current ? "border-brand font-semibold" : "border-line text-muted",
              )}
            >
              {s.choice && <Check className="h-3 w-3 text-ok" aria-hidden="true" />}
              {i + 1}. {t(`define.step.${s.step}`)}
            </button>
          </li>
        ))}
      </ol>
      <div className="flex items-center justify-between gap-2">
        <h4 className="font-semibold">{t(`define.step.${step.step}`)}</h4>
        <Button
          size="sm"
          variant="ai"
          onClick={() =>
            askCopilot(
              t("define.explain", {
                step: t(`define.step.${step.step}`),
                options: step.options.map((o) => o.title).join(", "),
              }),
            )
          }
        >
          <Sparkles className="h-4 w-4" aria-hidden="true" />
          {t("define.explainButton")}
        </Button>
      </div>
      <div
        className="grid gap-2 sm:grid-cols-3"
        role="radiogroup"
        aria-label={t(`define.step.${step.step}`)}
      >
        {step.options.map((o) => (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={step.choice === o.id}
            disabled={!o.available}
            title={o.reason ?? undefined}
            onClick={() => choose(o.id)}
            className={cn(
              "rounded-pt border p-3 text-left text-sm disabled:opacity-50",
              step.choice === o.id ? "border-brand bg-canvas" : "border-line hover:bg-canvas",
            )}
          >
            <span className="flex items-center justify-between gap-2 font-semibold">
              {o.title}
              {o.recommended && <Badge tone="ok">{t("define.recommended")}</Badge>}
            </span>
            <span className="mt-1 block text-xs text-muted">
              {o.available ? o.description : o.reason}
            </span>
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        {!plan.data.complete && (
          <Button variant="secondary" size="sm" onClick={acceptRecommended}>
            {t("define.useRecommended")}
          </Button>
        )}
        <Button
          size="sm"
          disabled={!plan.data.complete || plan.isFetching}
          loading={build.isPending}
          onClick={() =>
            build.mutate(
              { dataset_version_id: datasetVersionId, pipeline_id: pipelineId, choices },
              { onSuccess: onBuilt },
            )
          }
        >
          {t("define.build")}
        </Button>
      </div>
      <ErrorNote error={build.error} />
    </div>
  );
}
