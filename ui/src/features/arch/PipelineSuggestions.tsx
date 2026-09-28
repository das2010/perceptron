/** Sugerencias de cambios al pipeline (RF-PIP-05): cada una con su justificación y el diff del
 * paso; el usuario acepta o descarta una por una y se aplican solo las aceptadas. */
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Sparkles } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { AiSuggestion, Button, Card, CardTitle, ErrorNote, Select } from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap, useDatasets } from "@/lib/api/hooks";

type Result = Schemas["SuggestionsResult"];
type Item = Result["items"][number];

function StepJson({ label, step }: { label: string; step: Item["before"] }) {
  return (
    <div className="min-w-0 flex-1">
      <p className="text-xs font-semibold text-muted">{label}</p>
      <pre className="mt-1 overflow-x-auto rounded-pt bg-canvas p-2 text-xs">
        {step
          ? JSON.stringify({ kind: step.kind, columns: step.columns, params: step.params }, null, 2)
          : "—"}
      </pre>
    </div>
  );
}

export function PipelineSuggestions({
  projectId,
  pipelineId,
}: {
  projectId: string;
  pipelineId: string;
}) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const datasets = useDatasets(projectId);
  const [dv, setDv] = useState("");
  const chosen = dv || datasets.data?.at(-1)?.id || "";
  const [decisions, setDecisions] = useState<Record<number, boolean>>({});
  const suggest = useMutation({
    mutationFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/pipelines/{pipeline_id}/suggestions", {
          params: { path: { pipeline_id: pipelineId } },
          body: { dataset_version_id: chosen, mode: "auto" },
        }),
      ) as Result,
    onSuccess: () => setDecisions({}),
  });
  const apply = useMutation({
    mutationFn: async (result: Result) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/pipelines/{pipeline_id}/suggestions/apply", {
          params: { path: { pipeline_id: pipelineId } },
          body: {
            version: result.pipeline_version,
            changes: result.items.filter((i) => decisions[i.index]).map((i) => i.change),
          },
        }),
      ),
    onSuccess: () => {
      suggest.reset();
      void qc.invalidateQueries({ queryKey: ["pipelines", pipelineId] });
    },
  });
  const result = suggest.data;
  const accepted = result ? result.items.filter((i) => decisions[i.index]).length : 0;

  return (
    <Card>
      <CardTitle className="flex items-center gap-2">
        <Sparkles className="h-4 w-4" aria-hidden="true" />
        {t("pipelineAdvice.title")}
      </CardTitle>
      <div className="flex flex-wrap items-end gap-2">
        <Select
          value={chosen}
          onChange={(e) => setDv(e.target.value)}
          aria-label={t("wizard.data.pick")}
        >
          {(datasets.data ?? []).map((d) => (
            <option key={d.id} value={d.id}>
              {d.content_hash.slice(0, 10)} · {d.num_samples}
            </option>
          ))}
        </Select>
        <Button
          variant="ai"
          disabled={!chosen}
          loading={suggest.isPending}
          onClick={() => suggest.mutate()}
        >
          {t("pipelineAdvice.ask")}
        </Button>
      </div>
      <ErrorNote error={suggest.error ?? apply.error} />
      {result && (
        <div className="mt-3 space-y-3">
          {result.origin === "rules" && (
            <p className="text-xs text-muted">
              {t("pipelineAdvice.rules", { reason: result.fallback_reason ?? "" })}
            </p>
          )}
          {result.items.length === 0 && <p className="text-sm">{t("pipelineAdvice.none")}</p>}
          {result.items.map((item) => (
            <AiSuggestion
              key={item.index}
              title={t(`pipelineAdvice.op.${item.change.op}`, { step: item.change.step_id })}
              accepted={decisions[item.index] === true}
              onAccept={() => setDecisions((d) => ({ ...d, [item.index]: true }))}
              onReject={() => setDecisions((d) => ({ ...d, [item.index]: false }))}
            >
              <p>{item.change.rationale}</p>
              <div className="mt-2 flex flex-col gap-2 sm:flex-row">
                <StepJson label={t("pipelineAdvice.before")} step={item.before} />
                <StepJson label={t("pipelineAdvice.after")} step={item.after} />
              </div>
            </AiSuggestion>
          ))}
          {result.items.length > 0 && (
            <Button
              disabled={accepted === 0}
              loading={apply.isPending}
              onClick={() => apply.mutate(result)}
            >
              {t("pipelineAdvice.apply", { count: accepted })}
            </Button>
          )}
        </div>
      )}
    </Card>
  );
}
