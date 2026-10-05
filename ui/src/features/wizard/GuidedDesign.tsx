/**
 * Diseño guiado (ADR-0041, iteración 2): un clic arma el diseño completo — preparación,
 * propuestas evaluadas contra los requisitos, entrenamiento corto de comparación entre las
 * mejores, elegida con evidencia y estrategia de HPO. Nada se aplica solo: la persona acepta
 * el diseño o elige otra candidata.
 */
import { useQueryClient } from "@tanstack/react-query";
import { Check, Sparkles, Trophy } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { Badge, Button, ErrorNote, Field, Select, Table, Td, Th } from "@/components/ui";
import { RequirementsPanel } from "@/features/train/DesignRequirements";
import {
  type DesignOutcome,
  type DraftValues,
  useDraftDesign,
  useJob,
  useUpdateDraft,
} from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

type Mode = "auto" | "always" | "never";
type Save = (values: Partial<DraftValues>) => void;

export function GuidedDesign({
  projectId,
  values,
  save,
}: {
  projectId: string;
  values: DraftValues;
  save: Save;
}) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const start = useDraftDesign(projectId);
  const [mode, setMode] = useState<Mode>("auto");
  const [jobId, setJobId] = useState<string>();
  const job = useJob(jobId);
  const status = job.data?.status;
  const finished = Boolean(jobId && status && isDone(status));
  const running = start.isPending || (Boolean(jobId) && !finished);
  const stage = (job.data?.progress as { stage?: string } | undefined)?.stage;

  useEffect(() => {
    // Al terminar, el diseño ya está en el borrador: recargarlo.
    if (finished) void qc.invalidateQueries({ queryKey: ["projects", projectId, "draft"] });
  }, [finished, projectId, qc]);

  const run = () =>
    start.mutate({ tournament: mode }, { onSuccess: (j) => setJobId(j.id ?? undefined) });

  return (
    <section
      aria-label={t("guided.title")}
      className="space-y-3 rounded-pt border border-copilot bg-copilot-bg/30 p-4"
    >
      <div className="flex flex-wrap items-end gap-3">
        <div className="min-w-60 flex-1">
          <h3 className="flex items-center gap-2 font-semibold">
            <Sparkles className="h-4 w-4 text-copilot" aria-hidden="true" />
            {t("guided.title")}
          </h3>
          <p className="text-xs text-muted">{t("guided.intro")}</p>
        </div>
        <Field label={t("guided.compare")}>
          <Select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            <option value="auto">{t("guided.mode.auto")}</option>
            <option value="always">{t("guided.mode.always")}</option>
            <option value="never">{t("guided.mode.never")}</option>
          </Select>
        </Field>
        <Button variant="ai" loading={running} onClick={run}>
          {values.design ? t("guided.rerun") : t("guided.run")}
        </Button>
      </div>
      {running && (
        <p role="status" className="text-sm text-muted">
          {t(`guided.stage.${stage ?? "start"}`)}
        </p>
      )}
      {status === "failed" && <ErrorNote error={job.data?.error} />}
      <ErrorNote error={start.error} />
      {values.design && !running && (
        <Outcome projectId={projectId} design={values.design} values={values} save={save} />
      )}
    </section>
  );
}

function isDone(status: string): boolean {
  return status === "succeeded" || status === "failed" || status === "cancelled";
}

function Outcome({
  projectId,
  design,
  values,
  save,
}: {
  projectId: string;
  design: DesignOutcome;
  values: DraftValues;
  save: Save;
}) {
  const { t, i18n } = useTranslation();
  const candidates = design.candidates ?? [];
  const pick = candidates.find((c) => c.archspec_id === design.pick);
  const accepted = Boolean(pick && values.archspec_id === pick.archspec_id);
  const winner = design.tournament?.winner;
  const update = useUpdateDraft(projectId);
  const chosen = {
    archspec_id: design.pick ?? null,
    strategy: design.strategy ?? null,
    max_epochs_per_trial: design.max_epochs_per_trial ?? null,
  };
  const accept = () => save(chosen);
  // Un clic: acepta el diseño (arquitectura, búsqueda y épocas) y va a la revisión.
  const acceptAndReview = () => update.mutate({ values: chosen, step: "review" });
  return (
    <div className="space-y-3">
      <RequirementsPanel requirements={design.requirements} />
      {design.tournament_skipped && (
        <p className="text-xs text-muted">{t(`guided.skipped.${design.tournament_skipped}`)}</p>
      )}
      <Table>
        <thead>
          <tr>
            <Th>{t("guided.candidate")}</Th>
            <Th>{t("guided.score")}</Th>
            {design.tournament && <Th>{design.tournament.metric}</Th>}
            <Th />
          </tr>
        </thead>
        <tbody>
          {candidates.map((c) => (
            <tr key={c.archspec_id}>
              <Td>
                <span className="font-semibold">{c.title}</span>{" "}
                {c.recommended && <Badge tone="brand">{t("design.recommended")}</Badge>}{" "}
                {winner === c.archspec_id && (
                  <Badge tone="ok">
                    <Trophy className="h-3 w-3" aria-hidden="true" />
                    {t("guided.winner")}
                  </Badge>
                )}
                {!c.eligible && <Badge tone="warn">{t("guided.mustUnmet")}</Badge>}
              </Td>
              <Td>{formatNumber(c.score ?? null, i18n.language, 0)}</Td>
              {design.tournament && (
                <Td>
                  {c.tournament_metric != null
                    ? formatNumber(c.tournament_metric, i18n.language, 4)
                    : "—"}
                </Td>
              )}
              <Td className="text-right">
                {values.archspec_id === c.archspec_id ? (
                  <span className="inline-flex items-center gap-1 text-xs text-ok">
                    <Check className="h-3 w-3" aria-hidden="true" />
                    {t("guided.chosen")}
                  </span>
                ) : (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => save({ archspec_id: c.archspec_id, strategy: null })}
                  >
                    {t("train.choose")}
                  </Button>
                )}
              </Td>
            </tr>
          ))}
        </tbody>
      </Table>
      {pick && (
        <div className="rounded-pt border border-line bg-canvas p-3 text-sm">
          <p>
            <span className="font-semibold">{t("guided.pick", { title: pick.title })}</span>{" "}
            {design.pick_reason}
          </p>
          <p className="mt-1 text-xs text-muted">
            {t("guided.plan", {
              epochs: design.max_epochs_per_trial,
              strategy: (design.strategy as { strategy?: string } | null)?.strategy ?? "—",
            })}
          </p>
          <div className="mt-2">
            {accepted ? (
              <span className="inline-flex items-center gap-1 text-sm text-ok">
                <Check className="h-4 w-4" aria-hidden="true" />
                {t("guided.accepted")}
              </span>
            ) : (
              <div className="flex flex-wrap gap-2">
                <Button size="sm" loading={update.isPending} onClick={acceptAndReview}>
                  {t("guided.acceptReview")}
                </Button>
                <Button size="sm" variant="secondary" onClick={accept}>
                  {t("guided.accept")}
                </Button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/** Memo del diseño en la revisión: qué se eligió, por qué y qué requisitos cumple. */
export function DesignMemo({ values }: { values: DraftValues }) {
  const { t } = useTranslation();
  const design = values.design;
  if (!design || !values.archspec_id) return null;
  const chosen = (design.candidates ?? []).find((c) => c.archspec_id === values.archspec_id);
  if (!chosen) return null;
  const isPick = chosen.archspec_id === design.pick;
  const unmet = (chosen.checks ?? []).filter((c) => !c.met);
  const met = (chosen.checks ?? []).filter((c) => c.met);
  return (
    <section
      aria-label={t("guided.memo.title")}
      className="rounded-pt border border-line bg-canvas p-4 text-sm"
    >
      <h3 className="font-semibold">{t("guided.memo.title")}</h3>
      <p className="mt-1">
        <span className="font-semibold">{chosen.title}.</span>{" "}
        {isPick ? design.pick_reason : t("guided.memo.notPick")}
      </p>
      {met.length > 0 && (
        <p className="mt-1 text-xs text-ok">
          {t("guided.memo.met", { list: met.map((c) => t(`design.code.${c.code}`)).join(", ") })}
        </p>
      )}
      {unmet.length > 0 && (
        <p className="mt-1 text-xs text-bad">
          {t("guided.memo.unmet", {
            list: unmet.map((c) => t(`design.code.${c.code}`)).join(", "),
          })}
        </p>
      )}
    </section>
  );
}
