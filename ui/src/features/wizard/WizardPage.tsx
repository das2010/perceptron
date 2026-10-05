/**
 * Wizard de creación de la red (SPEC §7.6, RF-WIZ-01..04). Cada paso guarda el borrador
 * (`ProjectDraft`), así se puede cerrar y retomar. Las propuestas de la IA (arquitectura,
 * estrategia) se muestran en violeta y se aceptan o descartan; sin LLM se usan reglas.
 */
import { useNavigate } from "@tanstack/react-router";
import { Check, ChevronLeft, ChevronRight, Sparkles } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  EmptyState,
  ErrorNote,
  Field,
  Input,
  Select,
  Spinner,
  Textarea,
  PendingHint,
} from "@/components/ui";
import { useUiStore } from "@/app/store";
import { DefineWizard } from "@/features/arch/DefineWizard";
import { UploadPanel } from "@/features/data/DataPage";
import { ProfileView } from "@/features/data/ProfileView";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { NotRecommendedNote, RequirementsPanel } from "@/features/train/DesignRequirements";
import { ProposalCard } from "@/features/train/TrainPage";
import { DesignMemo, GuidedDesign } from "@/features/wizard/GuidedDesign";
import { getApiClient } from "@/lib/api/client";
import {
  unwrap,
  useCreateStudy,
  useDatasets,
  useDraft,
  useHpoStrategy,
  useProposeArchitecture,
  useProposePipeline,
  useUpdateDraft,
  type ArchProposals,
  type DraftValues,
  type Schemas,
  type UseCaseBrief,
  type WizardPlan,
} from "@/lib/api/hooks";
import { cn } from "@/lib/cn";

import { SymbolicCard } from "@/features/experiments/SymbolicCard";

import {
  BriefPanel,
  PlanChecks,
  PlanDiffBanner,
  PlanSuggestion,
  ReconcileCard,
  ThresholdStep,
} from "./BriefPanel";

const METRICS: Record<string, string[]> = {
  classification: ["val_loss", "val_accuracy", "val_f1_macro", "val_recall_macro", "val_roc_auc"],
  regression: ["val_loss", "val_mae", "val_rmse", "val_r2"],
  forecasting: ["val_loss", "val_mae", "val_smape"],
  anomaly_detection: ["val_loss"],
  object_detection: ["val_loss", "val_map_50"],
  segmentation: ["val_loss", "val_iou"],
  ocr: ["val_loss", "val_cer"],
};
const TASKS = Object.keys(METRICS);

type Save = (values: Partial<DraftValues>, origin?: "user" | "copilot") => void;

function StepGoal({ values, save }: { values: DraftValues; save: Save }) {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const [goal, setGoal] = useState(values.goal ?? "");
  return (
    <div className="space-y-4">
      <Field label={t("wizard.goal.label")} hint={t("wizard.goal.hint")}>
        <Textarea
          rows={4}
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          onBlur={() => goal !== (values.goal ?? "") && save({ goal })}
          placeholder={t("wizard.goal.placeholder")}
        />
      </Field>
      <BriefPanel
        projectId={projectId}
        brief={values.brief ?? ({} as UseCaseBrief)}
        save={(brief, origin) => save({ brief }, origin)}
      />
    </div>
  );
}

function StepData({ values, save }: { values: DraftValues; save: Save }) {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const datasets = useDatasets(projectId).data ?? [];
  return (
    <div className="space-y-4">
      <UploadPanel
        onIngested={(dv) => save({ dataset_version_id: dv.id, target: dv.target ?? null })}
      />
      {datasets.length > 0 && (
        <Field label={t("train.dataset")}>
          <Select
            value={values.dataset_version_id ?? ""}
            onChange={(e) => {
              const dv = datasets.find((d) => d.id === e.target.value);
              save({ dataset_version_id: e.target.value, target: dv?.target ?? null });
            }}
          >
            <option value="">{t("wizard.data.pick")}</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.content_hash.slice(0, 10)} · {d.num_samples} · {d.target ?? "—"}
              </option>
            ))}
          </Select>
        </Field>
      )}
    </div>
  );
}

function StepTask({ values, save, plan }: { values: DraftValues; save: Save; plan: WizardPlan }) {
  const { t } = useTranslation();
  const task = values.task ?? "classification";
  return (
    <div>
      <PlanSuggestion
        plan={plan}
        field="task"
        current={values.task ?? null}
        label={(v) => t(`task.${v}`)}
        onUse={(v) => save({ task: v as NonNullable<DraftValues["task"]> })}
      />
      <PlanSuggestion
        plan={plan}
        field="target_metric"
        current={values.target_metric ?? null}
        onUse={(v) => save({ target_metric: v })}
      />
      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        <Field label={t("wizard.task.task")}>
          <Select
            value={task}
            onChange={(e) => save({ task: e.target.value as NonNullable<DraftValues["task"]> })}
          >
            {TASKS.map((k) => (
              <option key={k} value={k}>
                {t(`task.${k}`)}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("wizard.task.metric")} hint={t("wizard.task.metricHint")}>
          <Select
            value={values.target_metric ?? ""}
            onChange={(e) => save({ target_metric: e.target.value || null })}
          >
            <option value="">{t("wizard.task.metricAuto")}</option>
            {(METRICS[task] ?? []).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("wizard.task.threshold")}>
          <Input
            type="number"
            step="any"
            defaultValue={values.success_threshold ?? ""}
            onBlur={(e) =>
              save({ success_threshold: e.target.value === "" ? null : Number(e.target.value) })
            }
          />
        </Field>
      </div>
    </div>
  );
}

function StepArchitecture({
  values,
  save,
  plan,
}: {
  values: DraftValues;
  save: Save;
  plan: WizardPlan;
}) {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const pipeline = useProposePipeline(projectId);
  const propose = useProposeArchitecture(projectId);
  const [proposals, setProposals] = useState<ArchProposals | null>(null);
  const [defining, setDefining] = useState(false);
  const dv = values.dataset_version_id;
  if (!dv) return <EmptyState>{t("train.noData")}</EmptyState>;

  const openDefine = () => {
    if (defining) return setDefining(false);
    if (values.pipeline_id) return setDefining(true);
    pipeline.mutate(dv, {
      onSuccess: (p) => {
        save({ pipeline_id: p.id ?? null });
        setDefining(true);
      },
    });
  };

  const run = () =>
    pipeline.mutate(dv, {
      onSuccess: (p) => {
        save({ pipeline_id: p.id ?? null });
        propose.mutate(
          { dataset_version_id: dv, pipeline_id: p.id ?? "", mode: "auto", n: 3 },
          {
            onSuccess: (res) => {
              setProposals(res);
              if (res.origin !== "llm")
                save({ archspec_id: res.proposals[0]?.archspec.id ?? null });
            },
          },
        );
      },
    });

  return (
    <div className="space-y-4">
      <PlanSuggestion plan={plan} field="architecture_hint" />
      <GuidedDesign projectId={projectId} values={values} save={save} />
      <div className="flex flex-wrap gap-2">
        <Button loading={pipeline.isPending || propose.isPending} onClick={run}>
          {t("train.proposeArch")}
        </Button>
        <Button variant="secondary" onClick={openDefine} aria-expanded={defining}>
          {defining ? t("define.close") : t("define.open")}
        </Button>
      </div>
      <PendingHint active={pipeline.isPending || propose.isPending}>
        {t("train.proposeArchWait")}
      </PendingHint>
      {defining && values.pipeline_id && (
        <DefineWizard
          projectId={projectId}
          datasetVersionId={dv}
          pipelineId={values.pipeline_id}
          onBuilt={(rec) => {
            save({ archspec_id: rec.id, strategy: null });
            setDefining(false);
            setProposals(null);
          }}
        />
      )}
      <ErrorNote error={pipeline.error ?? propose.error} />
      {proposals?.fallback_reason && (
        <p className="text-xs text-muted">
          {t("train.fallback", { reason: proposals.fallback_reason })}
        </p>
      )}
      {proposals && <RequirementsPanel requirements={proposals.requirements} />}
      <div className="grid gap-3 lg:grid-cols-2">
        {proposals?.proposals.map((p) => (
          <ProposalCard
            key={p.archspec.id}
            p={p}
            origin={proposals.origin}
            chosen={values.archspec_id === p.archspec.id}
            onChoose={() => save({ archspec_id: p.archspec.id ?? null, strategy: null })}
          />
        ))}
      </div>
      <NotRecommendedNote proposals={proposals?.proposals} chosenId={values.archspec_id} />
      {!proposals && values.archspec_id && (
        <p className="flex items-center gap-2 text-sm">
          <Check className="h-4 w-4 text-ok" aria-hidden="true" />
          {t("wizard.arch.chosen", { id: values.archspec_id })}
        </p>
      )}
    </div>
  );
}

function StepHpo({ values, save }: { values: DraftValues; save: Save }) {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const recommend = useHpoStrategy(projectId);
  const [trials, setTrials] = useState(values.max_trials ?? 10);
  const [epochs, setEpochs] = useState(values.max_epochs_per_trial ?? 15);
  if (!values.archspec_id) return <EmptyState>{t("wizard.hpo.needArch")}</EmptyState>;
  const s = values.strategy as {
    strategy?: string;
    pruner?: string;
    rationale?: string;
    origin?: string;
  } | null;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <Field label={t("train.trials")}>
          <Input
            type="number"
            min={1}
            value={trials}
            onChange={(e) => setTrials(Number(e.target.value))}
            onBlur={() => trials !== values.max_trials && save({ max_trials: trials })}
          />
        </Field>
        <Field label={t("train.epochs")}>
          <Input
            type="number"
            min={1}
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
            onBlur={() =>
              epochs !== values.max_epochs_per_trial && save({ max_epochs_per_trial: epochs })
            }
          />
        </Field>
        <Button
          loading={recommend.isPending}
          onClick={() =>
            recommend.mutate(
              {
                archspec_id: values.archspec_id ?? "",
                budget: { max_trials: trials, max_epochs_per_trial: epochs },
                mode: "auto",
                dataset_version_id: values.dataset_version_id ?? null,
              },
              { onSuccess: (res) => save({ strategy: res as unknown as Record<string, unknown> }) },
            )
          }
        >
          {t("train.recommend")}
        </Button>
      </div>
      {values.design?.strategy && !values.strategy && values.archspec_id === values.design.pick && (
        <Button
          variant="secondary"
          onClick={() =>
            save({
              strategy: values.design?.strategy ?? null,
              max_epochs_per_trial: values.design?.max_epochs_per_trial ?? null,
            })
          }
        >
          {t("guided.useStrategy")}
        </Button>
      )}
      <ErrorNote error={recommend.error} />
      {s && (
        <div
          className={cn(
            "rounded-pt border p-4 text-sm",
            s.origin === "llm" ? "border-copilot bg-copilot-bg/40" : "border-line",
          )}
        >
          <p>
            <strong>{s.strategy}</strong> + {t("train.pruner")} <strong>{s.pruner}</strong>{" "}
            {s.origin === "llm" && <Badge tone="brand">IA</Badge>}
          </p>
          {s.rationale && <p className="mt-2">{s.rationale}</p>}
        </div>
      )}
    </div>
  );
}

function StepBudget({ values, save }: { values: DraftValues; save: Save }) {
  const { t } = useTranslation();
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <Field label={t("wizard.budget.maxTime")} hint={t("wizard.budget.maxTimeHint")}>
        <Input
          type="number"
          min={1}
          defaultValue={values.max_time_s ? values.max_time_s / 60 : ""}
          onBlur={(e) =>
            save({ max_time_s: e.target.value === "" ? null : Number(e.target.value) * 60 })
          }
        />
      </Field>
      <Field label={t("wizard.budget.device")}>
        <Select
          value={values.device ?? ""}
          onChange={(e) =>
            save({ device: (e.target.value || null) as NonNullable<DraftValues["device"]> | null })
          }
        >
          <option value="">{t("wizard.budget.deviceAuto")}</option>
          <option value="cpu">CPU</option>
          <option value="cuda">CUDA</option>
        </Select>
      </Field>
      <Field label={t("wizard.budget.autonomous")} hint={t("wizard.budget.autonomousHint")}>
        <Select
          value={values.autonomous ? "yes" : "no"}
          onChange={(e) => save({ autonomous: e.target.value === "yes" })}
        >
          <option value="no">{t("wizard.budget.guided")}</option>
          <option value="yes">{t("wizard.budget.agent")}</option>
        </Select>
      </Field>
      {values.autonomous && (
        <Field label={t("agent.maxCost")}>
          <Input
            type="number"
            min={0}
            step={0.1}
            defaultValue={values.llm_budget_usd ?? 1.5}
            onBlur={(e) => save({ llm_budget_usd: Number(e.target.value) })}
          />
        </Field>
      )}
    </div>
  );
}

function StepReview({ values }: { values: DraftValues }) {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const navigate = useNavigate();
  const createStudy = useCreateStudy(projectId);
  const [agentError, setAgentError] = useState<unknown>(null);
  const [launchingAgent, setLaunchingAgent] = useState(false);
  const ready = Boolean(
    values.dataset_version_id && values.pipeline_id && (values.archspec_id || values.autonomous),
  );
  const budget = {
    max_trials: values.max_trials ?? 10,
    max_epochs_per_trial: values.max_epochs_per_trial ?? 15,
    ...(values.max_time_s ? { max_time_s: values.max_time_s } : {}),
  };

  const launchAgent = async () => {
    setLaunchingAgent(true);
    try {
      const api = await getApiClient();
      const res = unwrap(
        await api.POST("/api/v1/projects/{project_id}/agent/runs", {
          params: { path: { project_id: projectId } },
          body: {
            dataset_version_id: values.dataset_version_id ?? "",
            pipeline_id: values.pipeline_id ?? "",
            limits: {
              max_time_s: values.max_time_s ?? 3600,
              max_iterations: 3,
              max_steps: 30,
              max_trials: budget.max_trials,
              max_epochs_per_trial: budget.max_epochs_per_trial,
              max_llm_cost_usd: values.llm_budget_usd ?? 1.5,
              max_disk_mb: 5000,
              selection_metric: values.target_metric ?? "val_loss",
            },
            approval: { mode: "never", budget_pct: 50 },
          },
        }),
      );
      void navigate({
        to: "/projects/$projectId/agent",
        params: { projectId },
        search: { agent: res.agent_run.id ?? "" },
      });
    } catch (e) {
      setAgentError(e);
    } finally {
      setLaunchingAgent(false);
    }
  };

  const rows: [string, string][] = [
    [t("project.goal"), values.goal ?? "—"],
    [t("train.dataset"), values.dataset_version_id ?? "—"],
    [t("data.target"), values.target ?? "—"],
    [t("wizard.task.task"), values.task ? t(`task.${values.task}`) : "—"],
    [t("wizard.task.metric"), values.target_metric ?? t("wizard.task.metricAuto")],
    [t("wizard.review.arch"), values.archspec_id ?? "—"],
    [t("train.trials"), String(budget.max_trials)],
    [t("train.epochs"), String(budget.max_epochs_per_trial)],
    [
      t("wizard.budget.autonomous"),
      values.autonomous ? t("wizard.budget.agent") : t("wizard.budget.guided"),
    ],
  ];
  return (
    <div className="space-y-4">
      <dl className="grid gap-2 text-sm sm:grid-cols-2">
        {rows.map(([k, v]) => (
          <div key={k} className="flex gap-2">
            <dt className="min-w-40 text-muted">{k}</dt>
            <dd className="font-semibold break-all">{v}</dd>
          </div>
        ))}
      </dl>
      <DesignMemo values={values} />
      <p className="text-sm">
        {t(values.autonomous ? "wizard.review.summaryAgent" : "wizard.review.summary", budget)}
      </p>
      {values.autonomous ? (
        <Button
          variant="ai"
          disabled={!ready}
          loading={launchingAgent}
          onClick={() => void launchAgent()}
        >
          {t("agent.launch")}
        </Button>
      ) : (
        <Button
          disabled={!ready}
          loading={createStudy.isPending}
          onClick={() =>
            createStudy.mutate(
              {
                dataset_version_id: values.dataset_version_id ?? "",
                pipeline_id: values.pipeline_id ?? "",
                archspec_id: values.archspec_id ?? "",
                ...(values.strategy
                  ? {
                      strategy: values.strategy as unknown as NonNullable<
                        Schemas["StudyCreate"]["strategy"]
                      >,
                    }
                  : {}),
                budget,
              },
              {
                onSuccess: (launch) =>
                  void navigate({
                    to: "/projects/$projectId/experiments",
                    params: { projectId },
                    search: { job: launch.job.id },
                  }),
              },
            )
          }
        >
          {t("train.launch")}
        </Button>
      )}
      {!ready && <p className="text-xs text-muted">{t("wizard.review.missing")}</p>}
      <ErrorNote error={createStudy.error ?? agentError} />
    </div>
  );
}

export function WizardPage() {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const { data: view, isPending, error } = useDraft(projectId);
  const update = useUpdateDraft(projectId);
  const askCopilot = useUiStore((s) => s.askCopilot);

  if (isPending) return <Spinner />;
  if (error || !view) return <ErrorNote error={error} />;
  const plan = view.plan;
  const steps = plan.steps.map((s) => s.id);
  // Un paso salteado por el plan (p. ej. etiquetado) lleva al siguiente que sí está.
  const at = view.steps.indexOf(view.draft.step);
  const current = Math.max(
    0,
    steps.findIndex((s) => view.steps.indexOf(s) >= at),
  );
  const step = steps[current] ?? "goal";
  const reason = plan.steps.find((s) => s.id === step)?.reason;
  const values = view.values;
  const save: Save = (v, origin) => update.mutate({ values: v, ...(origin ? { origin } : {}) });
  const go = (i: number) => update.mutate({ step: steps[i] ?? step });

  return (
    <div className="space-y-4">
      <ol className="flex flex-wrap gap-2" aria-label={t("wizard.steps")}>
        {steps.map((s, i) => (
          <li key={s}>
            <button
              type="button"
              onClick={() => go(i)}
              aria-current={i === current ? "step" : undefined}
              className={cn(
                "rounded-full border px-3 py-1 text-xs",
                i === current
                  ? "border-brand bg-pt-dark font-semibold text-pt-lime"
                  : "border-line text-muted",
              )}
            >
              {i + 1}. {t(`wizard.step.${s}`)}
            </button>
          </li>
        ))}
      </ol>
      <PlanDiffBanner diff={values.plan_diff} />
      {(plan.skipped ?? []).map((s) => (
        <p key={s.id} className="text-xs text-muted">
          {t("wizard.plan.skipped", { step: t(`wizard.step.${s.id}`), reason: s.reason ?? "" })}
        </p>
      ))}
      <Card>
        <CardTitle className="flex items-center justify-between gap-2">
          {t(`wizard.step.${step}`)}
          <Button
            size="sm"
            variant="ai"
            onClick={() => askCopilot(t("wizard.why.question", { step: t(`wizard.step.${step}`) }))}
          >
            <Sparkles className="h-4 w-4" aria-hidden="true" />
            {t("wizard.why.button")}
          </Button>
        </CardTitle>
        <p className="mb-4 text-sm text-muted">{t(`wizard.help.${step}`)}</p>
        {reason && (
          <p className="mb-4 text-sm">
            <Badge tone="brand" className="mr-2">
              {t("wizard.plan.why")}
            </Badge>
            {reason}
          </p>
        )}
        <PlanChecks plan={plan} step={step} />
        {step === "goal" && <StepGoal values={values} save={save} />}
        {step === "data" && <StepData values={values} save={save} />}
        {step === "quality" &&
          (values.dataset_version_id ? (
            <div className="space-y-4">
              {values.brief && (
                <ReconcileCard
                  projectId={projectId}
                  brief={values.brief}
                  save={(brief, origin) => save({ brief }, origin)}
                />
              )}
              <ProfileView datasetVersionId={values.dataset_version_id} />
            </div>
          ) : (
            <EmptyState>{t("train.noData")}</EmptyState>
          ))}
        {step === "formula" &&
          (values.dataset_version_id ? (
            <SymbolicCard
              projectId={projectId}
              runs={[]}
              datasetVersionId={values.dataset_version_id}
            />
          ) : (
            <EmptyState>{t("train.noData")}</EmptyState>
          ))}
        {step === "threshold" && (
          <>
            <ThresholdStep
              brief={values.brief ?? ({} as UseCaseBrief)}
              save={(brief, origin) => save({ brief }, origin)}
            />
            <PlanSuggestion
              plan={plan}
              field="target_metric"
              current={values.target_metric ?? null}
              onUse={(v) => save({ target_metric: v })}
            />
          </>
        )}
        {step === "labeling" && (
          <p className="text-sm">
            {values.target
              ? t("wizard.labeling.ok", { target: values.target })
              : t("wizard.labeling.none")}
          </p>
        )}
        {step === "task" && <StepTask values={values} save={save} plan={plan} />}
        {step === "architecture" && <StepArchitecture values={values} save={save} plan={plan} />}
        {step === "hpo" && <StepHpo values={values} save={save} />}
        {step === "budget" && <StepBudget values={values} save={save} />}
        {step === "review" && <StepReview values={values} />}
        <ErrorNote error={update.error} />
      </Card>
      <div className="flex justify-between">
        <Button variant="secondary" disabled={current === 0} onClick={() => go(current - 1)}>
          <ChevronLeft className="h-4 w-4" aria-hidden="true" />
          {t("wizard.prev")}
        </Button>
        {current < steps.length - 1 && (
          <Button onClick={() => go(current + 1)}>
            {t("wizard.next")}
            <ChevronRight className="h-4 w-4" aria-hidden="true" />
          </Button>
        )}
      </div>
    </div>
  );
}
