import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  AiSuggestion,
  Badge,
  Button,
  Card,
  CardTitle,
  EmptyState,
  ErrorNote,
  Field,
  Input,
  Select,
} from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import {
  useCreateStudy,
  useDatasets,
  useHpoStrategy,
  useProposeArchitecture,
  useProposePipeline,
  type ArchProposals,
  type HPOStrategy,
  type Pipeline,
  type Schemas,
} from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

type Proposal = ArchProposals["proposals"][number];

function Step({
  n,
  title,
  children,
  done,
}: {
  n: number;
  title: string;
  children: React.ReactNode;
  done?: boolean;
}) {
  return (
    <Card className={done ? "border-brand" : undefined}>
      <CardTitle className="flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-pt-dark text-xs text-pt-lime">
          {n}
        </span>
        {title}
      </CardTitle>
      {children}
    </Card>
  );
}

export function ProposalCard({
  p,
  origin,
  chosen,
  onChoose,
}: {
  p: Proposal;
  origin: string;
  chosen: boolean;
  onChoose: () => void;
}) {
  const { t, i18n } = useTranslation();
  const est = p.estimates ?? {};
  const body = (
    <>
      <p>{p.rationale}</p>
      <p className="mt-2 text-xs text-muted">
        {t("train.estimates", {
          params: formatNumber(est.num_params ?? null, i18n.language, 0),
          memory: formatNumber(est.memory_mb ?? null, i18n.language, 0),
          time: formatNumber(est.epoch_time_s ?? null, i18n.language, 1),
        })}
      </p>
      {p.pros?.length || p.cons?.length ? (
        <ul className="mt-2 list-disc pl-5 text-xs">
          {p.pros?.map((x) => (
            <li key={`p${x}`}>+ {x}</li>
          ))}
          {p.cons?.map((x) => (
            <li key={`c${x}`}>− {x}</li>
          ))}
        </ul>
      ) : null}
    </>
  );
  if (origin === "llm") {
    return (
      <AiSuggestion title={p.title} onAccept={onChoose} accepted={chosen}>
        {body}
      </AiSuggestion>
    );
  }
  return (
    <div className={`rounded-pt border p-4 ${chosen ? "border-brand" : "border-line"}`}>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-semibold">{p.title}</h3>
        <Badge>{t("train.byRules")}</Badge>
      </div>
      <div className="text-sm">{body}</div>
      {!chosen && (
        <Button size="sm" variant="secondary" className="mt-3" onClick={onChoose}>
          {t("train.choose")}
        </Button>
      )}
    </div>
  );
}

export function TrainPage() {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const navigate = useNavigate();
  const { data: datasetList } = useDatasets(projectId);
  const datasets = datasetList ?? [];
  const [picked, setDvId] = useState<string>("");
  const dvId = picked || datasets.at(-1)?.id || "";
  const [pipeline, setPipeline] = useState<Pipeline | null>(null);
  const [proposals, setProposals] = useState<ArchProposals | null>(null);
  const [archspecId, setArchspecId] = useState<string>("");
  const [trials, setTrials] = useState(10);
  const [epochs, setEpochs] = useState(15);
  const [strategy, setStrategy] = useState<HPOStrategy | null>(null);

  const proposePipeline = useProposePipeline(projectId);
  const proposeArch = useProposeArchitecture(projectId);
  const recommend = useHpoStrategy(projectId);
  const createStudy = useCreateStudy(projectId);

  if (datasets.length === 0) return <EmptyState>{t("train.noData")}</EmptyState>;

  const budget = { max_trials: trials, max_epochs_per_trial: epochs };
  const rationale = (pipeline?.graph as { rationale?: string[] } | undefined)?.rationale ?? [];

  return (
    <div className="space-y-4">
      <Step n={1} title={t("train.step.data")} done={Boolean(pipeline)}>
        <div className="flex flex-wrap items-end gap-3">
          <Field label={t("train.dataset")}>
            <Select
              value={dvId}
              onChange={(e) => {
                setDvId(e.target.value);
                setPipeline(null);
                setProposals(null);
                setStrategy(null);
              }}
            >
              {datasets.map((dv) => (
                <option key={dv.id} value={dv.id}>
                  {dv.content_hash.slice(0, 10)} · {dv.num_samples}
                </option>
              ))}
            </Select>
          </Field>
          <Button
            loading={proposePipeline.isPending}
            onClick={() => proposePipeline.mutate(dvId, { onSuccess: setPipeline })}
          >
            {t("train.proposePipeline")}
          </Button>
        </div>
        {rationale.length > 0 && (
          <ul className="mt-3 list-disc pl-5 text-sm">
            {rationale.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
        <ErrorNote error={proposePipeline.error} />
      </Step>

      {pipeline && (
        <Step n={2} title={t("train.step.arch")} done={Boolean(archspecId)}>
          <Button
            loading={proposeArch.isPending}
            onClick={() =>
              proposeArch.mutate(
                { dataset_version_id: dvId, pipeline_id: pipeline.id ?? "", mode: "auto", n: 3 },
                {
                  onSuccess: (res) => {
                    setProposals(res);
                    setArchspecId(
                      res.origin === "llm" ? "" : (res.proposals[0]?.archspec.id ?? ""),
                    );
                    setStrategy(null);
                  },
                },
              )
            }
          >
            {t("train.proposeArch")}
          </Button>
          <ErrorNote error={proposeArch.error} />
          {proposals?.fallback_reason && (
            <p className="mt-3 text-xs text-muted">
              {t("train.fallback", { reason: proposals.fallback_reason })}
            </p>
          )}
          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            {proposals?.proposals.map((p) => (
              <ProposalCard
                key={p.archspec.id}
                p={p}
                origin={proposals.origin}
                chosen={archspecId === p.archspec.id}
                onChoose={() => {
                  setArchspecId(p.archspec.id ?? "");
                  setStrategy(null);
                }}
              />
            ))}
          </div>
        </Step>
      )}

      {archspecId && (
        <Step n={3} title={t("train.step.hpo")} done={Boolean(strategy)}>
          <div className="flex flex-wrap items-end gap-3">
            <Field label={t("train.trials")}>
              <Input
                type="number"
                min={1}
                max={200}
                value={trials}
                onChange={(e) => setTrials(Number(e.target.value))}
              />
            </Field>
            <Field label={t("train.epochs")}>
              <Input
                type="number"
                min={1}
                max={500}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
              />
            </Field>
            <Button
              loading={recommend.isPending}
              onClick={() =>
                recommend.mutate(
                  { archspec_id: archspecId, budget, mode: "auto", dataset_version_id: dvId },
                  { onSuccess: setStrategy },
                )
              }
            >
              {t("train.recommend")}
            </Button>
          </div>
          <ErrorNote error={recommend.error} />
          {strategy && (
            <div className="mt-4">
              {(() => {
                const body = (
                  <>
                    <p>
                      <strong>{strategy.strategy}</strong> + {t("train.pruner")}{" "}
                      <strong>{strategy.pruner}</strong> ·{" "}
                      {t("train.trialsN", { count: strategy.budget?.max_trials ?? trials })}
                    </p>
                    <p className="mt-1 text-xs text-muted">
                      {t("train.searchSpace")}:{" "}
                      {(strategy.search_space ?? []).map((s) => s.name).join(", ") || "—"}
                    </p>
                    {strategy.rationale && <p className="mt-2">{strategy.rationale}</p>}
                  </>
                );
                return strategy.origin === "llm" ? (
                  <AiSuggestion title={t("train.strategy")} onReject={() => setStrategy(null)}>
                    {body}
                  </AiSuggestion>
                ) : (
                  <div className="rounded-pt border border-line p-4 text-sm">{body}</div>
                );
              })()}
            </div>
          )}
        </Step>
      )}

      {strategy && (
        <Step n={4} title={t("train.step.launch")}>
          <Button
            loading={createStudy.isPending}
            onClick={() =>
              createStudy.mutate(
                {
                  dataset_version_id: dvId,
                  pipeline_id: pipeline?.id ?? "",
                  archspec_id: archspecId,
                  strategy: strategy as NonNullable<Schemas["StudyCreate"]["strategy"]>,
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
          <ErrorNote error={createStudy.error} />
        </Step>
      )}
    </div>
  );
}
