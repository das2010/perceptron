import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, Square } from "lucide-react";
import { useCallback, useState } from "react";
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
  Spinner,
} from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { getApiClient } from "@/lib/api/client";
import { unwrap, useDatasets, type Schemas } from "@/lib/api/hooks";
import { useEngineSocket } from "@/lib/api/ws";
import { formatNumber } from "@/lib/format";

type AgentRun = Schemas["AgentRun"] & { id: string };
type ApprovalMode = Schemas["ApprovalPolicy"]["mode"];
interface LogEntry {
  ts: number;
  kind: string;
  message: string;
  tool?: string | null;
  iteration?: number;
  step?: number;
}

const TERMINAL = new Set(["finished", "failed", "stopped"]);
const STATE_TONE = {
  running: "brand",
  awaiting_approval: "warn",
  finished: "ok",
  failed: "bad",
  stopped: "neutral",
} as const;

function useAgentRun(agentId: string | undefined) {
  return useQuery({
    queryKey: ["agent", agentId],
    enabled: Boolean(agentId),
    refetchInterval: (q) => (q.state.data && TERMINAL.has(q.state.data.state) ? false : 3000),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/agent/runs/{agent_id}", {
          params: { path: { agent_id: agentId ?? "" } },
        }),
      ) as AgentRun,
  });
}

function LaunchForm({ onLaunched }: { onLaunched: (id: string) => void }) {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const datasets = useDatasets(projectId).data ?? [];
  const [picked, setPicked] = useState("");
  const dv = picked || datasets[0]?.id || ""; // la lista viene de la más nueva a la más vieja
  const [trials, setTrials] = useState(12);
  const [iterations, setIterations] = useState(3);
  const [epochs, setEpochs] = useState(15);
  const [cost, setCost] = useState(1.5);
  const [approval, setApproval] = useState<ApprovalMode>("never");

  const launch = useMutation({
    mutationFn: async () => {
      const api = await getApiClient();
      const pipeline = unwrap(
        await api.POST("/api/v1/projects/{project_id}/pipelines/propose", {
          params: { path: { project_id: projectId } },
          body: { dataset_version_id: dv },
        }),
      );
      return unwrap(
        await api.POST("/api/v1/projects/{project_id}/agent/runs", {
          params: { path: { project_id: projectId } },
          body: {
            dataset_version_id: dv,
            pipeline_id: pipeline.id ?? "",
            limits: {
              max_time_s: 3600,
              max_iterations: iterations,
              max_steps: 30,
              max_trials: trials,
              max_epochs_per_trial: epochs,
              max_llm_cost_usd: cost,
              max_disk_mb: 5000,
              selection_metric: "val_loss",
            },
            approval: { mode: approval, budget_pct: 50 },
          },
        }),
      );
    },
    onSuccess: (res) => onLaunched(res.agent_run.id ?? ""),
  });

  if (datasets.length === 0) return <EmptyState>{t("train.noData")}</EmptyState>;
  return (
    <Card>
      <CardTitle className="flex items-center gap-2">
        <Bot className="h-4 w-4 text-copilot" aria-hidden="true" />
        {t("agent.launchTitle")}
      </CardTitle>
      <p className="mb-4 text-sm text-muted">{t("agent.launchHint")}</p>
      <div className="grid gap-3 sm:grid-cols-3">
        <Field label={t("train.dataset")}>
          <Select value={dv} onChange={(e) => setPicked(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.content_hash.slice(0, 10)} · {d.num_samples}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("agent.maxTrials")}>
          <Input
            type="number"
            min={1}
            value={trials}
            onChange={(e) => setTrials(Number(e.target.value))}
          />
        </Field>
        <Field label={t("agent.maxIterations")}>
          <Input
            type="number"
            min={1}
            value={iterations}
            onChange={(e) => setIterations(Number(e.target.value))}
          />
        </Field>
        <Field label={t("train.epochs")}>
          <Input
            type="number"
            min={1}
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
          />
        </Field>
        <Field label={t("agent.maxCost")}>
          <Input
            type="number"
            min={0}
            step={0.1}
            value={cost}
            onChange={(e) => setCost(Number(e.target.value))}
          />
        </Field>
        <Field label={t("agent.approval")}>
          <Select value={approval} onChange={(e) => setApproval(e.target.value as ApprovalMode)}>
            {(["never", "each_iteration", "family_change", "budget_pct"] as const).map((m) => (
              <option key={m} value={m}>
                {t(`agent.approvalMode.${m}`)}
              </option>
            ))}
          </Select>
        </Field>
      </div>
      <Button
        className="mt-4"
        variant="ai"
        loading={launch.isPending}
        onClick={() => launch.mutate()}
      >
        {t("agent.launch")}
      </Button>
      <ErrorNote error={launch.error} />
    </Card>
  );
}

function AgentRunView({ agentId }: { agentId: string }) {
  const { t, i18n } = useTranslation();
  const projectId = useProjectId();
  const qc = useQueryClient();
  const { data: ar, error } = useAgentRun(agentId);
  const [live, setLive] = useState<LogEntry[]>([]);

  const onMessage = useCallback((msg: { entry?: LogEntry }) => {
    const entry = msg.entry;
    if (entry) setLive((prev) => (prev.some((e) => e.ts === entry.ts) ? prev : [...prev, entry]));
  }, []);
  useEngineSocket<{ entry?: LogEntry }>(`/api/v1/agent/runs/${agentId}/log`, onMessage);

  const act = useMutation({
    mutationFn: async (action: "approve" | "reject" | "stop") => {
      const api = await getApiClient();
      const path = { params: { path: { agent_id: agentId } } };
      if (action === "stop")
        return unwrap(await api.POST("/api/v1/agent/runs/{agent_id}/stop", path));
      const url =
        action === "approve"
          ? "/api/v1/agent/runs/{agent_id}/approve"
          : "/api/v1/agent/runs/{agent_id}/reject";
      return unwrap(await api.POST(url, { ...path, body: { comment: null } }));
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["agent", agentId] }),
  });

  if (error) return <ErrorNote error={error} />;
  if (!ar) return <Spinner />;
  const log = (live.length ? live : ((ar.log ?? []) as unknown as LogEntry[])).filter(
    (e) => e.kind !== "observation",
  );
  const running = !TERMINAL.has(ar.state);

  return (
    <div className="space-y-4">
      <Card>
        <CardTitle className="flex flex-wrap items-center gap-2">
          {t("agent.runTitle")} <span className="font-mono text-xs">{ar.id}</span>
          <Badge tone={STATE_TONE[ar.state]}>{t(`agent.state.${ar.state}`)}</Badge>
          {ar.fallback && <Badge tone="warn">{t("agent.fallback")}</Badge>}
        </CardTitle>
        <dl className="grid gap-2 text-sm sm:grid-cols-4">
          <div>
            <dt className="text-muted">{t("agent.iterations")}</dt>
            <dd className="font-semibold">{ar.iterations}</dd>
          </div>
          <div>
            <dt className="text-muted">{t("agent.trials")}</dt>
            <dd className="font-semibold">{ar.trials}</dd>
          </div>
          <div>
            <dt className="text-muted">{t("agent.cost")}</dt>
            <dd className="font-semibold">${formatNumber(ar.cost_usd, i18n.language, 4)}</dd>
          </div>
          <div>
            <dt className="text-muted">{t("agent.stopReason")}</dt>
            <dd>{ar.stop_reason ?? "—"}</dd>
          </div>
        </dl>
        {running && (
          <Button
            className="mt-3"
            size="sm"
            variant="secondary"
            loading={act.isPending}
            onClick={() => act.mutate("stop")}
          >
            <Square className="h-3 w-3" aria-hidden="true" />
            {t("agent.stop")}
          </Button>
        )}
        <ErrorNote error={act.error} />
      </Card>

      {ar.state === "awaiting_approval" && (
        <AiSuggestion
          title={t("agent.approvalNeeded")}
          onAccept={() => act.mutate("approve")}
          onReject={() => act.mutate("reject")}
        >
          <p>{log.filter((e) => e.kind === "approval").at(-1)?.message}</p>
        </AiSuggestion>
      )}

      {Object.keys(ar.test_metrics ?? {}).length > 0 && (
        <Card>
          <CardTitle>{t("agent.result")}</CardTitle>
          <dl className="grid gap-2 text-sm sm:grid-cols-4">
            {Object.entries(ar.test_metrics ?? {})
              .slice(0, 8)
              .map(([k, v]) => (
                <div key={k}>
                  <dt className="text-muted">{k}</dt>
                  <dd className="font-semibold">{formatNumber(v, i18n.language)}</dd>
                </div>
              ))}
          </dl>
          {ar.best_run_id && (
            <Link
              to="/projects/$projectId/runs/$runId"
              params={{ projectId, runId: ar.best_run_id }}
              className="mt-3 inline-block text-sm underline"
            >
              {t("agent.openBest")}
            </Link>
          )}
        </Card>
      )}

      <Card>
        <CardTitle>{t("agent.log")}</CardTitle>
        <ol className="space-y-2 text-sm" aria-live="polite">
          {log.map((e, i) => (
            <li
              key={`${e.ts}-${i}`}
              className={e.kind === "decision" ? "rounded-pt bg-copilot-bg/40 p-2" : "p-2"}
            >
              <Badge className="mr-2">{t(`agent.kind.${e.kind}`, { defaultValue: e.kind })}</Badge>
              {e.message}
            </li>
          ))}
        </ol>
      </Card>
    </div>
  );
}

export function AgentPage() {
  const projectId = useProjectId();
  const navigate = useNavigate();
  const { agent } = useSearch({ from: "/projects/$projectId/agent" });
  return agent ? (
    <AgentRunView agentId={agent} />
  ) : (
    <LaunchForm
      onLaunched={(id) =>
        void navigate({
          to: "/projects/$projectId/agent",
          params: { projectId },
          search: { agent: id },
        })
      }
    />
  );
}
