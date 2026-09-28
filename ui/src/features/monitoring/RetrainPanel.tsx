/** Reentrenamiento automático (RF-MON-05): política, ejecuciones y aprobación. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
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
  Table,
  Td,
  Th,
} from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap } from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

import { type Deployment, useStreamSources } from "./hooks";

type Policy = Schemas["RetrainPolicy"];
type RetrainRun = Schemas["RetrainRun"] & { id: string };
const RUN_TONE = {
  running: "brand",
  awaiting_approval: "warn",
  promoted: "ok",
  not_improved: "neutral",
  rejected: "neutral",
  skipped: "neutral",
  failed: "bad",
} as const;

function useRetrain(projectId: string) {
  const qc = useQueryClient();
  const policy = useQuery({
    queryKey: ["projects", projectId, "retrain-policy"],
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/retrain-policy", {
          params: { path: { project_id: projectId } },
        }),
      ) as Policy | null,
  });
  const runs = useQuery({
    queryKey: ["projects", projectId, "retrain-runs"],
    refetchInterval: 5000,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/retrain-runs", {
          params: { path: { project_id: projectId } },
        }),
      ) as RetrainRun[],
  });
  const refresh = () => void qc.invalidateQueries({ queryKey: ["projects", projectId] });
  const save = useMutation({
    mutationFn: async (body: Schemas["RetrainPolicyBody"]) =>
      unwrap(
        await (
          await getApiClient()
        ).PUT("/api/v1/projects/{project_id}/retrain-policy", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
    onSuccess: refresh,
  });
  const runNow = useMutation({
    mutationFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/retrain-policy/run", {
          params: { path: { project_id: projectId } },
        }),
      ),
    onSuccess: refresh,
  });
  const decide = useMutation({
    mutationFn: async ({ id, ok }: { id: string; ok: boolean }) => {
      const api = await getApiClient();
      const params = { params: { path: { retrain_run_id: id } } };
      return unwrap(
        ok
          ? await api.POST("/api/v1/retrain-runs/{retrain_run_id}/approve", params)
          : await api.POST("/api/v1/retrain-runs/{retrain_run_id}/reject", params),
      );
    },
    onSuccess: refresh,
  });
  return { policy, runs, save, runNow, decide };
}

function PolicyForm({
  projectId,
  deployments,
  policy,
  onSave,
  saving,
}: {
  projectId: string;
  deployments: Deployment[];
  policy: Policy | null;
  onSave: (body: Schemas["RetrainPolicyBody"]) => void;
  saving: boolean;
}) {
  const { t } = useTranslation();
  const trig = (type: string) => (policy?.triggers ?? []).find((x) => x["type"] === type);
  const [deployment, setDeployment] = useState(policy?.deployment_id ?? deployments[0]?.id ?? "");
  const [drift, setDrift] = useState(String(trig("drift")?.["min_severity"] ?? "medium"));
  const [useDrift, setUseDrift] = useState(Boolean(trig("drift")) || !policy);
  const [volume, setVolume] = useState(String(trig("volume")?.["min_rows"] ?? ""));
  const [cron, setCron] = useState(String(trig("cron")?.["expr"] ?? ""));
  const [approval, setApproval] = useState(policy?.require_approval ?? true);
  const [enabled, setEnabled] = useState(policy?.enabled ?? true);
  const [sourceIds, setSourceIds] = useState<string[]>(policy?.source_ids ?? []);
  const streams = useStreamSources(projectId);
  return (
    <form
      className="grid gap-3 sm:grid-cols-3"
      onSubmit={(e) => {
        e.preventDefault();
        const triggers: Schemas["Trigger"][] = [];
        if (useDrift) triggers.push({ type: "drift", min_severity: drift as never });
        if (volume) triggers.push({ type: "volume", min_rows: Number(volume) });
        if (cron) triggers.push({ type: "cron", expr: cron });
        onSave({
          deployment_id: deployment || null,
          source_ids: sourceIds,
          triggers,
          require_approval: approval,
          enabled,
          budget: (policy?.budget as Record<string, unknown> | undefined) ?? { max_trials: 3 },
          min_improvement: policy?.min_improvement ?? 0,
          holdout_fraction: policy?.holdout_fraction ?? 0.3,
          cooldown_s: policy?.cooldown_s ?? 3600,
          use_feedback: policy?.use_feedback ?? true,
        });
      }}
    >
      <Field label={t("retrain.deployment")}>
        <Select value={deployment} onChange={(e) => setDeployment(e.target.value)}>
          {deployments.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </Select>
      </Field>
      <Field label={t("retrain.onDrift")}>
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={useDrift}
            onChange={(e) => setUseDrift(e.target.checked)}
          />
          <Select value={drift} onChange={(e) => setDrift(e.target.value)} disabled={!useDrift}>
            <option value="medium">{t("driftSeverity.medium")}</option>
            <option value="high">{t("driftSeverity.high")}</option>
          </Select>
        </div>
      </Field>
      {(streams.data ?? []).length > 0 && (
        <fieldset className="sm:col-span-3">
          <legend className="text-sm font-semibold">{t("retrain.sources")}</legend>
          <div className="mt-1 flex flex-wrap gap-3">
            {(streams.data ?? []).map(({ source }) => (
              <label key={source.id} className="flex items-center gap-1 text-sm">
                <input
                  type="checkbox"
                  checked={sourceIds.includes(source.id ?? "")}
                  onChange={() =>
                    setSourceIds((ids) =>
                      ids.includes(source.id ?? "")
                        ? ids.filter((x) => x !== source.id)
                        : [...ids, source.id ?? ""],
                    )
                  }
                />
                {source.name}
              </label>
            ))}
          </div>
        </fieldset>
      )}
      <Field label={t("retrain.volume")} hint={t("retrain.volumeHint")}>
        <Input inputMode="numeric" value={volume} onChange={(e) => setVolume(e.target.value)} />
      </Field>
      <Field label={t("retrain.cron")} hint={t("retrain.cronHint")}>
        <Input value={cron} onChange={(e) => setCron(e.target.value)} placeholder="0 3 * * 1" />
      </Field>
      <label className="flex items-center gap-2 self-end pb-2 text-sm">
        <input type="checkbox" checked={approval} onChange={(e) => setApproval(e.target.checked)} />
        {t("retrain.approval")}
      </label>
      <label className="flex items-center gap-2 self-end pb-2 text-sm">
        <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
        {t("retrain.enabled")}
      </label>
      <div className="sm:col-span-3">
        <Button type="submit" loading={saving}>
          {t("retrain.save")}
        </Button>
      </div>
    </form>
  );
}

export function RetrainPanel({
  projectId,
  deployments,
}: {
  projectId: string;
  deployments: Deployment[];
}) {
  const { t, i18n } = useTranslation();
  const { policy, runs, save, runNow, decide } = useRetrain(projectId);
  return (
    <Card>
      <CardTitle className="flex flex-wrap items-center justify-between gap-2">
        {t("retrain.title")}
        {policy.data && (
          <Button
            size="sm"
            variant="ghost"
            loading={runNow.isPending}
            onClick={() => runNow.mutate()}
          >
            <RefreshCw className="h-4 w-4" aria-hidden="true" />
            {t("retrain.runNow")}
          </Button>
        )}
      </CardTitle>
      <p className="mb-4 text-sm text-muted">{t("retrain.hint")}</p>
      <ErrorNote error={policy.error ?? save.error ?? runNow.error ?? decide.error} />
      {!policy.isPending && (
        <PolicyForm
          projectId={projectId}
          key={policy.data?.id ?? "nueva"}
          deployments={deployments}
          policy={policy.data ?? null}
          onSave={(body) => save.mutate(body)}
          saving={save.isPending}
        />
      )}
      <div className="mt-4">
        {runs.data?.length === 0 && <EmptyState>{t("retrain.noRuns")}</EmptyState>}
        {runs.data && runs.data.length > 0 && (
          <Table>
            <thead>
              <tr>
                <Th>{t("retrain.when")}</Th>
                <Th>{t("retrain.trigger")}</Th>
                <Th>{t("retrain.status")}</Th>
                <Th>{t("retrain.result")}</Th>
                <Th />
              </tr>
            </thead>
            <tbody>
              {runs.data.map((r) => {
                const ch = r.challenge as { metric?: string; improvement?: number } | null;
                return (
                  <tr key={r.id}>
                    <Td className="text-xs">{formatDate(r.created_at, i18n.language)}</Td>
                    <Td className="text-xs">
                      {String((r.trigger as { type?: string }).type ?? "—")}
                    </Td>
                    <Td>
                      <Badge tone={RUN_TONE[r.status ?? "running"]}>
                        {t(`retrain.st.${r.status ?? "running"}`)}
                      </Badge>
                    </Td>
                    <Td className="text-xs">
                      {ch?.metric
                        ? `${ch.metric} ${ch.improvement !== undefined && ch.improvement >= 0 ? "+" : ""}${(ch.improvement ?? 0).toFixed(4)}`
                        : (r.error ?? "—")}
                    </Td>
                    <Td className="space-x-1 text-right">
                      {r.status === "awaiting_approval" && (
                        <>
                          <Button size="sm" onClick={() => decide.mutate({ id: r.id, ok: true })}>
                            {t("retrain.approve")}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => decide.mutate({ id: r.id, ok: false })}
                          >
                            {t("retrain.reject")}
                          </Button>
                        </>
                      )}
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </div>
    </Card>
  );
}
