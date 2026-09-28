/** Monitoreo del modelo en uso (RF-MON-01..04): deployments, drift por feature y alertas. */
import { Activity, BellRing, Check } from "lucide-react";
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
  Table,
  Td,
  Th,
} from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { formatDate, formatNumber } from "@/lib/format";

import {
  type Deployment,
  type DriftReport,
  useAlertAction,
  useAlerts,
  useCheckDeployment,
  useDeployments,
  useDriftReports,
  useUpdateDeployment,
} from "./hooks";

const TONE = { none: "ok", low: "neutral", medium: "warn", high: "bad" } as const;
type Sev = keyof typeof TONE;

interface FeatureRow {
  feature: string;
  kind: string;
  severity: Sev;
  psi?: number | null;
  ks_pvalue?: number | null;
  js?: number | null;
  chi2_pvalue?: number | null;
  unseen_fraction?: number | null;
  reference_mean?: number | null;
  current_mean?: number | null;
}

function SeverityBadge({ value }: { value: string }) {
  const { t } = useTranslation();
  const sev = (value in TONE ? value : "none") as Sev;
  return <Badge tone={TONE[sev]}>{t(`driftSeverity.${sev}`)}</Badge>;
}

function Report({ report }: { report: DriftReport }) {
  const { t, i18n } = useTranslation();
  const metrics = report.metrics as {
    data?: { features?: FeatureRow[]; share_drifted?: number; n_current?: number };
    performance?: {
      metric: string;
      current: number | null;
      baseline: number | null;
      n_labeled: number;
      severity: Sev;
    } | null;
  };
  const features = [...(metrics.data?.features ?? [])].sort(
    (a, b) =>
      ["none", "low", "medium", "high"].indexOf(b.severity) -
      ["none", "low", "medium", "high"].indexOf(a.severity),
  );
  const n = (v: number | null | undefined, d = 3) =>
    v === null || v === undefined ? "—" : formatNumber(v, i18n.language, d);
  const perf = metrics.performance;
  return (
    <div className="space-y-4">
      <p className="flex flex-wrap items-center gap-2 text-sm">
        <SeverityBadge value={report.severity ?? "none"} />
        {t("monitoring.window", {
          from: formatDate(report.window_start, i18n.language),
          to: formatDate(report.window_end, i18n.language),
          n: metrics.data?.n_current ?? 0,
        })}
      </p>
      {perf && (
        <p className="text-sm">
          <SeverityBadge value={perf.severity} />{" "}
          {t("monitoring.performance", {
            metric: perf.metric,
            current: n(perf.current),
            baseline: n(perf.baseline),
            n: perf.n_labeled,
          })}
        </p>
      )}
      <Table>
        <thead>
          <tr>
            <Th>{t("monitoring.feature")}</Th>
            <Th>{t("monitoring.severity")}</Th>
            <Th>PSI</Th>
            <Th>JS</Th>
            <Th>{t("monitoring.pvalue")}</Th>
            <Th>{t("monitoring.change")}</Th>
          </tr>
        </thead>
        <tbody>
          {features.map((f) => (
            <tr key={f.feature}>
              <Td className="font-mono text-xs">{f.feature}</Td>
              <Td>
                <SeverityBadge value={f.severity} />
              </Td>
              <Td className="text-xs">{n(f.psi)}</Td>
              <Td className="text-xs">{n(f.js)}</Td>
              <Td className="text-xs">
                {n(f.kind === "numeric" ? f.ks_pvalue : f.chi2_pvalue, 4)}
              </Td>
              <Td className="text-xs">
                {f.kind === "numeric"
                  ? `${n(f.reference_mean, 2)} → ${n(f.current_mean, 2)}`
                  : f.unseen_fraction
                    ? t("monitoring.unseen", { pct: n((f.unseen_fraction ?? 0) * 100, 1) })
                    : "—"}
              </Td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
}

function DeploymentCard({ projectId, dep }: { projectId: string; dep: Deployment }) {
  const { t } = useTranslation();
  const reports = useDriftReports(dep.id);
  const check = useCheckDeployment(projectId, dep.id);
  const update = useUpdateDeployment(projectId);
  const [webhook, setWebhook] = useState("");
  const [email, setEmail] = useState(
    ((dep.monitoring as { email?: string[] }).email ?? []).join(", "),
  );
  const latest = reports.data?.[0];
  return (
    <Card>
      <CardTitle className="flex flex-wrap items-center gap-2">
        <Activity className="h-4 w-4" aria-hidden="true" />
        {dep.name}
        <Badge tone={dep.status === "active" ? "ok" : "neutral"}>
          {t(`monitoring.status.${dep.status ?? "active"}`)}
        </Badge>
        <span className="font-mono text-xs text-muted">{dep.endpoint}</span>
      </CardTitle>
      <div className="mb-4 flex flex-wrap gap-2">
        <Button size="sm" loading={check.isPending} onClick={() => check.mutate()}>
          {t("monitoring.checkNow")}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() =>
            update.mutate({
              id: dep.id,
              patch: { status: dep.status === "active" ? "stopped" : "active" },
            })
          }
        >
          {dep.status === "active" ? t("monitoring.stop") : t("monitoring.start")}
        </Button>
      </div>
      <ErrorNote error={check.error ?? update.error ?? reports.error} />
      {reports.isPending && <Spinner />}
      {latest ? <Report report={latest} /> : <EmptyState>{t("monitoring.noReports")}</EmptyState>}
      <form
        className="mt-4 grid gap-3 border-t border-line pt-4 sm:grid-cols-3"
        onSubmit={(e) => {
          e.preventDefault();
          const list = email
            .split(",")
            .map((x) => x.trim())
            .filter(Boolean);
          update.mutate({
            id: dep.id,
            patch: {
              monitoring: { ...(dep.monitoring as object), email: list } as never,
              ...(webhook ? { webhook_url: webhook } : {}),
            },
          });
          setWebhook("");
        }}
      >
        <Field label={t("monitoring.email")} hint={t("monitoring.emailHint")}>
          <Input value={email} onChange={(e) => setEmail(e.target.value)} />
        </Field>
        <Field label={t("monitoring.webhook")} hint={t("monitoring.webhookHint")}>
          <Input
            type="url"
            value={webhook}
            onChange={(e) => setWebhook(e.target.value)}
            placeholder="https://"
          />
        </Field>
        <div className="flex items-end">
          <Button type="submit" variant="secondary" loading={update.isPending}>
            {t("monitoring.saveChannels")}
          </Button>
        </div>
      </form>
    </Card>
  );
}

function Alerts({ projectId }: { projectId: string }) {
  const { t, i18n } = useTranslation();
  const alerts = useAlerts(projectId);
  const act = useAlertAction(projectId);
  const [filter, setFilter] = useState("open");
  const items = (alerts.data ?? []).filter((a) => filter === "all" || a.status === filter);
  return (
    <Card>
      <CardTitle className="flex items-center gap-2">
        <BellRing className="h-4 w-4" aria-hidden="true" />
        {t("monitoring.alerts")}
      </CardTitle>
      <div className="mb-3 w-48">
        <Select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label={t("monitoring.filter")}
        >
          <option value="open">{t("monitoring.alertStatus.open")}</option>
          <option value="acknowledged">{t("monitoring.alertStatus.acknowledged")}</option>
          <option value="resolved">{t("monitoring.alertStatus.resolved")}</option>
          <option value="all">{t("monitoring.all")}</option>
        </Select>
      </div>
      <ErrorNote error={alerts.error ?? act.error} />
      {items.length === 0 ? (
        <EmptyState>{t("monitoring.noAlerts")}</EmptyState>
      ) : (
        <ul className="space-y-2">
          {items.map((a) => (
            <li key={a.id} className="rounded-pt border border-line p-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <SeverityBadge value={a.severity} />
                <span className="font-semibold">{a.title}</span>
                <span className="text-xs text-muted">
                  {formatDate(a.created_at, i18n.language)}
                </span>
                <span className="ml-auto flex gap-1">
                  {a.status === "open" && (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => act.mutate({ id: a.id, action: "acknowledge" })}
                    >
                      {t("monitoring.acknowledge")}
                    </Button>
                  )}
                  {a.status !== "resolved" && (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => act.mutate({ id: a.id, action: "resolve" })}
                    >
                      <Check className="h-4 w-4" aria-hidden="true" />
                      {t("monitoring.resolve")}
                    </Button>
                  )}
                </span>
              </div>
              {a.message && <p className="mt-1 text-muted">{a.message}</p>}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export function MonitoringPage() {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const deployments = useDeployments(projectId);
  return (
    <div className="space-y-4">
      <Alerts projectId={projectId} />
      {deployments.isPending && <Spinner />}
      <ErrorNote error={deployments.error} />
      {deployments.data?.length === 0 && <EmptyState>{t("monitoring.noDeployments")}</EmptyState>}
      {deployments.data?.map((d) => (
        <DeploymentCard key={d.id} projectId={projectId} dep={d} />
      ))}
    </div>
  );
}
