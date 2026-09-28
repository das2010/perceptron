/** Estado del Team Server para el Admin (RF-SRV-06): almacenamiento, cuotas y workers. */
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Card,
  CardTitle,
  EmptyState,
  ErrorNote,
  Spinner,
  Table,
  Td,
  Th,
} from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap } from "@/lib/api/hooks";

type Status = Schemas["SystemStatus"];

const size = (b: number) =>
  b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(2)} GB` : `${(b / 1024 ** 2).toFixed(1)} MB`;

export function SystemTab() {
  const { t } = useTranslation();
  const q = useQuery({
    queryKey: ["admin", "system"],
    refetchInterval: 15_000,
    queryFn: async () => unwrap(await (await getApiClient()).GET("/api/v1/admin/system")) as Status,
  });
  if (q.isPending) return <Spinner />;
  if (q.error || !q.data) return <ErrorNote error={q.error} />;
  const s = q.data;
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle>{t("system.storage")}</CardTitle>
        <p className="mb-2 text-sm">
          {t("system.usage", { used: size(s.used_bytes), free: size(s.free_bytes) })}{" "}
          <span className="font-mono text-xs text-muted">{s.workspace_path}</span>
        </p>
        <Table>
          <thead>
            <tr>
              <Th>{t("system.project")}</Th>
              <Th>{t("system.size")}</Th>
            </tr>
          </thead>
          <tbody>
            {s.projects.map((p) => (
              <tr key={p.project_id}>
                <Td>{p.name}</Td>
                <Td>{size(p.bytes)}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
      <Card>
        <CardTitle>{t("system.quotas")}</CardTitle>
        <dl className="grid gap-1 text-sm sm:grid-cols-[18rem_1fr]">
          {Object.entries(s.quotas).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-muted">{t(`system.quota.${k}`, { defaultValue: k })}</dt>
              <dd className="font-semibold">{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-2 text-xs text-muted">{t("system.quotasHint")}</p>
      </Card>
      <Card>
        <CardTitle className="flex items-center gap-2">
          {t("system.workers")}{" "}
          <Badge>{t(`system.mode.${s.queue_mode}`, { defaultValue: s.queue_mode })}</Badge>
        </CardTitle>
        {s.workers.length === 0 ? (
          <EmptyState>{t("system.noWorkers")}</EmptyState>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>{t("system.worker")}</Th>
                <Th>{t("system.queues")}</Th>
                <Th>GPU</Th>
                <Th>{t("system.state")}</Th>
                <Th>{t("system.heartbeat")}</Th>
              </tr>
            </thead>
            <tbody>
              {s.workers.map((w, i) => {
                const worker = w as {
                  worker?: string;
                  queues?: string[];
                  gpus?: unknown[];
                  busy_job?: string | null;
                  seen_s_ago?: number;
                };
                return (
                  <tr key={worker.worker ?? i}>
                    <Td>{worker.worker ?? "—"}</Td>
                    <Td>{(worker.queues ?? []).join(", ")}</Td>
                    <Td>{worker.gpus?.length ?? 0}</Td>
                    <Td>
                      <Badge tone={worker.busy_job ? "brand" : "ok"}>
                        {t(worker.busy_job ? "system.busy" : "system.idle")}
                      </Badge>
                    </Td>
                    <Td>{t("system.secondsAgo", { s: worker.seen_s_ago ?? 0 })}</Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}
