/** Cola del Team Server (RF-SRV-04): workers, su hardware y los estudios en curso. */
import { Link } from "@tanstack/react-router";
import { Cpu, Gpu } from "lucide-react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Card,
  CardTitle,
  EmptyState,
  ErrorNote,
  PageHeader,
  Spinner,
  Table,
  Td,
  Th,
} from "@/components/ui";
import { useSession } from "@/features/auth/session";
import { useQueue } from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

export function QueuePage() {
  const { t, i18n } = useTranslation();
  const me = useSession();
  const queue = useQueue(Boolean(me));
  if (!me) return <EmptyState>{t("queue.serverOnly")}</EmptyState>;
  const view = queue.data;
  return (
    <div className="space-y-4">
      <PageHeader
        title={t("queue.title")}
        description={
          view
            ? t(view.mode === "queue" ? "queue.modeQueue" : "queue.modeLocal", {
                user: view.quota_per_user,
                workspace: view.quota_per_workspace,
              })
            : undefined
        }
      />
      {queue.isPending && <Spinner />}
      <ErrorNote error={queue.error} />
      {view && (
        <>
          <Card>
            <CardTitle>{t("queue.workers")}</CardTitle>
            {view.workers.length === 0 ? (
              <EmptyState>
                {view.mode === "queue" ? t("queue.noWorkers") : t("queue.inProcess")}
              </EmptyState>
            ) : (
              <Table>
                <thead>
                  <tr>
                    <Th>{t("queue.worker")}</Th>
                    <Th>{t("queue.queues")}</Th>
                    <Th>{t("queue.hardware")}</Th>
                    <Th>{t("queue.state")}</Th>
                  </tr>
                </thead>
                <tbody>
                  {view.workers.map((w) => (
                    <tr key={w.worker}>
                      <Td className="font-mono text-xs">{w.worker}</Td>
                      <Td className="space-x-1">
                        {w.queues.map((q) => (
                          <Badge key={q}>{q}</Badge>
                        ))}
                      </Td>
                      <Td className="text-xs">
                        {w.gpus.length > 0 ? (
                          <span className="flex items-center gap-1">
                            <Gpu className="h-4 w-4" aria-hidden="true" />
                            {w.gpus.map((g) => String(g.name ?? "GPU")).join(", ")}
                          </span>
                        ) : (
                          <span className="flex items-center gap-1">
                            <Cpu className="h-4 w-4" aria-hidden="true" />
                            {t("queue.cpuOnly")}
                          </span>
                        )}
                      </Td>
                      <Td>
                        {w.busy_job ? (
                          <Badge tone="warn">{t("queue.busy")}</Badge>
                        ) : (
                          <Badge tone="ok">{t("queue.idle")}</Badge>
                        )}
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Card>
          <Card>
            <CardTitle>{t("queue.studies")}</CardTitle>
            {view.jobs.length === 0 ? (
              <EmptyState>{t("queue.empty")}</EmptyState>
            ) : (
              <Table>
                <thead>
                  <tr>
                    <Th>{t("queue.study")}</Th>
                    <Th>{t("queue.queue")}</Th>
                    <Th>{t("queue.state")}</Th>
                    <Th>{t("queue.worker")}</Th>
                    <Th>{t("queue.since")}</Th>
                  </tr>
                </thead>
                <tbody>
                  {view.jobs.map((j) => (
                    <tr key={j.id}>
                      <Td>
                        {j.refs.project_id ? (
                          <Link
                            to="/projects/$projectId/experiments"
                            params={{ projectId: j.refs.project_id }}
                            search={{ job: j.id }}
                            className="font-mono text-xs underline"
                          >
                            {j.refs.study_id ?? j.id}
                          </Link>
                        ) : (
                          j.id
                        )}
                      </Td>
                      <Td>
                        <Badge>{j.refs.queue ?? "local"}</Badge>
                      </Td>
                      <Td>
                        <Badge tone={j.status === "running" ? "warn" : "neutral"}>
                          {t(`status.${j.status}`)}
                        </Badge>
                      </Td>
                      <Td className="font-mono text-xs">{j.worker ?? "—"}</Td>
                      <Td className="text-xs">{formatDate(j.created_at, i18n.language)}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
