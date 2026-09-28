/** Edición concurrente de un proyecto de equipo (RF-SRV-03): qué cambió en el servidor, qué
 * cambió acá y qué cambió en los dos lados. Los conflictos se resuelven uno por uno. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowDownToLine, ArrowUpFromLine, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  ErrorNote,
  Select,
  Spinner,
  Table,
  Td,
  Th,
} from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import { type Project, type Schemas, unwrap, useJob, useRemoteServers } from "@/lib/api/hooks";
import { getPlatform } from "@/lib/platform/bridge";

type Status = Schemas["SyncStatus"];
type Choice = "theirs" | "mine";
/** Resultado del job de sincronización (`PullOutcome` del Engine). */
interface Outcome {
  pulled: number;
  pushed: number;
  conflicts_left: string[];
  downloaded: number;
}
const TONE = {
  synced: "ok",
  pull: "brand",
  push: "brand",
  server_only: "brand",
  local_only: "neutral",
  conflict: "warn",
} as const;

export function TeamSync({ project }: { project: Project }) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const desktop = getPlatform().kind === "desktop";
  const servers = useRemoteServers(desktop);
  const list = Array.isArray(servers.data) ? servers.data : [];
  const [picked, setPicked] = useState("");
  const server = picked || list[0]?.name || "";
  const [choices, setChoices] = useState<Record<string, Choice>>({});
  const key = ["projects", project.id, "remote-status", server] as const;
  const status = useQuery({
    queryKey: key,
    enabled: desktop && Boolean(server),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/remote/status", {
          params: { path: { project_id: project.id }, query: { server } },
        }),
      ) as Status,
  });
  const run = useMutation({
    mutationFn: async (action: "pull" | "push") => {
      const api = await getApiClient();
      const params = { params: { path: { project_id: project.id } } };
      return unwrap(
        action === "pull"
          ? await api.POST("/api/v1/projects/{project_id}/remote/pull", {
              ...params,
              body: { server, resolutions: choices },
            })
          : await api.POST("/api/v1/projects/{project_id}/remote/push", {
              ...params,
              body: { server },
            }),
      ) as { id: string };
    },
    onSuccess: () => setChoices({}), // ya viajaron en el pedido
  });
  const job = useJob(run.data?.id);
  const finished = job.data?.status === "succeeded" || job.data?.status === "failed";
  useEffect(() => {
    if (finished) void qc.invalidateQueries({ queryKey: ["projects", project.id] });
  }, [finished, qc, project.id]);

  if (!desktop || list.length === 0) return null;
  const items = status.data?.items ?? [];
  const count = (s: string) => items.filter((i) => i.state === s).length;
  const conflicts = items.filter((i) => i.state === "conflict");
  const busy = run.isPending || (Boolean(run.data) && !finished);
  const result = job.data?.status === "succeeded" ? (job.data.result as Outcome) : null;
  return (
    <Card>
      <CardTitle className="flex items-center justify-between gap-2">
        {t("teamSync.title")}
        {list.length > 1 && (
          <Select
            value={server}
            onChange={(e) => setPicked(e.target.value)}
            aria-label={t("teamSync.server")}
            className="w-48"
          >
            {list.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </Select>
        )}
      </CardTitle>
      {status.isPending ? (
        <Spinner />
      ) : (
        <>
          <p className="text-sm" role="status">
            {count("conflict")
              ? t("teamSync.conflicts", { count: count("conflict") })
              : count("pull") + count("server_only") + count("push") + count("local_only")
                ? t("teamSync.pending", {
                    pull: count("pull") + count("server_only"),
                    push: count("push") + count("local_only"),
                  })
                : t("teamSync.synced")}
          </p>
          {items.some((i) => i.state !== "synced") && (
            <Table>
              <thead>
                <tr>
                  <Th>{t("teamSync.entity")}</Th>
                  <Th>{t("teamSync.state")}</Th>
                  <Th>{t("teamSync.resolution")}</Th>
                </tr>
              </thead>
              <tbody>
                {items
                  .filter((i) => i.state !== "synced")
                  .map((i) => (
                    <tr key={i.id}>
                      <Td>
                        <span className="text-xs text-muted">
                          {t(`teamSync.kinds.${i.kind}`, { defaultValue: i.kind })}
                        </span>{" "}
                        {i.name ?? <span className="font-mono text-xs">{i.id}</span>}
                      </Td>
                      <Td>
                        <Badge tone={TONE[i.state]}>{t(`teamSync.states.${i.state}`)}</Badge>
                      </Td>
                      <Td>
                        {i.state === "conflict" ? (
                          <Select
                            value={choices[i.id] ?? ""}
                            aria-label={t("teamSync.resolveFor", { name: i.name ?? i.id })}
                            onChange={(e) =>
                              setChoices((c) => ({ ...c, [i.id]: e.target.value as Choice }))
                            }
                          >
                            <option value="">{t("teamSync.undecided")}</option>
                            <option value="theirs">{t("teamSync.theirs")}</option>
                            <option value="mine">{t("teamSync.mine")}</option>
                          </Select>
                        ) : (
                          "—"
                        )}
                      </Td>
                    </tr>
                  ))}
              </tbody>
            </Table>
          )}
        </>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        <Button variant="ghost" size="sm" onClick={() => void status.refetch()}>
          <RefreshCw className="h-4 w-4" aria-hidden="true" />
          {t("teamSync.refresh")}
        </Button>
        <Button
          variant="secondary"
          loading={busy && run.variables === "pull"}
          disabled={busy || !(count("pull") + count("server_only") + Object.keys(choices).length)}
          onClick={() => run.mutate("pull")}
        >
          <ArrowDownToLine className="h-4 w-4" aria-hidden="true" />
          {t("teamSync.pull")}
        </Button>
        <Button
          loading={busy && run.variables === "push"}
          disabled={busy || !(count("push") + count("local_only")) || conflicts.length > 0}
          onClick={() => run.mutate("push")}
        >
          <ArrowUpFromLine className="h-4 w-4" aria-hidden="true" />
          {t("teamSync.push")}
        </Button>
      </div>
      {result && (
        <p className="mt-2 text-xs text-muted">
          {t("teamSync.done", { pulled: result.pulled, pushed: result.pushed })}
        </p>
      )}
      <ErrorNote
        error={status.error ?? run.error ?? (job.data?.status === "failed" ? job.data.error : null)}
      />
    </Card>
  );
}
