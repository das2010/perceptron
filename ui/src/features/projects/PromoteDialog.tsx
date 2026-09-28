/** Promover un proyecto local a proyecto de equipo (RF-PRJ-04): se elige el Team Server, los
 * datasets y los runs que suben; el resto queda solo en este desktop. */
import { useMutation } from "@tanstack/react-query";
import { Users } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, Dialog, ErrorNote, Field, Select } from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import {
  type Project,
  unwrap,
  useDatasets,
  useJob,
  useRemoteServers,
  useRuns,
} from "@/lib/api/hooks";
import { getPlatform } from "@/lib/platform/bridge";

function Checklist({
  items,
  selected,
  onToggle,
  label,
}: {
  items: { id: string; text: string }[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  label: string;
}) {
  return (
    <fieldset className="max-h-40 overflow-y-auto rounded-pt border border-line p-2 text-sm">
      <legend className="px-1 text-xs text-muted">{label}</legend>
      {items.map((i) => (
        <label key={i.id} className="flex items-center gap-2 py-0.5">
          <input type="checkbox" checked={selected.has(i.id)} onChange={() => onToggle(i.id)} />
          <span className="truncate font-mono text-xs">{i.text}</span>
        </label>
      ))}
    </fieldset>
  );
}

export function PromoteButton({ project }: { project: Project }) {
  const { t } = useTranslation();
  const desktop = getPlatform().kind === "desktop";
  const servers = useRemoteServers(desktop);
  const [open, setOpen] = useState(false);
  const datasets = useDatasets(project.id).data ?? [];
  const runs = (useRuns(project.id).data ?? []).filter((r) => r.status === "succeeded");
  const [server, setServer] = useState("");
  const [dvs, setDvs] = useState<Set<string>>(new Set());
  const [runIds, setRunIds] = useState<Set<string>>(new Set());
  const promote = useMutation({
    mutationFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/remote/promote", {
          params: { path: { project_id: project.id } },
          body: {
            server: server || (servers.data?.[0]?.name ?? ""),
            dataset_version_ids: [...dvs],
            run_ids: [...runIds],
          },
        }),
      ) as { id: string },
  });
  const job = useJob(promote.data?.id ?? "");
  const list = Array.isArray(servers.data) ? servers.data : [];
  if (!desktop || list.length === 0) return null;
  const toggle = (set: Set<string>, update: (s: Set<string>) => void) => (id: string) => {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    update(next);
  };
  const status = job.data?.status;

  return (
    <>
      <Button variant="secondary" onClick={() => setOpen(true)}>
        <Users className="h-4 w-4" aria-hidden="true" />
        {t("promote.button")}
      </Button>
      <Dialog
        open={open}
        onOpenChange={setOpen}
        title={t("promote.title", { name: project.name })}
        description={t("promote.description")}
      >
        <form
          className="flex flex-col gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            promote.mutate();
          }}
        >
          <Field label={t("promote.server")}>
            <Select value={server || list[0]?.name} onChange={(e) => setServer(e.target.value)}>
              {list.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name} ({s.url})
                </option>
              ))}
            </Select>
          </Field>
          <Checklist
            label={t("promote.datasets")}
            items={datasets.map((d) => ({ id: d.id, text: d.id }))}
            selected={dvs}
            onToggle={toggle(dvs, setDvs)}
          />
          <Checklist
            label={t("promote.runs")}
            items={runs.map((r) => ({ id: r.id, text: r.id }))}
            selected={runIds}
            onToggle={toggle(runIds, setRunIds)}
          />
          <p className="text-xs text-muted">{t("promote.hint")}</p>
          <ErrorNote
            error={
              promote.error ??
              (job.data?.error
                ? new Error(String((job.data.error as { message?: string }).message ?? ""))
                : null)
            }
          />
          {status && (
            <p className="text-sm" role="status">
              {t(`promote.status.${status}`, { defaultValue: status })}
            </p>
          )}
          <Button type="submit" loading={promote.isPending || status === "running"}>
            {t("promote.confirm")}
          </Button>
        </form>
      </Dialog>
    </>
  );
}
