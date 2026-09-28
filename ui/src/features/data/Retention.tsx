/** Retención de versiones de datos (RF-MON-07): vista previa y aplicación. */
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, Card, CardTitle, ErrorNote, Field, Input } from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, keys, unwrap } from "@/lib/api/hooks";

type Report = Schemas["RetentionReport"];

export function RetentionCard({ projectId, versions }: { projectId: string; versions: number }) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const [keep, setKeep] = useState(5);
  const run = useMutation({
    mutationFn: async (dry: boolean) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/datasets/retention", {
          params: { path: { project_id: projectId } },
          body: { keep_last: keep, dry_run: dry },
        }),
      ) as Report,
    onSuccess: (r) => {
      if (!r.dry_run) void qc.invalidateQueries({ queryKey: keys.datasets(projectId) });
    },
  });
  const r = run.data;
  if (versions < 2) return null;
  return (
    <Card>
      <CardTitle>{t("retention.title")}</CardTitle>
      <div className="flex flex-wrap items-end gap-2">
        <Field label={t("retention.keep")} hint={t("retention.hint")}>
          <Input
            type="number"
            min={1}
            value={keep}
            onChange={(e) => setKeep(Math.max(1, Number(e.target.value) || 1))}
            className="w-28"
          />
        </Field>
        <Button variant="secondary" loading={run.isPending} onClick={() => run.mutate(true)}>
          {t("retention.preview")}
        </Button>
      </div>
      <ErrorNote error={run.error} />
      {r && (
        <div className="mt-3 space-y-2 text-sm" role="status">
          <p>
            {t(r.dry_run ? "retention.wouldDelete" : "retention.deleted", {
              count: r.deleted.length,
              mb: (r.freed_bytes / 2 ** 20).toFixed(1),
            })}
            {r.in_use.length > 0 && ` ${t("retention.inUse", { count: r.in_use.length })}`}
          </p>
          {r.dry_run && r.deleted.length > 0 && (
            <Button variant="danger" loading={run.isPending} onClick={() => run.mutate(false)}>
              {t("retention.apply", { count: r.deleted.length })}
            </Button>
          )}
        </div>
      )}
    </Card>
  );
}
