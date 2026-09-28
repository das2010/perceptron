/** Historial de actividad del proyecto (RF-PRJ-05): quién hizo qué y cuándo. */
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import { Card, CardTitle, EmptyState, Spinner } from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap } from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

type Entry = Schemas["ActivityEntry"];

export function ActivityCard({ projectId }: { projectId: string }) {
  const { t, i18n } = useTranslation();
  const activity = useQuery({
    queryKey: ["projects", projectId, "activity"] as const,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/activity", {
          params: { path: { project_id: projectId }, query: { limit: 30 } },
        }),
      ) as Entry[],
  });
  const label = (op: string) => t(`activity.op.${op}`, { defaultValue: op });

  return (
    <Card>
      <CardTitle>{t("activity.title")}</CardTitle>
      {activity.isPending && <Spinner />}
      {activity.data?.length === 0 && <EmptyState>{t("activity.empty")}</EmptyState>}
      {activity.data && activity.data.length > 0 && (
        <ol className="space-y-1 text-sm" aria-label={t("activity.title")}>
          {activity.data.map((e) => (
            <li key={e.id} className="flex flex-wrap items-baseline gap-x-2">
              <span className="text-xs text-muted">{formatDate(e.created_at, i18n.language)}</span>
              <span className="font-semibold">{e.actor ?? t("activity.someone")}</span>
              <span>{label(e.operation)}</span>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}
