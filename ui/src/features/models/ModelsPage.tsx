import { Link } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

import { Badge, Card, CardTitle, EmptyState, ErrorNote, Spinner, Table, Td, Th } from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { useModels } from "@/lib/api/hooks";
import { formatDate, formatNumber } from "@/lib/format";

export function ModelsPage() {
  const { t, i18n } = useTranslation();
  const projectId = useProjectId();
  const { data, isPending, error } = useModels(projectId);
  const models = data ?? [];

  return (
    <Card>
      <CardTitle>{t("models.title")}</CardTitle>
      {isPending && <Spinner />}
      <ErrorNote error={error} />
      {!isPending && models.length === 0 && <EmptyState>{t("models.empty")}</EmptyState>}
      {models.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>{t("models.version")}</Th>
              <Th>{t("models.stage")}</Th>
              <Th>{t("models.metrics")}</Th>
              <Th>{t("models.run")}</Th>
              <Th>{t("data.created")}</Th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const metrics = ((m.model_card ?? {}) as { metrics?: Record<string, number> }).metrics ?? {};
              return (
                <tr key={m.id}>
                  <Td className="font-mono text-xs">{m.id}</Td>
                  <Td>
                    <Badge>{t(`stage.${m.stage}`)}</Badge>
                  </Td>
                  <Td className="text-xs">
                    {Object.entries(metrics)
                      .slice(0, 3)
                      .map(([k, v]) => `${k} ${formatNumber(v, i18n.language, 3)}`)
                      .join(" · ")}
                  </Td>
                  <Td>
                    <Link
                      to="/projects/$projectId/runs/$runId"
                      params={{ projectId, runId: m.run_id }}
                      className="font-mono text-xs underline"
                    >
                      {m.run_id.split("-").at(-1)}
                    </Link>
                  </Td>
                  <Td className="text-xs text-muted">{formatDate(m.created_at, i18n.language)}</Td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      )}
    </Card>
  );
}
