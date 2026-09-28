import { Link } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

import { Badge, Card, CardTitle, EmptyState, ErrorNote, Spinner, Table, Td, Th } from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { useArchSpecs, usePipelines } from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

const ORIGIN_TONE = { llm: "brand", agent: "brand", rules: "neutral", manual: "ok" } as const;

/** Pipelines y arquitecturas del proyecto (reglas, LLM, agente o editadas a mano). */
export function DesignPage() {
  const { t, i18n } = useTranslation();
  const projectId = useProjectId();
  const archspecs = useArchSpecs(projectId);
  const pipelines = usePipelines(projectId);
  const specs = [...(archspecs.data ?? [])].reverse();
  const pipes = [...(pipelines.data ?? [])].reverse();
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardTitle>{t("pipeline.title")}</CardTitle>
        {pipelines.isPending && <Spinner />}
        <ErrorNote error={pipelines.error} />
        {!pipelines.isPending && pipes.length === 0 && <EmptyState>{t("pipeline.empty")}</EmptyState>}
        {pipes.length > 0 && (
          <Table>
            <thead>
              <tr>
                <Th>{t("arch.name")}</Th>
                <Th>{t("arch.origin")}</Th>
                <Th>{t("data.created")}</Th>
              </tr>
            </thead>
            <tbody>
              {pipes.map((p) => (
                <tr key={p.id}>
                  <Td>
                    <Link
                      to="/projects/$projectId/pipelines/$pipelineId"
                      params={{ projectId, pipelineId: p.id ?? "" }}
                      className="font-semibold underline"
                    >
                      {p.name}
                    </Link>{" "}
                    <span className="text-xs text-muted">v{p.version}</span>
                  </Td>
                  <Td>
                    <Badge tone={ORIGIN_TONE[p.origin]}>{t(`arch.originName.${p.origin}`)}</Badge>
                  </Td>
                  <Td className="text-xs text-muted">{formatDate(p.updated_at, i18n.language)}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
      <Card>
        <CardTitle>{t("arch.title")}</CardTitle>
        {archspecs.isPending && <Spinner />}
        <ErrorNote error={archspecs.error} />
        {!archspecs.isPending && specs.length === 0 && <EmptyState>{t("arch.empty")}</EmptyState>}
        {specs.length > 0 && (
          <Table>
            <thead>
              <tr>
                <Th>{t("arch.name")}</Th>
                <Th>{t("arch.origin")}</Th>
                <Th>{t("data.created")}</Th>
              </tr>
            </thead>
            <tbody>
              {specs.map((a) => (
                <tr key={a.id}>
                  <Td>
                    <Link
                      to="/projects/$projectId/archspecs/$archspecId"
                      params={{ projectId, archspecId: a.id }}
                      className="font-semibold underline"
                    >
                      {a.name}
                    </Link>
                  </Td>
                  <Td>
                    <Badge tone={ORIGIN_TONE[a.origin]}>{t(`arch.originName.${a.origin}`)}</Badge>
                  </Td>
                  <Td className="text-xs text-muted">{formatDate(a.created_at, i18n.language)}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}
