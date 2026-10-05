/**
 * Estudios del proyecto con su estado y control (Detener / Reanudar). Un estudio detenido o
 * interrumpido conserva los trials terminados y sigue desde ahí al reanudar.
 */
import { Link } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

import { Badge, Button, Card, CardTitle, ErrorNote, Table, Td, Th } from "@/components/ui";
import { type StudyView, useStudies, useStudyAction } from "@/lib/api/hooks";

import { RunProgress } from "./RunProgress";

const TONE = {
  queued: "neutral",
  running: "brand",
  stopped: "warn",
  interrupted: "bad",
  finished: "ok",
  failed: "bad",
} as const;

export function StudiesCard({ projectId }: { projectId: string }) {
  const { t } = useTranslation();
  const studies = useStudies(projectId);
  const action = useStudyAction(projectId);
  const rows = studies.data ?? [];
  if (rows.length === 0) return null;
  const run = (study: StudyView, kind: "pause" | "resume") =>
    action.mutate({ studyId: study.study.id ?? "", action: kind });

  return (
    <Card>
      <CardTitle>{t("studies.title")}</CardTitle>
      <Table>
        <thead>
          <tr>
            <Th>{t("studies.name")}</Th>
            <Th>{t("studies.status")}</Th>
            <Th>{t("studies.trials")}</Th>
            <Th>{t("studies.best")}</Th>
            <Th />
          </tr>
        </thead>
        <tbody>
          {rows.map((v) => (
            <tr key={v.study.id}>
              <Td className="font-semibold">{v.study.name}</Td>
              <Td>
                <Badge tone={TONE[v.status]}>{t(`studies.state.${v.status}`)}</Badge>
                {v.reason && <p className="mt-1 max-w-md text-xs text-muted">{v.reason}</p>}
                {v.status === "running" && v.job_id && <RunProgress jobId={v.job_id} />}
              </Td>
              <Td>
                {v.trials_total
                  ? t("studies.trialsOf", { done: v.trials_done, total: v.trials_total })
                  : v.trials_done}
              </Td>
              <Td>
                {v.best_run_id ? (
                  <Link
                    to="/projects/$projectId/runs/$runId"
                    params={{ projectId, runId: v.best_run_id }}
                    className="font-mono text-xs underline"
                  >
                    {v.best_run_id.split("-").at(-1)}
                  </Link>
                ) : (
                  "—"
                )}
              </Td>
              <Td className="text-right">
                {(v.status === "running" || v.status === "queued") && (
                  <Button
                    size="sm"
                    variant="secondary"
                    loading={action.isPending}
                    title={t("studies.stopHint")}
                    onClick={() => run(v, "pause")}
                  >
                    {t("studies.stop")}
                  </Button>
                )}
                {v.resumable && (
                  <Button size="sm" loading={action.isPending} onClick={() => run(v, "resume")}>
                    {t("studies.resume")}
                  </Button>
                )}
              </Td>
            </tr>
          ))}
        </tbody>
      </Table>
      <ErrorNote error={action.error} />
    </Card>
  );
}
