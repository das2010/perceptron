import { Link } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Card, CardTitle, Spinner } from "@/components/ui";
import { useDatasets, useModels, useProject, useRuns } from "@/lib/api/hooks";

import { useProjectId } from "./ProjectLayout";

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <Card className="p-4">
      <p className="text-xs uppercase text-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
    </Card>
  );
}

export function OverviewPage() {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const project = useProject(projectId).data;
  const datasets = useDatasets(projectId).data ?? [];
  const runs = useRuns(projectId).data ?? [];
  const models = useModels(projectId).data ?? [];

  if (!project) return <Spinner />;

  const next =
    datasets.length === 0
      ? { to: "/projects/$projectId/data" as const, label: t("overview.nextData") }
      : runs.length === 0
        ? { to: "/projects/$projectId/train" as const, label: t("overview.nextTrain") }
        : { to: "/projects/$projectId/experiments" as const, label: t("overview.nextResults") };

  return (
    <div className="space-y-6">
      <Card>
        <CardTitle>{t("project.goal")}</CardTitle>
        <p className="text-sm">{project.goal || t("overview.noGoal")}</p>
        <p className="mt-3 text-xs text-muted">{t(`privacy.${project.privacy_level}`)}</p>
      </Card>
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label={t("overview.datasets")} value={datasets.length} />
        <Stat label={t("overview.runs")} value={runs.length} />
        <Stat label={t("overview.models")} value={models.length} />
      </div>
      <Link
        to={next.to}
        params={{ projectId }}
        className="inline-flex items-center gap-2 rounded-pt bg-primary px-4 py-2 text-sm font-semibold text-on-primary"
      >
        {next.label}
        <ArrowRight className="h-4 w-4" aria-hidden="true" />
      </Link>
    </div>
  );
}
