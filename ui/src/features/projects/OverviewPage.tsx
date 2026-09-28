import { Link, useNavigate } from "@tanstack/react-router";
import { Archive, ArchiveRestore, ArrowRight, Copy, Package, Trash2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  Dialog,
  ErrorNote,
  Field,
  Input,
  Spinner,
} from "@/components/ui";
import {
  useDatasets,
  useDeleteProject,
  useDuplicateProject,
  useModels,
  useProject,
  useRuns,
  useUpdateProject,
  type Project,
} from "@/lib/api/hooks";
import { downloadFromEngine } from "@/lib/api/download";

import { useProjectId } from "./ProjectLayout";

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <Card className="p-4">
      <p className="text-xs uppercase text-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
    </Card>
  );
}

/** Duplicar, archivar/restaurar y eliminar (RF-PRJ-01). */
function ProjectActions({ project }: { project: Project }) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const update = useUpdateProject(project.id);
  const duplicate = useDuplicateProject(project.id);
  const remove = useDeleteProject(project.id);
  const [confirming, setConfirming] = useState(false);
  const [typed, setTyped] = useState("");
  const [withData, setWithData] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<unknown>(null);
  const archived = project.status === "archived";

  const exportPackage = () => {
    setExporting(true);
    setExportError(null);
    downloadFromEngine(
      `/api/v1/projects/${project.id}/package?include_data=${withData}`,
      `${project.name}.perceptron`,
    )
      .catch(setExportError)
      .finally(() => setExporting(false));
  };

  const toggleArchive = () =>
    update.mutate({ version: project.version, status: archived ? "active" : "archived" });
  const copy = () =>
    duplicate.mutate(undefined, {
      onSuccess: (p) => void navigate({ to: "/projects/$projectId", params: { projectId: p.id } }),
    });
  const destroy = () =>
    remove.mutate(undefined, {
      onSuccess: () => {
        setConfirming(false);
        void navigate({ to: "/" });
      },
    });

  return (
    <Card>
      <CardTitle className="flex items-center gap-2">
        {t("project.actions.title")}
        {archived && <Badge>{t("project.archived")}</Badge>}
      </CardTitle>
      <div className="flex flex-wrap gap-2">
        <Button variant="secondary" loading={duplicate.isPending} onClick={copy}>
          <Copy className="h-4 w-4" aria-hidden="true" />
          {t("project.actions.duplicate")}
        </Button>
        <Button variant="secondary" loading={update.isPending} onClick={toggleArchive}>
          {archived ? (
            <ArchiveRestore className="h-4 w-4" aria-hidden="true" />
          ) : (
            <Archive className="h-4 w-4" aria-hidden="true" />
          )}
          {t(archived ? "project.actions.restore" : "project.actions.archive")}
        </Button>
        <Button variant="secondary" loading={exporting} onClick={exportPackage}>
          <Package className="h-4 w-4" aria-hidden="true" />
          {t("project.actions.export")}
        </Button>
        <label className="flex items-center gap-2 text-sm text-muted">
          <input
            type="checkbox"
            checked={withData}
            onChange={(e) => setWithData(e.target.checked)}
          />
          {t("project.actions.withData")}
        </label>
        <Button variant="danger" onClick={() => setConfirming(true)}>
          <Trash2 className="h-4 w-4" aria-hidden="true" />
          {t("project.actions.delete")}
        </Button>
      </div>
      <p className="mt-2 text-xs text-muted">{t("project.actions.hint")}</p>
      <ErrorNote error={update.error ?? duplicate.error ?? exportError} />
      <Dialog
        open={confirming}
        onOpenChange={(o) => {
          setConfirming(o);
          setTyped("");
        }}
        title={t("project.actions.deleteTitle", { name: project.name })}
        description={t("project.actions.deleteWarning")}
      >
        <form
          className="flex flex-col gap-4"
          onSubmit={(e) => {
            e.preventDefault();
            destroy();
          }}
        >
          <Field label={t("project.actions.typeName", { name: project.name })}>
            <Input value={typed} onChange={(e) => setTyped(e.target.value)} autoComplete="off" />
          </Field>
          <ErrorNote error={remove.error} />
          <Button
            type="submit"
            variant="danger"
            disabled={typed !== project.name}
            loading={remove.isPending}
          >
            {t("project.actions.deleteConfirm")}
          </Button>
        </form>
      </Dialog>
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
      <ProjectActions project={project} />
    </div>
  );
}
