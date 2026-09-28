import { Link, Outlet, useParams } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

import { Badge, ErrorNote, Spinner } from "@/components/ui";
import { useProject } from "@/lib/api/hooks";

const TABS = [
  { to: "/projects/$projectId", key: "overview", exact: true },
  { to: "/projects/$projectId/data", key: "data" },
  { to: "/projects/$projectId/train", key: "train" },
  { to: "/projects/$projectId/experiments", key: "experiments" },
  { to: "/projects/$projectId/models", key: "models" },
  { to: "/projects/$projectId/audit", key: "audit" },
] as const;

export function useProjectId(): string {
  return useParams({ from: "/projects/$projectId" }).projectId;
}

export function ProjectLayout() {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const { data: project, isPending, error } = useProject(projectId);

  return (
    <div>
      <div className="mb-4">
        {isPending && <Spinner />}
        <ErrorNote error={error} />
        {project && (
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold">{project.name}</h1>
            <Badge tone="brand">{project.privacy_level}</Badge>
            {project.task && <Badge>{t(`task.${project.task}`)}</Badge>}
          </div>
        )}
      </div>
      <nav
        aria-label={t("project.sections")}
        className="mb-6 flex flex-wrap gap-1 border-b border-line"
      >
        {TABS.map((tab) => (
          <Link
            key={tab.key}
            to={tab.to}
            params={{ projectId }}
            activeOptions={{ exact: "exact" in tab }}
            className="border-b-2 border-transparent px-3 py-2 text-sm text-muted data-[status=active]:border-brand data-[status=active]:font-semibold data-[status=active]:text-ink"
          >
            {t(`project.tabs.${tab.key}`)}
          </Link>
        ))}
      </nav>
      <Outlet />
    </div>
  );
}
