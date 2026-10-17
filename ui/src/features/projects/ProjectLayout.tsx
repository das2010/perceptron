import { Link, Outlet, useParams, useRouterState } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

import { Badge, ErrorNote, Spinner } from "@/components/ui";
import { atLeast, projectRole, useSession } from "@/features/auth/session";
import { useProject } from "@/lib/api/hooks";

const TABS = [
  { to: "/projects/$projectId/overview", key: "overview", exact: true },
  { to: "/projects/$projectId/wizard", key: "wizard" },
  { to: "/projects/$projectId/data", key: "data" },
  { to: "/projects/$projectId/labeling", key: "labeling" },
  { to: "/projects/$projectId/design", key: "design" },
  { to: "/projects/$projectId/train", key: "train" },
  { to: "/projects/$projectId/experiments", key: "experiments" },
  { to: "/projects/$projectId/agent", key: "agent" },
  { to: "/projects/$projectId/models", key: "models" },
  { to: "/projects/$projectId/monitoring", key: "monitoring" },
  { to: "/projects/$projectId/audit", key: "audit" },
] as const;

export function useProjectId(): string {
  return useParams({ from: "/projects/$projectId" }).projectId;
}

export function ProjectLayout() {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const { data: project, isPending, error } = useProject(projectId);
  const me = useSession();
  const role = project ? projectRole(me, project) : null;
  // En una herramienta técnica, las opciones avanzadas se muestran abiertas.
  const pathname = useRouterState({ select: (st) => st.location.pathname });
  const advanced = pathname.replace(/\/+$/, "") !== `/projects/${projectId}`;

  return (
    <div>
      <div className="mb-4">
        {isPending && <Spinner />}
        <ErrorNote error={error} />
        {project && (
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold">{project.name}</h1>
            {me && role && (
              <Badge tone={atLeast(role, "editor") ? "neutral" : "warn"}>
                {atLeast(role, "editor") ? t(`roles.${role}`) : t("roles.readOnly")}
              </Badge>
            )}
          </div>
        )}
      </div>
      <Link
        to="/projects/$projectId"
        params={{ projectId }}
        activeOptions={{ exact: true }}
        className="mb-3 inline-block rounded-pt border border-line px-3 py-1 font-semibold data-[status=active]:border-brand data-[status=active]:text-brand"
      >
        {t("journey.assistant")}
      </Link>
      <details className="mb-4" open={advanced}>
        <summary className="cursor-pointer text-sm text-muted">{t("journey.advanced")}</summary>
        {project && (
          <div className="my-2 flex gap-2">
            <Badge tone="brand">{project.privacy_level}</Badge>
            {project.task && <Badge>{t(`task.${project.task}`)}</Badge>}
          </div>
        )}
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
      </details>
      <Outlet />
    </div>
  );
}
