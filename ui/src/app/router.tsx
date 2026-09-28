import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
  type RouterHistory,
} from "@tanstack/react-router";

import { AdminPage } from "@/features/admin/AdminPage";
import { AgentPage } from "@/features/agent/AgentPage";
import { QueuePage } from "@/features/queue/QueuePage";
import { DesignPage } from "@/features/arch/DesignPage";
import { AuditPage } from "@/features/audit/AuditPage";
import { DataPage } from "@/features/data/DataPage";
import { ExperimentsPage } from "@/features/experiments/ExperimentsPage";
import { HomePage } from "@/features/home/HomePage";
import { ModelsPage } from "@/features/models/ModelsPage";
import { MonitoringPage } from "@/features/monitoring/MonitoringPage";
import { OverviewPage } from "@/features/projects/OverviewPage";
import { ProjectLayout } from "@/features/projects/ProjectLayout";
import { RunPage } from "@/features/runs/RunPage";
import { SettingsPage } from "@/features/settings/SettingsPage";
import { TrainPage } from "@/features/train/TrainPage";
import { WizardPage } from "@/features/wizard/WizardPage";

import { Layout } from "./Layout";

const rootRoute = createRootRoute({ component: Layout });

const homeRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: HomePage });
const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: SettingsPage,
});
const queueRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/queue",
  component: QueuePage,
});
const adminRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/admin",
  component: AdminPage,
});
export const projectRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/projects/$projectId",
  component: ProjectLayout,
});
const overviewRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "/",
  component: OverviewPage,
});
const dataRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "data",
  component: DataPage,
});
const trainRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "train",
  component: TrainPage,
});
const experimentsRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "experiments",
  component: ExperimentsPage,
  validateSearch: (search: Record<string, unknown>): { job?: string } =>
    typeof search.job === "string" ? { job: search.job } : {},
});
const modelsRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "models",
  component: ModelsPage,
});
const monitoringRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "monitoring",
  component: MonitoringPage,
});
const auditRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "audit",
  component: AuditPage,
});
const wizardRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "wizard",
  component: WizardPage,
});
const agentRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "agent",
  component: AgentPage,
  validateSearch: (search: Record<string, unknown>): { agent?: string } =>
    typeof search.agent === "string" ? { agent: search.agent } : {},
});
const labelingRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "labeling",
  component: lazyRouteComponent(() => import("@/features/labeling/LabelingPage"), "LabelingPage"),
});
const designRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "design",
  component: DesignPage,
});
// Los editores traen React Flow (y Monaco bajo demanda): se cargan solo al abrirlos.
const archEditorRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "archspecs/$archspecId",
  component: lazyRouteComponent(() => import("@/features/arch/ArchEditorPage"), "ArchEditorPage"),
});
const expertCodeRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "archspecs/$archspecId/code",
  component: lazyRouteComponent(() => import("@/features/arch/ExpertCodePage"), "ExpertCodePage"),
  validateSearch: (search: Record<string, unknown>): { from?: "current" } =>
    search.from === "current" ? { from: "current" } : {},
});
const pipelineEditorRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "pipelines/$pipelineId",
  component: lazyRouteComponent(
    () => import("@/features/arch/PipelineEditorPage"),
    "PipelineEditorPage",
  ),
});
export const runRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: "runs/$runId",
  component: RunPage,
});

const routeTree = rootRoute.addChildren([
  homeRoute,
  settingsRoute,
  adminRoute,
  queueRoute,
  projectRoute.addChildren([
    overviewRoute,
    dataRoute,
    trainRoute,
    experimentsRoute,
    modelsRoute,
    monitoringRoute,
    auditRoute,
    wizardRoute,
    agentRoute,
    designRoute,
    labelingRoute,
    archEditorRoute,
    expertCodeRoute,
    pipelineEditorRoute,
    runRoute,
  ]),
]);

export function createAppRouter(history?: RouterHistory) {
  return createRouter({ routeTree, defaultPreload: "intent", ...(history ? { history } : {}) });
}

/** Router en memoria para tests. */
export function createTestRouter(path = "/") {
  return createAppRouter(createMemoryHistory({ initialEntries: [path] }));
}

export type AppRouter = ReturnType<typeof createAppRouter>;

declare module "@tanstack/react-router" {
  interface Register {
    router: AppRouter;
  }
}
