/** Monitoreo y ciclo de vida de modelos (RF-MON-01..07). */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getApiClient } from "@/lib/api/client";
import { keys, type Schemas, unwrap } from "@/lib/api/hooks";

export type Deployment = Schemas["Deployment"] & { id: string };
export type DriftReport = Schemas["DriftReport"] & { id: string };
// "Alert" choca con la alerta del perfil de datos: el schema lo nombra por módulo.
export type Alert = Schemas["perceptron__domain__models__Alert"] & { id: string };
export type ChallengeResult = Schemas["ChallengeResult"];

const mk = {
  deployments: (pid: string) => ["projects", pid, "deployments"] as const,
  drift: (did: string) => ["deployments", did, "drift"] as const,
  alerts: (pid: string) => ["projects", pid, "alerts"] as const,
};

export function useDeployments(projectId: string) {
  return useQuery({
    queryKey: mk.deployments(projectId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/deployments", {
          params: { path: { project_id: projectId } },
        }),
      ) as Deployment[],
  });
}

export function useCreateDeployment(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ mv, body }: { mv: string; body: Schemas["DeploymentCreate"] }) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/models/{model_version_id}/deployments", {
          params: { path: { model_version_id: mv } },
          body,
        }),
      ) as Deployment,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: mk.deployments(projectId) });
      void qc.invalidateQueries({ queryKey: keys.models(projectId) });
    },
  });
}

export function useUpdateDeployment(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, patch }: { id: string; patch: Schemas["DeploymentPatch"] }) =>
      unwrap(
        await (
          await getApiClient()
        ).PATCH("/api/v1/deployments/{deployment_id}", {
          params: { path: { deployment_id: id } },
          body: patch,
        }),
      ),
    onSuccess: () => void qc.invalidateQueries({ queryKey: mk.deployments(projectId) }),
  });
}

export function useDriftReports(deploymentId: string | undefined) {
  return useQuery({
    queryKey: mk.drift(deploymentId ?? ""),
    enabled: Boolean(deploymentId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/deployments/{deployment_id}/drift", {
          params: { path: { deployment_id: deploymentId ?? "" } },
        }),
      ) as DriftReport[],
  });
}

export function useCheckDeployment(projectId: string, deploymentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/deployments/{deployment_id}/check", {
          params: { path: { deployment_id: deploymentId }, query: {} },
        }),
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: mk.drift(deploymentId) });
      void qc.invalidateQueries({ queryKey: mk.alerts(projectId) });
    },
  });
}

export function useAlerts(projectId: string) {
  return useQuery({
    queryKey: mk.alerts(projectId),
    refetchInterval: 15_000,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/alerts", {
          params: { path: { project_id: projectId }, query: {} },
        }),
      ) as Alert[],
  });
}

export function useAlertAction(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, action }: { id: string; action: "acknowledge" | "resolve" }) => {
      const api = await getApiClient();
      const params = { params: { path: { alert_id: id } } };
      return unwrap(
        action === "acknowledge"
          ? await api.POST("/api/v1/alerts/{alert_id}/acknowledge", params)
          : await api.POST("/api/v1/alerts/{alert_id}/resolve", params),
      );
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: mk.alerts(projectId) }),
  });
}

function useModelMutation<T>(projectId: string, fn: (id: string) => Promise<T>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.models(projectId) });
      void qc.invalidateQueries({ queryKey: mk.deployments(projectId) });
    },
  });
}

export function usePromote(projectId: string) {
  return useModelMutation(projectId, async (mv: string) =>
    unwrap(
      await (
        await getApiClient()
      ).POST("/api/v1/models/{model_version_id}/promote", {
        params: { path: { model_version_id: mv } },
      }),
    ),
  );
}

export function useRollback(projectId: string) {
  return useModelMutation(projectId, async () =>
    unwrap(
      await (
        await getApiClient()
      ).POST("/api/v1/projects/{project_id}/models/rollback", {
        params: { path: { project_id: projectId } },
      }),
    ),
  );
}

export function useChallenge(projectId: string) {
  return useModelMutation(projectId, async (mv: string) =>
    unwrap(
      await (
        await getApiClient()
      ).POST("/api/v1/models/{model_version_id}/challenge", {
        params: { path: { model_version_id: mv } },
        body: { holdout: "feedback", min_improvement: 0, promote: true },
      }),
    ),
  );
}
